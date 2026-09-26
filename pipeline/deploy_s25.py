"""Step 2 - Quantize, compile and benchmark FFNet-40S for the Galaxy S25 Ultra via Qualcomm AI Hub.

Pipeline (each step is a cloud job you can inspect at https://app.aihub.qualcomm.com):
  1. Trace the PyTorch model (normalization is baked in, input = RGB in [0, 1]).
  2. Compile to ONNX, then quantize to W8A8 using UrbanSyn calibration images.
  3. Compile the INT8 model to TFLite (NHWC, uint8 I/O) for the Snapdragon 8 Elite NPU.
  4. Profile on a real, hosted Galaxy S25 Ultra (INT8, plus an FP32 baseline).
  5. Run on-device inference on held-out images and compare with PyTorch FP32.
  6. Download the .tflite + a sample image into the Android app's assets.

One-time setup:  qai-hub configure --api_token <YOUR_TOKEN>

    python -m pipeline.deploy_s25 --calib 20
    python -m pipeline.deploy_s25 --portrait --skip-fp32   # upright 1024x2048 variant
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter

import numpy as np
import qai_hub as hub
import torch

from pipeline.common import (
    ANDROID_ASSETS_DIR,
    HEIGHT,
    PORTRAIT_TFLITE_NAME,
    RESULTS_DIR,
    TARGET_DEVICE,
    TFLITE_NAME,
    WIDTH,
    load_urbansyn,
    logits_to_mask,
    mean_iou,
    overlay,
    preprocess,
)

# Same options qai_hub_models recommends for FFNet W8A8 TFLite: channel-last tensors
# and uint8 input/output so the phone can feed camera bytes straight in.
TFLITE_OPTIONS = (
    "--target_runtime tflite --output_names mask "
    "--force_channel_last_input image --force_channel_last_output mask "
    "--quantize_io --quantize_io_type uint8"
)
TFLITE_FP32_OPTIONS = (
    "--target_runtime tflite --output_names mask "
    "--force_channel_last_input image --force_channel_last_output mask"
)


def summarize_profile(job: hub.ProfileJob) -> dict:
    profile = job.download_profile()
    summary = profile["execution_summary"]
    units = Counter(op.get("compute_unit", "UNK").upper() for op in profile["execution_detail"])
    lo, hi = summary["inference_memory_peak_range"]
    return {
        "job_url": job.url,
        "inference_ms": summary["estimated_inference_time"] / 1000,
        "peak_memory_mb": [round(lo / 2**20, 1), round(hi / 2**20, 1)],
        "layers_by_compute_unit": dict(units),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=TARGET_DEVICE, help="AI Hub device name (qai-hub list-devices)")
    parser.add_argument("--calib", type=int, default=20, help="calibration images for the quantize job")
    parser.add_argument("--test", type=int, default=5, help="held-out images for the on-device accuracy check")
    parser.add_argument("--skip-fp32", action="store_true", help="skip the FP32 baseline profile")
    parser.add_argument("--portrait", action="store_true", help="compile for upright 1024x2048 (WxH) frames")
    args = parser.parse_args()

    # FFNet is fully convolutional, so the same weights compile for either orientation.
    size = (HEIGHT, WIDTH) if args.portrait else (WIDTH, HEIGHT)  # (width, height)
    input_shape = (1, 3, size[1], size[0])
    input_specs = {"image": (input_shape, "float32")}
    tflite_name = PORTRAIT_TFLITE_NAME if args.portrait else TFLITE_NAME
    tag = "_portrait" if args.portrait else ""

    from qai_hub_models.models.ffnet_40s import Model as FFNet40S

    device = hub.Device(args.device)
    out_dir = RESULTS_DIR / ("s25_ultra" + tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_set, test_set = load_urbansyn(num_test=args.test)

    # 1. Trace
    model = FFNet40S.from_pretrained().eval()
    traced = torch.jit.trace(model, torch.rand(input_shape))

    # 2. ONNX -> W8A8 quantize
    onnx_job = hub.submit_compile_job(
        traced, device, name=f"ffnet40s{tag}_onnx", input_specs=input_specs,
        options="--target_runtime onnx --output_names mask",
    )
    calibration = {
        "image": [preprocess(s["image"], size).numpy() for s in calib_set.select(range(args.calib))]
    }
    quant_job = hub.submit_quantize_job(
        onnx_job.get_target_model(), calibration,
        weights_dtype=hub.QuantizeDtype.INT8, activations_dtype=hub.QuantizeDtype.INT8,
        name=f"ffnet40s{tag}_w8a8",
    )

    # 3. Compile INT8 TFLite (+ optional FP32 baseline, compiled from the traced model)
    int8_job = hub.submit_compile_job(
        quant_job.get_target_model(), device, name=f"ffnet40s{tag}_w8a8_tflite", options=TFLITE_OPTIONS
    )
    fp32_job = None
    if not args.skip_fp32:
        fp32_job = hub.submit_compile_job(
            traced, device, name=f"ffnet40s{tag}_fp32_tflite", input_specs=input_specs, options=TFLITE_FP32_OPTIONS
        )

    int8_model = int8_job.get_target_model()
    assert int8_model is not None, f"INT8 compile failed: {int8_job.url}"

    # 4. Profile on a real S25 Ultra
    report: dict = {"device": args.device, "input_wh": list(size), "calibration_images": args.calib}
    report["int8"] = summarize_profile(hub.submit_profile_job(int8_model, device, name=f"ffnet40s{tag}_w8a8_profile"))
    if fp32_job is not None and (fp32_model := fp32_job.get_target_model()) is not None:
        report["fp32"] = summarize_profile(hub.submit_profile_job(fp32_model, device, name=f"ffnet40s{tag}_fp32_profile"))
        report["speedup_int8_vs_fp32"] = round(report["fp32"]["inference_ms"] / report["int8"]["inference_ms"], 2)

    # 5. On-device accuracy vs PyTorch FP32
    inputs = [preprocess(s["image"], size) for s in test_set]
    inference_job = hub.submit_inference_job(
        int8_model, device, name=f"ffnet40s{tag}_w8a8_inference",
        inputs={"image": [x.permute(0, 2, 3, 1).numpy() for x in inputs]},  # NHWC
    )
    device_outputs = inference_job.download_output_data()["mask"]
    agreements, mious = [], []
    with torch.no_grad():
        for i, (x, y_dev) in enumerate(zip(inputs, device_outputs)):
            ref = logits_to_mask(model(x))
            dev = logits_to_mask(np.asarray(y_dev), channel_last=True)  # argmax is scale-invariant
            agreements.append(float((ref == dev).mean()))
            mious.append(mean_iou(dev, ref))
            if i < 3:
                overlay(test_set[i]["image"], dev).save(out_dir / f"device_sample{i}.jpg", quality=90)
    report["on_device_accuracy"] = {
        "job_url": inference_job.url,
        "test_images": len(inputs),
        "pixel_agreement_vs_fp32": float(np.mean(agreements)),
        "miou_vs_fp32": float(np.nanmean(mious)),
    }

    # 6. Ship to the Android app
    tflite_path = out_dir / tflite_name
    int8_model.download(str(tflite_path))
    ANDROID_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(tflite_path, ANDROID_ASSETS_DIR / tflite_name)
    if not args.portrait:
        test_set[0]["image"].convert("RGB").save(ANDROID_ASSETS_DIR / "sample.jpg", quality=92)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nModel copied to {ANDROID_ASSETS_DIR / tflite_name} - open android/ in Android Studio and run.")


if __name__ == "__main__":
    main()
