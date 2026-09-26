"""Step 1 - Post-training INT8 quantization of FFNet-40S with AIMET (runs locally).

Simulates W8A8 quantization with AIMET, calibrates on UrbanSyn street scenes,
then measures how closely the INT8 model tracks the FP32 model on held-out images.

    python -m pipeline.quantize_local --calib 20 --test 10
"""

from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch
from PIL import Image

from pipeline.common import (
    AIMET_CONFIG,
    INPUT_SHAPE,
    RESULTS_DIR,
    load_urbansyn,
    logits_to_mask,
    mean_iou,
    overlay,
    preprocess,
)


def save_comparison(image, fp32_mask, int8_mask, path) -> None:
    """FP32 overlay on top, INT8 overlay below, at half resolution."""
    top, bottom = overlay(image, fp32_mask), overlay(image, int8_mask)
    w, h = top.width // 2, top.height // 2
    canvas = Image.new("RGB", (w, h * 2))
    canvas.paste(top.resize((w, h)), (0, 0))
    canvas.paste(bottom.resize((w, h)), (0, h))
    canvas.save(path, quality=88)


def build_quantsim(fp32_model: torch.nn.Module, device: torch.device):
    from aimet_torch.batch_norm_fold import fold_all_batch_norms
    from aimet_torch.model_preparer import prepare_model
    from aimet_torch.quantsim import QuantizationSimModel

    # Fold/prepare on CPU (AIMET builds its dummy inputs there), then move to `device`.
    model = copy.deepcopy(fp32_model).cpu()
    fold_all_batch_norms(model, [INPUT_SHAPE])
    model = prepare_model(model).to(device)
    return QuantizationSimModel(
        model,
        dummy_input=torch.rand(INPUT_SHAPE, device=device),
        quant_scheme="tf_enhanced",
        default_param_bw=8,
        default_output_bw=8,
        config_file=str(AIMET_CONFIG),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calib", type=int, default=20, help="calibration images (< 90)")
    parser.add_argument("--test", type=int, default=10, help="held-out evaluation images")
    args = parser.parse_args()

    from qai_hub_models.models.ffnet_40s import Model as FFNet40S

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = RESULTS_DIR / "local"
    out_dir.mkdir(parents=True, exist_ok=True)

    calib_set, test_set = load_urbansyn(num_test=args.test)
    # FP32 reference stays on CPU: the wrapper builds its normalization constants there.
    fp32 = FFNet40S.from_pretrained().eval()

    print(f"Building AIMET QuantizationSim (W8A8, tf_enhanced) on {device} ...")
    sim = build_quantsim(fp32, device)

    def calibrate(sim_model: torch.nn.Module, _=None) -> None:
        with torch.no_grad():
            for sample in calib_set.select(range(args.calib)):
                sim_model(preprocess(sample["image"]).to(device))

    t0 = time.perf_counter()
    sim.compute_encodings(calibrate)
    print(f"Calibrated on {args.calib} images in {time.perf_counter() - t0:.1f}s")

    agreements, mious = [], []
    with torch.no_grad():
        for i, sample in enumerate(test_set):
            x = preprocess(sample["image"])
            ref = logits_to_mask(fp32(x))
            q = logits_to_mask(sim.model(x.to(device)))
            agreements.append(float((ref == q).mean()))
            mious.append(mean_iou(q, ref))
            if i < 3:
                save_comparison(sample["image"], ref, q, out_dir / f"sample{i}_fp32_vs_int8.jpg")

    summary = {
        "calibration_images": args.calib,
        "test_images": len(test_set),
        "pixel_agreement_int8_vs_fp32": float(np.mean(agreements)),
        "miou_int8_vs_fp32": float(np.nanmean(mious)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Overlays written to {out_dir}")


if __name__ == "__main__":
    main()
