# LinkedIn post (draft)

Every number below was measured on my own Galaxy S25 Ultra: the AI Hub profile is in `results/s25_ultra/report.json` and the in-app benchmark is in `docs/s25_ultra_benchmark.jpg`. Only the GitHub link still needs filling in.

Attach `docs/s25_ultra_segmentation.jpg` and `docs/s25_ultra_benchmark.jpg`, and ideally a short screen recording of the app pointed at a street.

---

I got a 14M-parameter semantic-segmentation model running live on my Samsung Galaxy S25 Ultra, fully on-device, in under 6 ms per inference on the NPU. 📱⚡

What I built — EdgeSeg:
🔹 Took FFNet-40S (Cityscapes, 19 classes, 2048×1024 input)
🔹 Quantized it to INT8: 99.1% pixel agreement with FP32, measured on the phone's NPU
🔹 Compiled it for the Snapdragon 8 Elite Hexagon NPU with Qualcomm AI Hub and profiled it on a real S25 Ultra: 3.99 ms INT8 vs 18.07 ms FP32 (4.5× faster), with all 99 layers on the NPU
🔹 Wrote an Android app (Kotlin, CameraX, LiteRT + Qualcomm QNN delegate) that segments the camera feed in real time and benchmarks the three processors on the phone itself:
    NPU 5.7 ms · GPU 62.5 ms · CPU 226.8 ms. The NPU is ~40× faster than the CPU.

Things I learned along the way:
✅ Bake preprocessing into the model: uint8 camera pixels go straight in
✅ Skip dequantization: argmax over quantized logits gives the same answer
✅ Once the NPU gets fast, the bottleneck moves to Kotlin: a bulk array copy cut preprocessing from 121 ms to 17 ms
✅ Tooling moves fast: I had to fix API changes in qai_hub_models and AIMET 2.x before anything ran

No cloud and no network at inference time, so the data never leaves the phone.

Code, pipeline and results 👉 [GitHub link]

#OnDeviceAI #EdgeAI #Qualcomm #Snapdragon #Android #ComputerVision #Quantization #MachineLearning
