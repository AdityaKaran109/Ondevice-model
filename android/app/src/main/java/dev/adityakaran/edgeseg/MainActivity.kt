package dev.adityakaran.edgeseg

import android.Manifest
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.graphics.Rect
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.text.SpannableStringBuilder
import android.text.style.ForegroundColorSpan
import android.util.Size
import android.view.View
import android.view.WindowManager
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import dev.adityakaran.edgeseg.databinding.ActivityMainBinding
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.ceil

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    /** All model work runs on this single thread; a LiteRT Interpreter is not thread-safe. */
    private val worker: ExecutorService = Executors.newSingleThreadExecutor()

    // Owned by `worker`. One segmenter per orientation, created lazily on the current backend.
    private var backend = Backend.NPU
    private val segmenters = mutableMapOf<Boolean, Segmenter>() // key: portrait?
    private var hasPortraitModel = false
    private val fitters = mutableMapOf<Boolean, FrameFitter>()
    private var lastInput: Bitmap? = null
    private var lastStill: Bitmap? = null
    private var smoothed = Smoothed()

    // What is on screen, kept so it survives a rotation (the layout is re-inflated). UI thread only.
    private var shownFrame: Bitmap? = null
    private var shownMask: Bitmap? = null
    private var shownStats: CharSequence = ""
    private var shownLegend: CharSequence = ""
    private var shownMessage: CharSequence? = null
    private var buttonsEnabled = false

    private var cameraProvider: ProcessCameraProvider? = null
    private var analysis: ImageAnalysis? = null
    @Volatile private var cameraRunning = false
    @Volatile private var busy = false

    private val requestCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else showMessage(getString(R.string.camera_denied))
    }

    private val pickPhoto = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let { segmentStill { decodeUri(it) } }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setupUi()

        if (!Segmenter.hasModel(this)) {
            showMessage(getString(R.string.missing_model))
            return
        }
        worker.execute {
            hasPortraitModel = Segmenter.hasModel(this, Segmenter.PORTRAIT_ASSET)
            val seg = runCatching { Segmenter.createBest(this) }.getOrElse { e ->
                showMessage("Could not load the model: ${e.message}")
                return@execute
            }
            backend = seg.backend
            segmenters[false] = seg
            runOnUiThread {
                setBackendLabel()
                setButtonsEnabled(true)
                ensureCameraPermission()
            }
        }
    }

    /** The activity handles rotation itself so the NPU models stay loaded; just swap layouts. */
    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        setupUi()
        analysis?.targetRotation = displayRotation()
    }

    private fun setupUi() {
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        enterImmersiveMode()

        binding.cameraButton.setOnClickListener { ensureCameraPermission() }
        binding.photoButton.setOnClickListener {
            stopCamera()
            pickPhoto.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        binding.sampleButton.setOnClickListener { segmentStill { decodeAsset("sample.jpg") } }
        binding.backendButton.setOnClickListener { cycleBackend() }
        binding.benchmarkButton.setOnClickListener { runBenchmark() }
        binding.messageView.setOnClickListener { hideMessage() }

        binding.frameView.setImageBitmap(shownFrame)
        binding.maskView.setImageBitmap(shownMask)
        binding.statsView.text = shownStats
        binding.legendView.text = shownLegend
        binding.messageView.text = shownMessage
        binding.messageView.visibility = if (shownMessage != null) View.VISIBLE else View.GONE
        setBackendLabel()
        setButtonsEnabled(buttonsEnabled)
    }

    /** Full-screen: hide status/navigation bars, keep clear of the camera cutout. */
    private fun enterImmersiveMode() {
        WindowCompat.getInsetsController(window, binding.root).apply {
            hide(WindowInsetsCompat.Type.systemBars())
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { v, insets ->
            val cutout = insets.getInsets(WindowInsetsCompat.Type.displayCutout())
            v.setPadding(cutout.left, cutout.top, cutout.right, cutout.bottom)
            insets
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraProvider?.unbindAll()
        worker.execute { closeSegmenters() }
        worker.shutdown()
    }

    // region Camera

    private fun ensureCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestCamera.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        hideMessage()
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get().also { cameraProvider = it }
            val useCase = ImageAnalysis.Builder()
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                        .setResolutionStrategy(
                            ResolutionStrategy(
                                Size(1920, 1080),
                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                            )
                        )
                        .build()
                )
                .setTargetRotation(displayRotation())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            useCase.setAnalyzer(worker) { image -> analyzeFrame(image) }
            analysis = useCase
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, useCase)
            cameraRunning = true
            smoothed = Smoothed()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCamera() {
        cameraRunning = false
        cameraProvider?.unbindAll()
    }

    private fun displayRotation(): Int = display?.rotation ?: 0

    private fun analyzeFrame(image: ImageProxy) {
        image.use {
            if (!cameraRunning || busy) return
            process(it.toBitmap(), it.imageInfo.rotationDegrees, live = true)
        }
    }

    // endregion

    // region Stills

    private fun segmentStill(load: () -> Bitmap) {
        stopCamera()
        worker.execute {
            val bitmap = runCatching(load).getOrElse { e ->
                showMessage("Could not open image: ${e.message}")
                return@execute
            }
            hideMessage()
            smoothed = Smoothed()
            lastStill = bitmap
            process(bitmap, rotation = 0, live = false)
        }
    }

    private fun decodeUri(uri: Uri): Bitmap =
        ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver, uri)) { decoder, info, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            val scale = 2048f / maxOf(info.size.width, info.size.height)
            if (scale < 1f) decoder.setTargetSize((info.size.width * scale).toInt(), (info.size.height * scale).toInt())
        }

    private fun decodeAsset(name: String): Bitmap =
        assets.open(name).use { BitmapFactory.decodeStream(it) ?: error("$name is not an image") }

    // endregion

    /** Runs on `worker`. Picks the portrait or landscape model to match the (rotated) source. */
    private fun process(source: Bitmap, rotation: Int, live: Boolean) {
        val swap = rotation % 180 != 0
        val portrait = (if (swap) source.width else source.height) > (if (swap) source.height else source.width)
        val seg = segmenterFor(portrait) ?: return
        val fit = fitters.getOrPut(portrait) { FrameFitter(seg.inputWidth, seg.inputHeight) }

        val t0 = System.nanoTime()
        // Photos keep the whole image (letterboxed); the camera fills the frame.
        val input = fit.fit(source, rotation, letterbox = !live)
        val fitMs = (System.nanoTime() - t0) / 1e6
        val result = seg.segment(input)

        // Show only the real image area, and the matching part of the mask.
        val content = Rect(fit.content)
        val display = Bitmap.createBitmap(
            input, content.left, content.top, content.width(), content.height(),
            Matrix().apply { setScale(0.5f, 0.5f) }, true,
        )
        val sx = result.maskWidth / seg.inputWidth.toFloat()
        val sy = result.maskHeight / seg.inputHeight.toFloat()
        val maskRegion = Rect(
            (content.left * sx).toInt(), (content.top * sy).toInt(),
            ceil(content.right * sx).toInt().coerceAtMost(result.maskWidth),
            ceil(content.bottom * sy).toInt().coerceAtMost(result.maskHeight),
        )
        val mask = Cityscapes.colorize(result, maskRegion)
        val coverage = Cityscapes.coverage(result, maskRegion)
        lastInput = input
        smoothed.update(fitMs + result.preprocessMs, result.inferenceMs, result.postprocessMs)

        val stats = buildString {
            appendLine(seg.backend.label)
            appendLine("${Build.MODEL} · ${Build.SOC_MODEL}")
            appendLine("FFNet-40S INT8 · ${seg.inputWidth}×${seg.inputHeight}")
            appendLine(String.format(Locale.US, "inference %6.1f ms", smoothed.inference))
            appendLine(String.format(Locale.US, "pre       %6.1f ms", smoothed.pre))
            append(String.format(Locale.US, "post      %6.1f ms", smoothed.post))
            if (live) append(String.format(Locale.US, "\n%.1f fps", smoothed.fps))
        }
        val legend = legend(coverage)
        runOnUiThread {
            shownFrame = display; shownMask = mask; shownStats = stats; shownLegend = legend
            binding.frameView.setImageBitmap(display)
            binding.maskView.setImageBitmap(mask)
            binding.statsView.text = stats
            binding.legendView.text = legend
        }
    }

    /** Runs on `worker`. Falls back to the landscape model (letterboxed) if there is no portrait one. */
    private fun segmenterFor(portrait: Boolean): Segmenter? {
        val usePortrait = portrait && hasPortraitModel
        segmenters[usePortrait]?.let { return it }
        val asset = if (usePortrait) Segmenter.PORTRAIT_ASSET else Segmenter.LANDSCAPE_ASSET
        val seg = runCatching { Segmenter.create(this, backend, asset) }
            .getOrElse { runCatching { Segmenter.createBest(this, asset) }.getOrNull() }
            ?: return null
        segmenters[usePortrait] = seg
        return seg
    }

    private fun closeSegmenters() {
        segmenters.values.forEach { it.close() }
        segmenters.clear()
    }

    private fun legend(coverage: List<Pair<Int, Float>>): CharSequence {
        val sb = SpannableStringBuilder()
        coverage.forEachIndexed { i, (cls, share) ->
            if (i > 0) sb.append(if (i % 3 == 0) "\n" else "   ")
            val start = sb.length
            sb.append("■")
            sb.setSpan(ForegroundColorSpan(Cityscapes.COLORS[cls]), start, sb.length, 0)
            sb.append(String.format(Locale.US, " %s %d%%", Cityscapes.NAMES[cls], (share * 100).toInt()))
        }
        return sb
    }

    // region Backends & benchmark

    private fun cycleBackend() {
        setButtonsEnabled(false)
        worker.execute {
            val order = Backend.entries
            closeSegmenters()
            for (step in 1..order.size) {
                val candidate = order[(backend.ordinal + step) % order.size]
                val seg = runCatching { Segmenter.create(this, candidate) }.getOrNull() ?: continue
                backend = candidate
                segmenters[false] = seg
                break
            }
            smoothed = Smoothed()
            lastStill?.let { if (!cameraRunning) process(it, 0, live = false) }
            runOnUiThread {
                setBackendLabel()
                setButtonsEnabled(true)
            }
        }
    }

    private fun runBenchmark() {
        val input = lastInput ?: return showMessage("Point the camera at a scene or open a photo first.")
        stopCamera()
        setButtonsEnabled(false)
        showMessage("Benchmarking NPU, GPU and CPU…")
        busy = true
        worker.execute {
            closeSegmenters()
            val asset = if (input.height > input.width) Segmenter.PORTRAIT_ASSET else Segmenter.LANDSCAPE_ASSET

            val rows = Backend.entries.map { b ->
                showMessage("Benchmarking ${b.label}…")
                val times = runCatching {
                    Segmenter.create(this, b, asset).use { seg ->
                        if (b == Backend.CPU) seg.benchmark(input, warmup = 2, runs = 10)
                        else seg.benchmark(input, warmup = 5, runs = 30)
                    }
                }.getOrNull()
                b to times
            }

            val report = buildString {
                appendLine("Benchmark · FFNet-40S INT8 · ${input.width}×${input.height}")
                appendLine("${Build.MANUFACTURER} ${Build.MODEL} · ${Build.SOC_MODEL}")
                appendLine()
                appendLine(String.format(Locale.US, "%-22s %9s %9s", "backend", "median", "p90"))
                for ((b, t) in rows) {
                    if (t == null) {
                        appendLine(String.format(Locale.US, "%-22s %19s", b.label, "unavailable"))
                    } else {
                        appendLine(
                            String.format(
                                Locale.US, "%-22s %6.1f ms %6.1f ms",
                                b.label, t[t.size / 2], t[(t.size * 9) / 10],
                            )
                        )
                    }
                }
                val npu = rows.first { it.first == Backend.NPU }.second
                val cpu = rows.first { it.first == Backend.CPU }.second
                if (npu != null && cpu != null) {
                    appendLine()
                    append(String.format(Locale.US, "NPU is %.1f× faster than CPU", cpu[cpu.size / 2] / npu[npu.size / 2]))
                }
            }

            busy = false
            showMessage(report)
            runOnUiThread { setButtonsEnabled(true) }
        }
    }

    private fun setBackendLabel() {
        binding.backendButton.text = backend.name
    }

    // endregion

    private fun setButtonsEnabled(enabled: Boolean) = runOnUiThread {
        buttonsEnabled = enabled
        listOf(
            binding.cameraButton, binding.photoButton, binding.sampleButton,
            binding.backendButton, binding.benchmarkButton,
        ).forEach { it.isEnabled = enabled }
    }

    private fun showMessage(text: String) = runOnUiThread {
        shownMessage = text
        binding.messageView.text = text
        binding.messageView.visibility = View.VISIBLE
    }

    private fun hideMessage() = runOnUiThread {
        shownMessage = null
        binding.messageView.visibility = View.GONE
    }

    /** Exponential moving averages so the on-screen numbers are readable. */
    private class Smoothed {
        var pre = 0.0; var inference = 0.0; var post = 0.0; var fps = 0.0
        private var lastFrameNs = 0L
        private var first = true

        fun update(preMs: Double, inferenceMs: Double, postMs: Double) {
            val now = System.nanoTime()
            val a = if (first) 1.0 else 0.1
            pre += a * (preMs - pre)
            inference += a * (inferenceMs - inference)
            post += a * (postMs - post)
            if (lastFrameNs != 0L) fps += (if (fps == 0.0) 1.0 else 0.1) * (1e9 / (now - lastFrameNs) - fps)
            lastFrameNs = now
            first = false
        }
    }
}
