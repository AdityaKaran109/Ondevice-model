package dev.adityakaran.edgeseg

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.qualcomm.qti.QnnDelegate
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.roundToInt

enum class Backend(val label: String) {
    NPU("NPU · Hexagon (QNN)"),
    GPU("GPU · Adreno (LiteRT)"),
    CPU("CPU · XNNPACK"),
}

/**
 * FFNet-40S Cityscapes segmentation on LiteRT.
 *
 * Handles both the uint8-I/O INT8 model produced by `pipeline/deploy_s25.py`
 * and float models, in NHWC or NCHW output layout.
 */
class Segmenter private constructor(
    val backend: Backend,
    private val interpreter: Interpreter,
    private val delegate: Delegate?,
) : Closeable {

    class Result(
        /** Class id per output pixel, row-major, [maskHeight * maskWidth]. */
        val mask: ByteArray,
        val maskWidth: Int,
        val maskHeight: Int,
        val preprocessMs: Double,
        val inferenceMs: Double,
        val postprocessMs: Double,
    )

    val inputHeight: Int
    val inputWidth: Int

    private val inputType: DataType
    private val inputBuffer: ByteBuffer
    private val inputLut = ByteArray(256) // 8-bit pixel -> quantized input byte
    private val pixels: IntArray
    private val inputBytes: ByteArray   // staging for quantized input, copied in bulk
    private val inputFloats: FloatArray // staging for float input

    private val outputType: DataType
    private val outputBuffer: ByteBuffer
    private val outputBytes: ByteArray
    private val outputChannelLast: Boolean
    private val outH: Int
    private val outW: Int
    private val numClasses: Int

    init {
        val inTensor = interpreter.getInputTensor(0)
        val inShape = inTensor.shape() // [1, H, W, 3]
        inputHeight = inShape[1]
        inputWidth = inShape[2]
        inputType = inTensor.dataType()
        inputBuffer = ByteBuffer.allocateDirect(inTensor.numBytes()).order(ByteOrder.nativeOrder())
        pixels = IntArray(inputHeight * inputWidth)
        val floatInput = inputType == DataType.FLOAT32
        inputBytes = ByteArray(if (floatInput) 0 else pixels.size * 3)
        inputFloats = FloatArray(if (floatInput) pixels.size * 3 else 0)

        if (inputType == DataType.UINT8 || inputType == DataType.INT8) {
            val q = inTensor.quantizationParams()
            val (lo, hi) = if (inputType == DataType.UINT8) 0 to 255 else -128 to 127
            for (v in 0..255) {
                val quant = ((v / 255f) / q.scale).roundToInt() + q.zeroPoint
                inputLut[v] = quant.coerceIn(lo, hi).toByte()
            }
        }

        val outTensor = interpreter.getOutputTensor(0)
        val outShape = outTensor.shape()
        outputType = outTensor.dataType()
        outputChannelLast = outShape[3] == Cityscapes.NUM_CLASSES || outShape[1] != Cityscapes.NUM_CLASSES
        if (outputChannelLast) {
            outH = outShape[1]; outW = outShape[2]; numClasses = outShape[3]
        } else {
            numClasses = outShape[1]; outH = outShape[2]; outW = outShape[3]
        }
        outputBuffer = ByteBuffer.allocateDirect(outTensor.numBytes()).order(ByteOrder.nativeOrder())
        outputBytes = ByteArray(outTensor.numBytes())
        Log.i(TAG, "Loaded on $backend: in=${inShape.contentToString()} $inputType, " +
            "out=${outShape.contentToString()} $outputType")
    }

    /** [input] must already be [inputWidth] x [inputHeight]. */
    fun segment(input: Bitmap): Result {
        val t0 = System.nanoTime()
        fillInput(input)
        val t1 = System.nanoTime()
        runModel()
        val t2 = System.nanoTime()
        val mask = argmax()
        val t3 = System.nanoTime()
        return Result(mask, outW, outH, ms(t0, t1), ms(t1, t2), ms(t2, t3))
    }

    /** Times [runs] inferences (after [warmup]) on an already-filled input. Returns sorted latencies in ms. */
    fun benchmark(input: Bitmap, warmup: Int = 5, runs: Int = 30): DoubleArray {
        fillInput(input)
        repeat(warmup) { runModel() }
        return DoubleArray(runs) {
            val t = System.nanoTime(); runModel(); ms(t, System.nanoTime())
        }.also { it.sort() }
    }

    private fun runModel() {
        inputBuffer.rewind()
        outputBuffer.rewind()
        interpreter.run(inputBuffer, outputBuffer)
    }

    private fun fillInput(bitmap: Bitmap) {
        require(bitmap.width == inputWidth && bitmap.height == inputHeight)
        bitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        inputBuffer.rewind()
        // Fill a plain array, then copy once: per-element ByteBuffer.put is ~10x slower.
        if (inputType == DataType.FLOAT32) {
            val f = inputFloats
            var j = 0
            for (p in pixels) {
                f[j] = ((p shr 16) and 0xFF) / 255f
                f[j + 1] = ((p shr 8) and 0xFF) / 255f
                f[j + 2] = (p and 0xFF) / 255f
                j += 3
            }
            inputBuffer.asFloatBuffer().put(f)
        } else {
            val lut = inputLut
            val b = inputBytes
            var j = 0
            for (p in pixels) {
                b[j] = lut[(p shr 16) and 0xFF]
                b[j + 1] = lut[(p shr 8) and 0xFF]
                b[j + 2] = lut[p and 0xFF]
                j += 3
            }
            inputBuffer.put(b)
        }
    }

    // Argmax of quantized logits == argmax of dequantized logits (scale > 0), so no dequantization needed.
    private fun argmax(): ByteArray {
        val n = outH * outW
        val mask = ByteArray(n)
        outputBuffer.rewind()
        when (outputType) {
            DataType.FLOAT32 -> {
                val f = outputBuffer.asFloatBuffer()
                for (i in 0 until n) {
                    var best = 0
                    var bestV = Float.NEGATIVE_INFINITY
                    for (c in 0 until numClasses) {
                        val v = f.get(index(i, c, n))
                        if (v > bestV) { bestV = v; best = c }
                    }
                    mask[i] = best.toByte()
                }
            }
            else -> {
                outputBuffer.get(outputBytes)
                val signed = outputType == DataType.INT8
                for (i in 0 until n) {
                    var best = 0
                    var bestV = Int.MIN_VALUE
                    for (c in 0 until numClasses) {
                        val b = outputBytes[index(i, c, n)].toInt()
                        val v = if (signed) b else b and 0xFF
                        if (v > bestV) { bestV = v; best = c }
                    }
                    mask[i] = best.toByte()
                }
            }
        }
        return mask
    }

    private fun index(pixel: Int, c: Int, n: Int) =
        if (outputChannelLast) pixel * numClasses + c else c * n + pixel

    override fun close() {
        interpreter.close()
        (delegate as? Closeable)?.close()
    }

    companion object {
        private const val TAG = "Segmenter"
        /** 2048x1024 (W x H) model for landscape frames. */
        const val LANDSCAPE_ASSET = "ffnet_40s_w8a8.tflite"
        /** Same weights compiled for 1024x2048 (W x H) upright frames. */
        const val PORTRAIT_ASSET = "ffnet_40s_w8a8_portrait.tflite"

        private fun ms(from: Long, to: Long) = (to - from) / 1e6

        fun hasModel(context: Context, asset: String = LANDSCAPE_ASSET) =
            runCatching { context.assets.openFd(asset).close() }.isSuccess

        /** Creates a segmenter on [backend]; throws if that backend can't run on this device. */
        fun create(context: Context, backend: Backend, asset: String = LANDSCAPE_ASSET): Segmenter {
            val model = loadModel(context, asset)
            val options = Interpreter.Options()
            val delegate: Delegate? = when (backend) {
                Backend.NPU -> qnnHtpDelegate(context, asset)
                Backend.GPU -> {
                    check(CompatibilityList().isDelegateSupportedOnThisDevice) { "GPU delegate unsupported" }
                    GpuDelegate()
                }
                Backend.CPU -> null.also { options.setNumThreads(4) }
            }
            delegate?.let { options.addDelegate(it) }
            return try {
                Segmenter(backend, Interpreter(model, options), delegate)
            } catch (e: Exception) {
                (delegate as? Closeable)?.close()
                throw e
            }
        }

        /** NPU first, then GPU, then CPU. */
        fun createBest(context: Context, asset: String = LANDSCAPE_ASSET): Segmenter {
            for (backend in Backend.entries) {
                try {
                    return create(context, backend, asset)
                } catch (e: Throwable) {
                    Log.w(TAG, "$backend unavailable: ${e.message}")
                }
            }
            error("No backend could load the model")
        }

        private fun qnnHtpDelegate(context: Context, asset: String): Delegate {
            check(QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_QUANTIZED)) {
                "Hexagon NPU (HTP) not available"
            }
            val opts = QnnDelegate.Options().apply {
                setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
                setSkelLibraryDir(context.applicationInfo.nativeLibraryDir)
                setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_QUANTIZED)
                setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST)
                setHtpUseConvHmx(QnnDelegate.Options.HtpUseConvHmx.HTP_CONV_HMX_ON)
                setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_WARN)
                // Cache the compiled NPU graph so later launches start fast.
                setCacheDir(context.cacheDir.absolutePath)
                setModelToken(asset.substringBefore(".tflite") + "_v1")
            }
            return QnnDelegate(opts)
        }

        private fun loadModel(context: Context, asset: String): MappedByteBuffer =
            context.assets.openFd(asset).use { fd ->
                FileInputStream(fd.fileDescriptor).channel.use {
                    it.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                }
            }
    }
}
