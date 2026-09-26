package dev.adityakaran.edgeseg

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Rotates and scales any frame into a reusable [width] x [height] bitmap (the model input)
 * in a single draw call.
 *
 * - `letterbox = false` (camera): fill the input, cropping whatever sticks out.
 * - `letterbox = true` (photos): keep the whole image, padding with black bars.
 *
 * After each call, [content] is the part of the input that holds real image pixels.
 */
class FrameFitter(val width: Int, val height: Int) {
    val content = Rect(0, 0, width, height)

    private val target: Bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(target)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val matrix = Matrix()

    fun fit(src: Bitmap, rotationDegrees: Int, letterbox: Boolean): Bitmap {
        if (rotationDegrees == 0 && src.width == width && src.height == height) {
            content.set(0, 0, width, height)
            return src
        }

        val swap = rotationDegrees % 180 != 0
        val rotatedW = if (swap) src.height else src.width
        val rotatedH = if (swap) src.width else src.height
        val sx = width / rotatedW.toFloat()
        val sy = height / rotatedH.toFloat()
        val scale = if (letterbox) min(sx, sy) else max(sx, sy)

        val cw = min(width, (rotatedW * scale).roundToInt())
        val ch = min(height, (rotatedH * scale).roundToInt())
        content.set((width - cw) / 2, (height - ch) / 2, (width + cw) / 2, (height + ch) / 2)

        matrix.reset()
        matrix.postTranslate(-src.width / 2f, -src.height / 2f)
        matrix.postRotate(rotationDegrees.toFloat())
        matrix.postScale(scale, scale)
        matrix.postTranslate(width / 2f, height / 2f)

        canvas.drawColor(Color.BLACK)
        canvas.drawBitmap(src, matrix, paint)
        return target
    }
}
