package dev.adityakaran.edgeseg

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Rect

object Cityscapes {
    const val NUM_CLASSES = 19

    val NAMES = arrayOf(
        "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
        "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
        "truck", "bus", "train", "motorcycle", "bicycle",
    )

    val COLORS = intArrayOf(
        Color.rgb(128, 64, 128), Color.rgb(244, 35, 232), Color.rgb(70, 70, 70),
        Color.rgb(102, 102, 156), Color.rgb(190, 153, 153), Color.rgb(153, 153, 153),
        Color.rgb(250, 170, 30), Color.rgb(220, 220, 0), Color.rgb(107, 142, 35),
        Color.rgb(152, 251, 152), Color.rgb(70, 130, 180), Color.rgb(220, 20, 60),
        Color.rgb(255, 0, 0), Color.rgb(0, 0, 142), Color.rgb(0, 0, 70),
        Color.rgb(0, 60, 100), Color.rgb(0, 80, 100), Color.rgb(0, 0, 230),
        Color.rgb(119, 11, 32),
    )

    /** Mask pixels in [region] -> ARGB bitmap at mask resolution; the view scales it up over the frame. */
    fun colorize(result: Segmenter.Result, region: Rect): Bitmap {
        val w = region.width()
        val colors = IntArray(w * region.height())
        for (y in 0 until region.height()) {
            val row = (region.top + y) * result.maskWidth + region.left
            for (x in 0 until w) colors[y * w + x] = COLORS[result.mask[row + x].toInt() and 0xFF]
        }
        return Bitmap.createBitmap(colors, w, region.height(), Bitmap.Config.ARGB_8888)
    }

    /** Top classes by pixel share inside [region], e.g. [(13, 0.21f), (0, 0.18f), ...]. */
    fun coverage(result: Segmenter.Result, region: Rect, top: Int = 6): List<Pair<Int, Float>> {
        val counts = IntArray(NUM_CLASSES)
        for (y in region.top until region.bottom) {
            val row = y * result.maskWidth
            for (x in region.left until region.right) counts[result.mask[row + x].toInt() and 0xFF]++
        }
        val total = (region.width() * region.height()).coerceAtLeast(1).toFloat()
        return counts.indices
            .filter { counts[it] > 0 }
            .sortedByDescending { counts[it] }
            .take(top)
            .map { it to counts[it] / total }
    }
}
