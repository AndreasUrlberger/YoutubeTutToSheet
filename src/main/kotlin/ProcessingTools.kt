import org.opencv.core.Core
import org.opencv.core.CvType.CV_8UC1
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc

fun computeAddedThresh(channels: MutableList<Mat>, weights: DoubleArray): Mat {
    if (weights.size < 3) {
        throw IllegalArgumentException("size of weights must be exactly 3")
    }
    val threshBlue = Mat()
    val threshGreen = Mat()
    val threshRed = Mat()
    val limitBlue = weights[2]
    val limitGreen = weights[1]
    val limitRed = weights[0]
    Imgproc.threshold(channels[0], threshBlue, limitBlue, 255.0, Imgproc.THRESH_BINARY)
    Imgproc.threshold(channels[1], threshGreen, limitGreen, 255.0, Imgproc.THRESH_BINARY)
    Imgproc.threshold(channels[2], threshRed, limitRed, 255.0, Imgproc.THRESH_BINARY)

    val addedThresh = Mat()
    Core.bitwise_and(threshBlue, threshGreen, addedThresh)
    Core.bitwise_and(addedThresh, threshRed, addedThresh)
    return addedThresh
}

fun saveImage(path: String, image: Mat) {
    Imgcodecs.imwrite(path, image)
}

fun loadImage(imagePath: String): Mat {
    return Imgcodecs.imread(imagePath)
        ?: throw IllegalArgumentException("Could not find image at path '$imagePath'")
}

fun adaptThresh(channel: Mat, blockSize: Int, C: Double): Mat {
    val adaptThresh = Mat()
    // BORDER_REPLICATE | #BORDER_ISOLATED
    println("type before: ${channel.type()}, $channel")
    var converted = Mat()
    if (channel.type() != CV_8UC1) {
        channel.convertTo(converted, CV_8UC1)
    } else {
        converted = channel
    }
    println("type after: ${converted.type()}, wanted: $CV_8UC1")

    Imgproc.adaptiveThreshold(
        converted,
        adaptThresh,
        255.0,
        Core.BORDER_REPLICATE,
        Imgproc.THRESH_BINARY_INV,
        blockSize,
        C
    )
    return adaptThresh
}

