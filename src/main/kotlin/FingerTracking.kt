import nu.pattern.OpenCV
import org.opencv.core.Core.minMaxLoc
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.dnn.Dnn.blobFromImage
import org.opencv.dnn.Dnn.readNetFromCaffe
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc.*


fun main() {
    OpenCV.loadLocally()
    val image = loadImage("input/setup12.png")
    val height = image.height()
    val width = image.width()
    FingerTracking().detect(
        image.submat((height * 0.5).toInt(), height, 0, width),
        "./src/main/resources/pose_iter_102000.caffemodel"
    )
}

class FingerTracking {

    var protoFile =
        "./src/main/resources/pose_deploy.prototxt"

    val nPoints = 44

    val POSE_PAIRS = arrayOf(
        intArrayOf(0, 1), intArrayOf(1, 2), intArrayOf(2, 3), intArrayOf(3, 4),
        intArrayOf(0, 5), intArrayOf(5, 6), intArrayOf(6, 7), intArrayOf(7, 8),
        intArrayOf(0, 9), intArrayOf(9, 10), intArrayOf(10, 11), intArrayOf(11, 12),
        intArrayOf(0, 13), intArrayOf(13, 14), intArrayOf(14, 15), intArrayOf(15, 16),
        intArrayOf(0, 17), intArrayOf(17, 18), intArrayOf(18, 19), intArrayOf(19, 20),
    )

    /**
     * You need to get the pose_iter_102000.caffemodel file for this function to work. File was too
     * big for Github
     * @param weightsFile Path to the models data
     */
    fun detect(image: Mat, weightsFile: String) {
        val thresh = 0.01f
        val frameCopy: Mat = image.clone()
        val frameWidth = image.cols()
        val frameHeight = image.rows()

        val aspectRatio: Double = frameWidth.toDouble() / frameHeight
        val inHeight = 368
        val inWidth: Int = ((aspectRatio * inHeight * 8) / 8).toInt()

        val net: Net = readNetFromCaffe(protoFile, weightsFile)

        val inpBlob: Mat = blobFromImage(
            image,
            1.0 / 255,
            Size(inWidth.toDouble(), inHeight.toDouble()),
            Scalar(0.0, 0.0, 0.0),
            false,
            false
        )

        net.setInput(inpBlob)

        val output: Mat = net.forward()

        val H: Int = output.size(2)
        val W: Int = output.size(3)

        // find the position of the body parts

        val points: Array<Point> = Array(nPoints) { Point(0.0, 0.0) }
        for (n in 0 until nPoints) {
            // Probability map of corresponding body's part.
            val sub = output.submat(0, 1, n, n + 1)
            val probMap: Mat = sub.reshape(1, sub.size(1) * sub.size(2))
            resize(probMap, probMap, Size(frameWidth.toDouble(), frameHeight.toDouble()))

            val result = minMaxLoc(probMap)
            val maxLoc: Point = result.maxLoc
            val prob: Double = result.maxVal

            if (prob > thresh) {
                circle(frameCopy, Point(maxLoc.x, maxLoc.y), 8, Scalar(0.0, 255.0, 255.0), -1)
                putText(
                    frameCopy,
                    "$n",
                    Point(maxLoc.x, maxLoc.y),
                    FONT_HERSHEY_COMPLEX,
                    1.0,
                    Scalar(0.0, 0.0, 255.0),
                    2
                )

            }
            points[n] = maxLoc
        }

        val nPairs: Int = POSE_PAIRS.size

        for (n in 0 until nPairs) {
            // lookup 2 connected body/hand parts
            val partA: Point = points[POSE_PAIRS[n][0]]
            val partB: Point = points[POSE_PAIRS[n][1]]

            if (partA.x <= 0 || partA.y <= 0 || partB.x <= 0 || partB.y <= 0)
                continue

            line(image, partA, partB, Scalar(0.0, 255.0, 255.0), 8)
            circle(image, partA, 8, Scalar(0.0, 0.0, 255.0), -1)
            circle(image, partB, 8, Scalar(0.0, 0.0, 255.0), -1)
        }

        saveImage("output/keypoints.png", frameCopy)
        saveImage("output/skeleton.png", image)
    }
}