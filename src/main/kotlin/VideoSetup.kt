import nu.pattern.OpenCV
import org.opencv.core.*
import org.opencv.core.Core.split
import org.opencv.imgproc.Imgproc.*
import java.io.FileNotFoundException
import java.nio.file.Files
import kotlin.io.path.Path

fun main(args: Array<String>) {
    if (args.isEmpty())
        throw IllegalArgumentException("Needs at least the parameter 1: setup image path")
    if (Files.notExists(Path(args[0])))
        throw FileNotFoundException("Could not find setup image at '${args[0]}'")

    OpenCV.loadLocally()
    var big = loadImage(args[0])
    //big = big.submat(0, (big.height() * 0.5).toInt(), 0, big.width())
    videoSetup(big)

}

private fun videoSetup(img: Mat) {
    // store rgb
    val rgbChannels = mutableListOf<Mat>()
    split(img, rgbChannels)
    saveImage("./output/blue.jpg", rgbChannels[0])
    saveImage("./output/green.jpg", rgbChannels[1])
    saveImage("./output/red.jpg", rgbChannels[2])

    // store hsv
    val hsv = Mat()
    cvtColor(img, hsv, COLOR_BGR2HSV)
    val hsvChannels = mutableListOf<Mat>()
    split(hsv, hsvChannels)
    saveImage("./output/hue.jpg", hsvChannels[0])
    saveImage("./output/saturation.jpg", hsvChannels[1])
    saveImage("./output/intensity.jpg", hsvChannels[2])

    // complex
    val red = 150.0
    val green = 110.0
    val blue = 150.0
    val addedThresh = computeAddedThresh(rgbChannels, doubleArrayOf(blue, green, red))
    saveImage("./output/addedThresh.jpg", addedThresh)

    // adaptive
    val adaptThresh0 = adaptThresh(rgbChannels[0], 11, 2.0)
    val adaptThresh1 = adaptThresh(rgbChannels[1], 5, 5.0)
    val adaptThresh2 = adaptThresh(rgbChannels[2], 11, 5.0)
    saveImage("output/adaptThreshBlue.jpg", adaptThresh0)
    saveImage("output/adaptThreshGreen.jpg", adaptThresh1)
    saveImage("output/adaptThreshRed.jpg", adaptThresh2)
    val gray = Mat()
    cvtColor(img, gray, COLOR_BGR2GRAY)
    val adaptThreshGray = adaptThresh(gray, 7, 7.0)
    saveImage("output/adaptThreshGray.jpg", adaptThreshGray)

    val addedAdaptThresh = Mat()
    Core.bitwise_and(adaptThresh0, adaptThresh1, addedAdaptThresh)
    Core.bitwise_and(addedThresh, adaptThresh2, addedAdaptThresh)
    saveImage("output/adaptThreshComb.jpg", addedAdaptThresh)

    val edges = Mat()
    Canny(img, edges, 230.0, 255.0)
    saveImage("./output/canny.jpg", edges)

    saveImage("./output/img.jpg", img)
    saveImage("./output/addedThreshCleared.jpg", addedThresh)
}

var counter = 0
fun extractPatrickNotes(img: Mat): Mat {
    val rgbChannels = mutableListOf<Mat>()
    val contours = mutableListOf<MatOfPoint>()
    val hierarchy = Mat()
    val smallerList = mutableListOf<MatOfPoint>()
    val greaterList = mutableListOf<MatOfPoint>()
    val kernel = getStructuringElement(MORPH_RECT, Size(5.0, 5.0))

    split(img, rgbChannels)
    val weightsBGR = doubleArrayOf(150.0, 110.0, 150.0)
    val addedThresh = computeAddedThresh(rgbChannels, weightsBGR)
    //saveImage("slices/slice${counter}a.jpg", addedThresh)
    counter++

    findContours(addedThresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE)
    for (index in 0 until contours.size) {
        val contour = contours[index]
        if (contour.size().height >= 25) {
            greaterList.add(contour)
        } else {
            smallerList.add(contour)
        }
    }
    for (contour in smallerList) {
        drawContours(addedThresh, mutableListOf(contour), -1, Scalar(0.0, 0.0, 0.0), -1)
    }
    for (contour in greaterList) {
        //rectangle(img, Point(left, top), Point(right, bottom), Scalar(0.0, 0.0, 255.0), 1)
        drawContours(addedThresh, mutableListOf(contour), -1, Scalar(255.0, 255.0, 255.0), -1)
    }

    /*val opening = Mat()
    morphologyEx(addedThresh, opening, MORPH_OPEN, kernel)*/
    //saveImage("./output/opening.jpg", opening)
    return addedThresh
}