import nu.pattern.OpenCV
import org.opencv.core.*
import org.opencv.core.Core.split
import org.opencv.imgproc.Imgproc.*
import java.io.FileNotFoundException
import java.nio.file.Files
import kotlin.io.path.Path
import kotlin.math.roundToInt

fun main(args: Array<String>) {
    if (args.isEmpty())
        throw IllegalArgumentException("Needs at least the parameter 1: setup image path")
    if (Files.notExists(Path(args[0])))
        throw FileNotFoundException("Could not find setup image at '${args[0]}'")

    OpenCV.loadLocally()
    var big = loadImage(args[0])
    //big = big.submat(0, (big.height() * 0.5).toInt(), 0, big.width())
    detectTest(big)
}

private fun renderLines(img: Mat) {
    val keys = mutableListOf<Pair<Double, Double>>()
    val (_, keyboardWidth) = initKeys(keys)
    val pixelsPerInch = img.width() / keyboardWidth
    val keyBorders = keys.asSequence().map { old ->
        Pair(
            old.first * pixelsPerInch,
            ((old.second * pixelsPerInch).coerceAtMost(img.width().toDouble()))
        )
    }.toList()

    val top = (img.height() - 1).toDouble()
    val widthWhite = keyBorders.maxOf { it.second - it.first }
    val keyBordersTuned = tuneKeyBorders(keyBorders)
    keyBordersTuned.forEach { (left, right) ->
        if ((right - left) > widthWhite * 0.8) {
            rectangle(img, Point(left, 0.0), Point(right, top), Scalar(0.0, 255.0, 0.0), 1)
        }
    }

    saveImage("output/lines.jpg", img)

    println("render Lines")
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
    val adaptThresh0 = adaptThresh(rgbChannels[0], 5.0)
    val adaptThresh1 = adaptThresh(rgbChannels[1], 5.0)
    val adaptThresh2 = adaptThresh(rgbChannels[2], 5.0)
    saveImage("output/adaptThreshBlue.jpg", adaptThresh0)
    saveImage("output/adaptThreshGreen.jpg", adaptThresh1)
    saveImage("output/adaptThreshRed.jpg", adaptThresh2)

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

private fun detectTest(img: Mat) {
    cvtColor(img, img, COLOR_BGR2GRAY)
    val keys = mutableListOf<Pair<Double, Double>>()
    val (_, keyboardWidth) = initKeys(keys)
    val pixelsPerInch = img.width() / keyboardWidth
    val tunedKeyBorders = tuneKeyBorders(keys)
    val keyBorders = tunedKeyBorders.asSequence().map { old ->
        Pair(
            old.first * pixelsPerInch,
            ((old.second * pixelsPerInch).coerceAtMost(img.width().toDouble()))
        )
    }.toList()

    // TODO: maybe improve by making the combined image using adaptive threshold
    val contours = mutableListOf<MatOfPoint>()
    val hierarchy = Mat()
    val contourList = mutableListOf<MatOfPoint>()
    keyBorders.forEachIndexed { keyIndex, border ->
        contours.clear()
        val sliceWidth = (border.second - border.first)
        val extraWidth = sliceWidth * 0.2
        val widthThreshold = sliceWidth * 0.55
        val borderStart =
            (border.first - extraWidth).roundToInt().coerceIn(0, img.width())
        val borderEnd = (border.second + extraWidth).roundToInt().coerceIn(0, img.width())
        val slice = img.submat(0, img.height(), borderStart, borderEnd)


        findContours(slice, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE)
        for (x in 0 until hierarchy.width()) {
            // entry = (next, previous, firstChild, parent)
            val entry = hierarchy.get(0, x)
            var parentIndex = entry[3].toInt()
            var depth = 0
            while (parentIndex != -1) {
                depth++
                parentIndex = hierarchy.get(0, parentIndex)[3].toInt()
            }
            if (depth != 1) {
                continue
            }
            val bounding = boundingRect(contours[x])
            if (bounding.width >= widthThreshold) {
                rectangle(
                    img,
                    Point(border.first - extraWidth + bounding.tl().y, bounding.tl().x - 1),
                    Point(border.first - extraWidth + bounding.br().y, bounding.br().x + 1),
                    Scalar(0.0, 255.0, 0.0),
                    3
                )
                val contour = MatOfPoint(
                    *contours[x].toArray()
                        .map { Point(border.first + it.x - extraWidth, it.y) }
                        .toTypedArray()
                )
                contourList.add(contour)
            }
        }
    }


    val contoursToRemove = mutableListOf<MatOfPoint>()
    contourList.forEach { one ->
        contourList.forEach { other ->
            val oneBounding = boundingRect(one)
            val otherBounding = boundingRect(other)
            if (oneBounding.contains(otherBounding.tl()) && oneBounding.contains(otherBounding.br())) {
                contoursToRemove.add(other)
            }
        }
    }
    contourList.removeAll(contoursToRemove)

    println("contours: ${contourList.size}")
    cvtColor(img, img, COLOR_GRAY2BGR)
    for (contour in contourList) {
        drawContours(
            img,
            listOf(contour),
            -1,
            Scalar(0.0, 255.0, 0.0),
            //Scalar(Random.nextDouble(255.0), Random.nextDouble(255.0), Random.nextDouble(255.0)),
            1
        )
    }
    saveImage("output/rgbAdaptThresh.jpg", img)
}

var counter = 0
fun extractPatrickNotes(img: Mat): Mat {
    val rgbChannels = mutableListOf<Mat>()
    val contours = mutableListOf<MatOfPoint>()
    val hierarchy = Mat()
    val smallerList = mutableListOf<MatOfPoint>()
    val greaterList = mutableListOf<MatOfPoint>()
    val kernel = getStructuringElement(MORPH_RECT, Size(5.0, 5.0))

    /*line(img, Point(0.0, 0.0), Point(img.width().toDouble(), 0.0), Scalar(255.0, 255.0, 255.0), 1)
    line(
        img,
        Point(0.0, (img.height() - 1).toDouble()),
        Point(img.width().toDouble(), (img.height() - 1).toDouble()),
        Scalar(255.0, 255.0, 255.0),
        1
    )*/

    split(img, rgbChannels)
    val weightsBGR = doubleArrayOf(150.0, 110.0, 150.0)
    val addedThresh = computeAddedThresh(rgbChannels, weightsBGR)
    saveImage("slices/slice${counter}a.jpg", addedThresh)
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
        val (vertical, horizontal) = findExtrema(contour)
        val (top, bottom) = vertical
        val (left, right) = horizontal
        //rectangle(img, Point(left, top), Point(right, bottom), Scalar(0.0, 0.0, 255.0), 1)
        drawContours(addedThresh, mutableListOf(contour), -1, Scalar(255.0, 255.0, 255.0), -1)
    }

    /*val opening = Mat()
    morphologyEx(addedThresh, opening, MORPH_OPEN, kernel)*/
    //saveImage("./output/opening.jpg", opening)
    return addedThresh
}