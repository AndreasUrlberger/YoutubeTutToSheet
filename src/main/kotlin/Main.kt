import nu.pattern.OpenCV
import org.opencv.core.*
import org.opencv.core.Core.bitwise_and
import org.opencv.core.Core.split
import org.opencv.core.CvType.CV_32S
import org.opencv.core.CvType.CV_8UC3
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Imgproc.*
import org.opencv.videoio.VideoCapture
import org.opencv.videoio.VideoWriter
import java.awt.Color
import java.io.FileNotFoundException
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.random.Random


fun main(args: Array<String>) {
    if (args.size < 2)
        throw IllegalArgumentException("Missing program arguments, needed: 1. path to input file 2. path to where the output file should get stored")
    Main(args[0], args[1]).start()
}

class Main(private val filepathInput: String, private val filepathOutput: String) {
    //Instantiating the ImageCodecs class
    val imageCodecs = Imgcodecs()

    init {
        OpenCV.loadLocally()
    }

    fun start() {
        //convertVideo()
        testImg()
    }

    private fun convertVideo() {
        val start = System.currentTimeMillis()
        val cap = VideoCapture(filepathInput)
        val frame = Mat()
        val out = Mat()
        val cut = Mat()
        cap.read(frame)
        var i = 1
        if (!frame.empty()) {
            reshape(frame, cut, 0.75)
            val size = Size(cut.width().toDouble(), cut.height().toDouble())
            val fourcc = VideoWriter.fourcc('X', 'V', 'I', 'D')
            val vw = VideoWriter(filepathOutput, fourcc, 30.0, size)
            do {
                println("reading frame $i")
                reshape(frame, cut, 0.75)
                sliceAndConquer(cut, out)
                vw.write(out)

                i += 1
                cap.read(frame)
            } while (cap.isOpened && !frame.empty())
            vw.release()
        }
        cap.release()
        val end = System.currentTimeMillis()
        println("time needed: ${end - start} ms")
    }

    private fun testImg() {
        val img: Mat =
            loadImage(filepathInput) ?: throw FileNotFoundException("Could not load image")
        if (img.empty())
            throw FileNotFoundException("Loaded image is empty, probably because it could not be found")
        val output = Mat()
        val start = System.currentTimeMillis()
        reshape(img, output, 0.78)
        sliceAndConquer(img, output)

        val end = System.currentTimeMillis()
        println("time: ${end - start}ms")
        saveImage(filepathOutput, output)
    }

    private fun reshape(img: Mat, out: Mat, heightPercentage: Double) {
        val height = img.rows()
        val width = img.cols()
        img.submat(0, (height * heightPercentage).toInt(), 0, width).copyTo(out)
    }

    private fun initKeys(
        keyLow: Char,
        lowNum: Int,
        keyHigh: Char,
        highNum: Int,
        outKeys: MutableList<Pair<Double, Double>>
    ): Pair<Int, Double> {
        val keyCodes = mapOf('a' to 9, 'h' to 11, 'c' to 0, 'd' to 2, 'e' to 4, 'f' to 5, 'g' to 7)
        checkBorderKeys(lowNum, highNum, keyLow, keyHigh)

        val keyDimensions = listOf(
            // (left position relative to the start of last octave, width of the key, isWhiteKey)
            Triple(0.0, 0.9, true), // c
            // (left position relative to end of last white key, same for the right position, isWhiteKey)
            Triple(-0.35, 0.15, false), // cis
            Triple(0.9, 0.9, true), // d
            Triple(-0.15, 0.35, false), // dis
            Triple(1.8, 0.9, true), // e
            Triple(2.7, 0.95, true), // f
            Triple(-0.4, 0.1, false), // fis
            Triple(3.65, 0.95, true), // g
            Triple(-0.25, 0.25, false), // gis
            Triple(4.6, 0.95, true), // a
            Triple(-0.1, 0.4, false), // ais
            Triple(5.55, 0.95, true), // h
        )

        var whiteKeys = 0
        var xOffset = 0.0
        var width = 0.0
        // lower single keys
        for (key in (keyCodes[keyLow] ?: 0)..(keyCodes.values.maxOrNull() ?: 0)) {
            val dim = keyDimensions[key]
            if (dim.third) { // white key
                outKeys.add(Pair(xOffset, xOffset + dim.second))
                xOffset += dim.second
                whiteKeys++
                width += dim.second
            } else {
                outKeys.add(Pair(xOffset + dim.first, xOffset + dim.second))
            }
        }
        // octave keys
        for (scale in (lowNum + 1) until highNum) {
            for (key in keyDimensions.indices) {
                val dim = keyDimensions[key]
                if (dim.third) { // white key
                    outKeys.add(Pair(xOffset, xOffset + dim.second))
                    xOffset += dim.second
                    whiteKeys++
                    width += dim.second
                } else {
                    outKeys.add(Pair(xOffset + dim.first, xOffset + dim.second))
                }
            }
        }
        // upper single keys
        for (key in (keyCodes.values.minOrNull() ?: 0)..(keyCodes[keyHigh] ?: 0)) {
            val dim = keyDimensions[key]
            if (dim.third) { // white key
                outKeys.add(Pair(xOffset, xOffset + dim.second))
                xOffset += dim.second
                whiteKeys++
                width += dim.second
            } else {
                outKeys.add(Pair(xOffset + dim.first, xOffset + dim.second))
            }
        }

        return Pair(whiteKeys, width)
    }

    private fun checkBorderKeys(lowNum: Int, highNum: Int, keyLow: Char, keyHigh: Char) {
        if (lowNum !in -3..5)
            throw java.lang.IllegalArgumentException("lowNum is out of range [-3-4]")
        if (highNum !in -3..5)
            throw java.lang.IllegalArgumentException("highNum is out of range [-3-4]")
        if (lowNum >= highNum)
            throw java.lang.IllegalArgumentException("lowNum must be greater than highNum")
        if (!listOf('a', 'h', 'c', 'd', 'e', 'f', 'g').contains(keyLow.lowercaseChar()))
            throw java.lang.IllegalArgumentException("keyLow must be one of [a, h, c, d, e, f, g]")
        if (!listOf('a', 'h', 'c', 'd', 'e', 'f', 'g').contains(keyHigh.lowercaseChar()))
            throw java.lang.IllegalArgumentException("keyHigh must be one of [a, h, c, d, e, f, g]")
    }

    private fun sliceAndConquer(img: Mat, out: Mat) {
        val keys = mutableListOf<Pair<Double, Double>>()
        val (whiteNotes, keyboardWidth) = initKeys('a', -3, 'c', 5, keys)
        val channels: MutableList<Mat> = mutableListOf()
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        val points = mutableMapOf<Int, MutableList<Pair<Point, Point>>>()
        val pixelsPerInch = img.width() / keyboardWidth

        val keyBorders = keys.asSequence().map { old ->
            Pair(
                old.first * pixelsPerInch,
                ((old.second * pixelsPerInch).coerceAtMost(img.width().toDouble()))
            )
        }
        val white = Scalar(255.0, 255.0, 255.0)
        line(img, Point(0.0, 0.0), Point(img.width().toDouble(), 0.0), white, 1)
        //println("keys: ${keyBorders.joinToString(", ")} ")
        //println("relative: ${keys.joinToString(separator = ", ")}}")

        var sliceIndex = 0
        for (border in keyBorders) {
            contours.clear()
            channels.clear()
            val width = (border.second - border.first)
            val widthThreshold = width * 0.55
            val slice = img.submat(
                0, img.height(),
                (border.first - width * 0.2).roundToInt().coerceIn(0, img.width()),
                (border.second + width * 0.2).roundToInt().coerceIn(0, img.width())
            )

            split(slice, channels)
            val weights = doubleArrayOf(80.0, 85.0, 80.0)
            val addedThresh = computeAddedThresh(channels, weights)
            findContours(addedThresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE)
            saveImage("./slices/slice_$sliceIndex.jpg", slice)
            saveImage("./slices/slice_${sliceIndex}a.jpg", addedThresh)
            saveImage("./slices/slice_${sliceIndex}a0.jpg", channels[0])
            saveImage("./slices/slice_${sliceIndex}a1.jpg", channels[1])
            saveImage("./slices/slice_${sliceIndex}a2.jpg", channels[2])

            for (x in 0 until hierarchy.width()) {
                for (y in 0 until hierarchy.height()) {
                    // entry = (next, previous, firstChild, parent)
                    val entry = hierarchy.get(y, x)
                    val parent = entry[3]
                    if (parent.toInt() != -1) {
                        val foundPoints = findMinMaxPoint(contours[x])
                        foundPoints.first.x += border.first - width * 0.2
                        foundPoints.second.x += border.first - width * 0.2
                        if (foundPoints.first.x - foundPoints.second.x >= widthThreshold) {
                            points.getOrPut(x) { mutableListOf() }.add(foundPoints)
                        }
                    }
                }
            }
            sliceIndex++
        }

        for (key in points.keys) {
            val item = points.getOrDefault(key, mutableListOf())
            for (y in 0 until item.size) {
                val notesInLine = item[y]
                //println("width of note: ${notesInLine.first.x - notesInLine.second.x}")
                rectangle(img, notesInLine.first, notesInLine.second, Scalar(0.0, 0.0, 255.0), 3)
            }
        }

        //saveImage("./notes.jpg", img)
        img.copyTo(out)
    }

    private fun findMinMaxPoint(elem: MatOfPoint): Pair<Point, Point> {
        val maxXY = elem.toArray().reduce { point1, point2 ->
            point1.x = max(point1.x, point2.x)
            point1.y = max(point1.y, point2.y)
            point1
        }
        val minXY = elem.toArray().reduce { point1, point2 ->
            point1.x = min(point1.x, point2.x)
            point1.y = min(point1.y, point2.y)
            point1
        }
        return Pair(maxXY, minXY)
    }

    private fun test3(img: Mat, out: Mat) {
        val channels: MutableList<Mat> = mutableListOf()
        split(img, channels)

        val weights = doubleArrayOf(110.0, 160.0, 160.0)
        val addedThresh = computeAddedThresh(channels, weights)

        saveImage("./addedThres.jpg", addedThresh)

        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        findContours(addedThresh, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE)
        drawContours(img, contours, -1, Scalar(0.0, 0.0, 0255.0), 2)
        img.copyTo(out)
    }

    private fun computeAddedThresh(channels: MutableList<Mat>, weights: DoubleArray): Mat {
        if (weights.size < 3) {
            throw IllegalArgumentException("size of weights must be exactly 3")
        }
        val threshBlue = Mat()
        val threshGreen = Mat()
        val threshRed = Mat()
        val limitBlue = weights[2]
        val limitGreen = weights[1]
        val limitRed = weights[0]
        threshold(channels[0], threshBlue, limitBlue, 255.0, THRESH_BINARY)
        threshold(channels[1], threshGreen, limitGreen, 255.0, THRESH_BINARY)
        threshold(channels[2], threshRed, limitRed, 255.0, THRESH_BINARY)
        /*saveImage("./threshBlue.jpg", threshBlue)
        saveImage("./threshGreen.jpg", threshGreen)
        saveImage("./threshRed.jpg", threshRed)*/

        val addedThresh = Mat()
        bitwise_and(threshBlue, threshGreen, addedThresh)
        bitwise_and(addedThresh, threshRed, addedThresh)
        return addedThresh
    }

    private fun test2(img: Mat, out: Mat) {
        val hsv = Mat()
        val thresh = Mat()
        cvtColor(img, hsv, COLOR_BGR2HSV)
        val channels: MutableList<Mat> = mutableListOf()
        split(hsv, channels)

        val blur = Mat()
        GaussianBlur(channels[2], blur, Size(7.0, 7.0), 0.0, 0.0)
        val cannyBlur = Mat()
        Canny(blur, cannyBlur, 230.0, 230.0)
        saveImage("./cannyBlur.jpg", cannyBlur)

        val hueBlur = Mat()
        GaussianBlur(channels[2], hueBlur, Size(7.0, 7.0), 0.0, 0.0)
        threshold(hueBlur, thresh, 128.0, 255.0, THRESH_BINARY)
        saveImage("./thresh.jpg", thresh)
        saveImage("./channel.jpg", channels[2])


        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        findContours(thresh, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE)
        drawContours(img, contours, -1, Scalar(0.0, 0.0, 255.0), Core.FILLED)
        img.copyTo(out)
    }

    private fun test(img: Mat, out: Mat) {
        val hsv = Mat()
        val thresh = Mat()
        cvtColor(img, hsv, COLOR_BGR2HSV)
        val channels: MutableList<Mat> = mutableListOf()
        split(hsv, channels)
        threshold(channels[2], thresh, 128.0, 255.0, THRESH_BINARY)

        /*val gray: Mat = Mat()
        val thresh: Mat = Mat()
        cvtColor(img, gray, COLOR_BGR2GRAY)
        threshold(gray, thresh, 170.0, 255.0, THRESH_BINARY)*/
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        findContours(thresh, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE)
        println(hierarchy)
        drawContours(img, contours, -1, Scalar(0.0, 0.0, 255.0), 2)
        img.copyTo(out)
    }

    private fun hsv(img: Mat, out: Mat) {
        val hsv: Mat = Mat()
        val thres: Mat = Mat()
        Imgproc.cvtColor(img, hsv, Imgproc.COLOR_BGR2HSV)
        val channels: MutableList<Mat> = mutableListOf()
        split(hsv, channels)

        println("channels ${channels.size}")
        threshold(channels[2], thres, 128.0, 255.0, Imgproc.THRESH_BINARY)
        val labelImage: Mat = Mat(img.size(), CV_32S);
        val nLabels = connectedComponents(thres, labelImage, 8)
        val colors = mutableListOf<Color>()
        println("labels: $nLabels")
        for (label in 0 until nLabels) {
            colors.add(Color(Random.nextInt(256), Random.nextInt(256), Random.nextInt(256)))
        }

        val dst: Mat = Mat(img.size(), CV_8UC3);
        for (r in 0 until dst.rows()) {
            for (c in 0 until dst.cols()) {
                val label: Int = labelImage.get(r, c).first().toInt()
                dst.put(
                    r,
                    c,
                    colors[label].red.toDouble(),
                    colors[label].green.toDouble(),
                    colors[label].blue.toDouble()
                )
            }
        }

        dst.copyTo(out)
//        std::vector<Vec3b> colors(nLabels);
//        colors[0] = Vec3b(0, 0, 0);//background
//        for(int label = 1; label < nLabels; ++label){
//            colors[label] = Vec3b( (rand()&255), (rand()&255), (rand()&255) );
//        }
    }

    private fun loadImage(imagePath: String): Mat? {
        return Imgcodecs.imread(imagePath)
    }

    private fun saveImage(path: String, image: Mat) {
        Imgcodecs.imwrite(path, image)
    }
}