import nu.pattern.OpenCV
import org.opencv.core.*
import org.opencv.core.Core.bitwise_and
import org.opencv.core.Core.split
import org.opencv.core.CvType.CV_32S
import org.opencv.core.CvType.CV_8UC3
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Imgproc.*
import java.awt.Color
import java.io.FileNotFoundException
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
        val img: Mat =
            loadImage(filepathInput) ?: throw FileNotFoundException("Could not load image")
        if (img.empty())
            throw FileNotFoundException("Loaded image is empty, probably because it could not be found")
        val output = Mat()
        reshape(img, output, 0.78)
        test3(output, output)


        saveImage(filepathOutput, output)
    }

    private fun reshape(img: Mat, out: Mat, heightPercentage: Double) {
        val height = img.rows()
        val width = img.cols()
        img.submat(0, (height * heightPercentage).toInt(), 0, width).copyTo(out)
    }

    private fun test3(img: Mat, out: Mat) {
        val channels: MutableList<Mat> = mutableListOf()
        split(img, channels)

        val threshBlue = Mat()
        val threshGreen = Mat()
        val threshRed = Mat()
        val limitBlue = 160.0
        val limitGreen = 160.0
        val limitRed = 110.0
        threshold(channels[0], threshBlue, limitBlue, 255.0, THRESH_BINARY)
        threshold(channels[1], threshGreen, limitGreen, 255.0, THRESH_BINARY)
        threshold(channels[2], threshRed, limitRed, 255.0, THRESH_BINARY)
        saveImage("./threshBlue.jpg", threshBlue)
        saveImage("./threshGreen.jpg", threshGreen)
        saveImage("./threshRed.jpg", threshRed)

        val addedThresh = Mat()
        bitwise_and(threshBlue, threshGreen, addedThresh)
        bitwise_and(addedThresh, threshRed, addedThresh)

        saveImage("./addedThres.jpg", addedThresh)

        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        findContours(addedThresh, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE)
        drawContours(img, contours, -1, Scalar(0.0, 0.0, 255.0), 2)
        img.copyTo(out)
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

    private fun convolFilter(image: Mat, output: Mat) {
        // Create a kernel that we will use to sharpen our image
        val kernel = Mat(3, 3, CvType.CV_32F)
        // an approximation of second derivative, a quite strong kernel
        val kernelData = FloatArray((kernel.total() * kernel.channels()).toInt())
        kernelData[0] = 1f
        kernelData[1] = 1f
        kernelData[2] = 1f
        kernelData[3] = 1f
        kernelData[4] = -8f
        kernelData[5] = 1f
        kernelData[6] = 1f
        kernelData[7] = 1f
        kernelData[8] = 1f
        kernel.put(0, 0, kernelData)
        Imgproc.filter2D(image, output, 8, kernel)
    }

    private fun applySobel(input: Mat, output: Mat) {
        Imgproc.Sobel(input, output, -1, 0, 1)
    }

    private fun loadImage(imagePath: String): Mat? {
        return Imgcodecs.imread(imagePath)
    }

    private fun saveImage(path: String, image: Mat) {
        Imgcodecs.imwrite(path, image)
    }
}