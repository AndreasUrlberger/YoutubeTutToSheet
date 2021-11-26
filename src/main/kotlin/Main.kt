import nu.pattern.OpenCV
import org.opencv.core.Core.split
import org.opencv.core.CvType
import org.opencv.core.CvType.CV_32S
import org.opencv.core.CvType.CV_8UC3
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Imgproc.connectedComponents
import org.opencv.imgproc.Imgproc.threshold
import java.awt.Color
import java.io.FileNotFoundException
import kotlin.random.Random


fun main(args: Array<String>) {
    Main().start()
}

class Main {
    //Instantiating the ImageCodecs class
    val imageCodecs = Imgcodecs()

    init {
        OpenCV.loadLocally()
    }

    fun start() {
        val filepathInput =
            "C:\\Users\\A.Urlberger\\IdeaProjects\\YoutubeTutToSheet\\src\\main\\resources\\sample.png"
        val filepathOutput =
            "C:\\Users\\A.Urlberger\\IdeaProjects\\YoutubeTutToSheet\\src\\main\\resources\\output.jpg"
        val img: Mat =
            loadImage(filepathInput) ?: throw FileNotFoundException("Could not load image")
        if (img.empty())
            throw FileNotFoundException("Loaded image is empty, probably because it could not be found")
        val output: Mat = Mat()
        reshape(img, output, 0.78)
        hsv(output, output)


        saveImage(filepathOutput, output)
    }

    private fun reshape(img: Mat, out: Mat, heightPercentage: Double){
        val height = img.rows()
        val width = img.cols()
        img.submat(0, (height * heightPercentage).toInt(), 0, width).copyTo(out)
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
                dst.put(r, c, colors[label].red.toDouble(), colors[label].green.toDouble(), colors[label].blue.toDouble())
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