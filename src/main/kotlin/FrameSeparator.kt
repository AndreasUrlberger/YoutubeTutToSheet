import nu.pattern.OpenCV
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.videoio.VideoCapture


fun main(args: Array<String>) {
    if (args.isEmpty())
        throw IllegalArgumentException("Missing the filepath parameter")
    val filepath = args[0]
    val separator = FrameSeparator()
    separator.start(filepath)
}

class FrameSeparator {
    //Instantiating the ImageCodecs class
    val imageCodecs = Imgcodecs()

    init {
        OpenCV.loadLocally()
    }

    fun start(filename: String) {
        val time: Long = System.currentTimeMillis()
        val cap = VideoCapture(filename)
        var i = 0
        val frame: Mat = Mat()
        var hasNext = true
        while (cap.isOpened && hasNext) {
            println("reading frame $i")
            hasNext = cap.read(frame)
            if (!hasNext)
                break
            val succWrite = Imgcodecs.imwrite(
                "C:\\Users\\Spieler 4\\IdeaProjects\\YoutubeTutToSheet\\src\\main\\resources\\frames\\frame_$i.bmp",
                frame
            )
            println("frame saved successfully: $succWrite")
            i += 1
        }
        println("Time Used:" + (System.currentTimeMillis() - time) + " Milliseconds");
    }
}