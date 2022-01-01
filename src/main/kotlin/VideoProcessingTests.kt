import nu.pattern.OpenCV
import org.opencv.core.Core.absdiff
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc.*
import org.opencv.videoio.VideoCapture

fun main() {
    OpenCV.loadLocally()
    videoDiff()
}

fun videoDiff() {
    val cap = VideoCapture("input/arrival_short_short.mp4")
    val frame = Mat()
    var oldFrame: Mat
    cap.read(frame)
    if (!frame.empty()) {
        var counter = 0
        oldFrame = frame
        cap.read(frame)
        while (!frame.empty() && counter < 1_000_000) {
            if (counter % 20 == 0) {
                println("processing image #$counter")
            }
            if (counter % 100 == 0) {
                System.gc()
            }

            val diffImage = Mat()
            val grayOld = Mat()
            val gray = Mat()
            val save = Mat()
            cvtColor(oldFrame, grayOld, COLOR_BGR2GRAY)
            cvtColor(frame, gray, COLOR_BGR2GRAY)
            absdiff(grayOld, gray, diffImage)
            threshold(diffImage, save, 1.0, 255.0, THRESH_BINARY)

            saveImage("slices/diff${counter}old.jpg", grayOld)
            saveImage("slices/diff${counter}.jpg", save)

            oldFrame = frame
            cap.read(frame)
            counter++
        }
    }
    cap.release()
}