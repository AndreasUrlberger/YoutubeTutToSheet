import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import nu.pattern.OpenCV
import org.opencv.core.Core.bitwise_and
import org.opencv.core.Core.split
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc.*
import org.opencv.videoio.VideoCapture
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import javax.sound.midi.MidiSystem
import javax.sound.midi.ShortMessage.NOTE_OFF
import javax.sound.midi.ShortMessage.NOTE_ON
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.system.exitProcess
import kotlin.system.measureTimeMillis

// -XX:ParallelGCThreads=1
// improves performance immensely
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
        //detectNotesInVideo()
        createMidiFromTimeline()
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

    private fun createMidiFromTimeline() {
        val timelineString: String
        FileInputStream("./output/timeline.json").use {
            timelineString = it.readAllBytes().decodeToString()
        }
        val timeline = Json.decodeFromString<Timeline>(timelineString)
        val singleTimeline = convertInSingleTimeline(timeline)
        val shiftedTimeline = shiftTimelineToStart(singleTimeline)
        createAndPlayMidi(shiftedTimeline)
    }

    private fun shiftTimelineToStart(timeline: List<Triple<Double, Int, Boolean>>): List<Triple<Double, Int, Boolean>> {
        if (timeline.isEmpty()) {
            return timeline
        }
        val zeroLine = mutableListOf<Triple<Double, Int, Boolean>>()
        var smallest = Double.MIN_VALUE
        val offset = timeline.minOf { it.first }
        timeline.forEach { note ->
            smallest = min(smallest, note.first - offset)
            zeroLine.add(Triple(note.first - offset, note.second, note.third))
        }
        println("offset: $offset smallest: $smallest")
        return zeroLine
    }

    private fun detectNotesInVideo() {
        val millis = measureTimeMillis {
            val cap = VideoCapture(filepathInput)
            val notes = mutableListOf<Map<Int, List<Pair<Double, Double>>>>()
            val frame = Mat()
            val cut = Mat()
            var keyCount = 0
            var frameHeight = 0

            cap.read(frame)
            var counter = 0
            if (!frame.empty()) {
                val keys = mutableListOf<Pair<Double, Double>>()
                val (_, keyboardWidth) = initKeys('a', -3, 'c', 5, keys)
                keyCount = keys.size
                val pixelsPerInch = frame.width() / keyboardWidth
                val keyBorders = keys.asSequence().map { old ->
                    Pair(
                        old.first * pixelsPerInch,
                        ((old.second * pixelsPerInch).coerceAtMost(frame.width().toDouble()))
                    )
                }

                while (!frame.empty()) {
                    counter++
                    if (counter % 10 == 0)
                        println("processing frame #$counter")
                    reshape(frame, cut, 0.75)
                    frameHeight = frame.height()
                    val imageNotes = detectNotesInImage(cut, keyBorders, counter)
                    notes.add(imageNotes)
                    cap.read(frame)
                }
            }
            cap.release()

            if (frameHeight <= 0) {
                println("Failed at processing the images")
            } else {
                val timeline = convertNotesToTimestamps(notes, keyCount, frameHeight.toDouble())

                val timelineString = Json.encodeToString(timeline)
                FileOutputStream("./output/timeline.json").use {
                    it.write(timelineString.toByteArray())
                }
            }
        }
        println("time needed: $millis ms")
    }

    private fun convertInSingleTimeline(timeline: Timeline): List<Triple<Double, Int, Boolean>> {
        // Triple = (time, key, onOrOff)
        val singleTimeline = mutableListOf<Triple<Double, Int, Boolean>>()
        timeline.timeline.forEachIndexed { keyIndex, events ->
            events.forEach { event ->
                // adjust time
                singleTimeline.add(
                    Triple(
                        event.first,
                        keyIndex,
                        true
                    )
                )
                singleTimeline.add(
                    Triple(
                        event.second,
                        keyIndex,
                        false
                    )
                )
            }
        }
        // might need to sort it
        return singleTimeline
    }

    private fun createAndPlayMidi(timeline: List<Triple<Double, Int, Boolean>>) {
        println("create midi")
        val sequencer = MidiSystem.getSequencer()
        sequencer.open()
        // Creating a sequence.
        val sequence = javax.sound.midi.Sequence(javax.sound.midi.Sequence.PPQ, 4)
        // PPQ(Pulse per ticks) is used to specify timing, type and 4 is the timing resolution.
        // Creating a track on our sequence upon which MIDI events would be placed
        val track = sequence.createTrack()

        val baseNote = 21 // the code of the lowest note on the piano, the A2
        // Add events
        timeline.forEach { event ->
            if (event.third) {
                // on
                track.add(
                    makeEvent(
                        NOTE_ON,
                        1,
                        event.second + baseNote,
                        100,
                        event.first.div(52).toInt()
                    )
                )
            } else {
                // off
                track.add(
                    makeEvent(
                        NOTE_OFF,
                        1,
                        event.second + baseNote,
                        100,
                        event.first.div(52).toInt()
                    )
                )
            }
        }

        // Setting our sequence so that the sequencer can run it on synthesizer
        sequencer.sequence = sequence

        // Specifies the beat rate in beats per minute.
        sequencer.tempoInBPM = 104.toFloat()

        println("storing midi file")
        val file = File("./output/midi.mid")
        MidiSystem.write(sequence, 0, file)
        println("start playing")
        // Sequencer starts to play notes
        /*sequencer.start()
        while (true) {

            // Exit the program when sequencer has stopped playing.
            if (!sequencer.isRunning) {
                sequencer.close()
                exitProcess(1)
            }
        }*/
        exitProcess(1)
    }


    private fun estimateKeySpeed(key: List<List<Pair<Double, Double>>>, guess: Double): Double {
        val currentNotes = mutableListOf<Pair<Double, Double>>()
        var estimatedSpeed = guess
        for (frame in key) {
            // TODO: misses implementation
        }
        return estimatedSpeed
    }

    private fun detectNotesInImage(
        img: Mat,
        keyBorders: Sequence<Pair<Double, Double>>,
        index: Int
    ): Map<Int, List<Pair<Double, Double>>> {
        val channels: MutableList<Mat> = mutableListOf()
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        val notes = mutableMapOf<Int, MutableList<Pair<Double, Double>>>()

        val white = Scalar(255.0, 255.0, 255.0)
        line(img, Point(0.0, 0.0), Point(img.width().toDouble(), 0.0), white, 1)
        split(img, channels)

        val slices = mutableListOf<Mat>()
        val thresholds = doubleArrayOf(80.0, 85.0, 80.0)
        keyBorders.forEachIndexed { sliceIndex, border ->
            contours.clear()
            slices.clear()
            val sliceWidth = (border.second - border.first)
            val widthThreshold = sliceWidth * 0.55
            val borderStart =
                (border.first - sliceWidth * 0.2).roundToInt().coerceIn(0, img.width())
            val borderEnd = (border.second + sliceWidth * 0.2).roundToInt().coerceIn(0, img.width())
            slices.add(0, channels[0].submat(0, img.height(), borderStart, borderEnd))
            slices.add(1, channels[1].submat(0, img.height(), borderStart, borderEnd))
            slices.add(2, channels[2].submat(0, img.height(), borderStart, borderEnd))

            val addedThresh = computeAddedThresh(slices, thresholds)
            findContours(addedThresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE)
            for (x in 0 until hierarchy.width()) {
                for (y in 0 until hierarchy.height()) {
                    // entry = (next, previous, firstChild, parent)
                    val entry = hierarchy.get(y, x)
                    val parent = entry[3]
                    if (parent.toInt() != -1) {
                        val (top, bottom, width) = findTopBottomAndWidth(contours[x])
                        if (width >= widthThreshold) {
                            //println("key $sliceIndex from $bottom to $top")
                            //rectangle(img, Point(border.first, top), Point(border.second, bottom), Scalar(0.0, 0.0, 255.0), 2)
                            notes.getOrPut(sliceIndex) { mutableListOf() }.add(Pair(top, bottom))
                        }
                    }
                }
            }
        }
        //saveImage("./slices/frame_$index.jpg", img)
        return notes
    }

    private fun findTopBottomAndWidth(elem: MatOfPoint): Triple<Double, Double, Double> {
        val bottom = elem.toArray().maxOf { point -> point.y }
        val top = elem.toArray().minOf { point -> point.y }
        val left = elem.toArray().minOf { point -> point.x }
        val right = elem.toArray().maxOf { point -> point.x }
        return Triple(top, bottom, right - left)
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

    private fun loadImage(imagePath: String): Mat? {
        return Imgcodecs.imread(imagePath)
    }

    private fun saveImage(path: String, image: Mat) {
        Imgcodecs.imwrite(path, image)
    }
}

/**
 * Returns a list containing a Pair<Start, End> for each note. The unit of Start and End are
 * pixels.
 */
fun getKeyTimeline(
    key: List<List<Pair<Double, Double>>>,
    speed: Double,
    height: Double
): List<Pair<Double, Double>> {
    val timeline = mutableListOf<Pair<Double, Double>>()
    var currentTime = 0.0
    // add a timestamp for every note and frame, they are probably overlapping quite a lot
    for (frame in key) {
        frame.forEach { note ->
            val start = currentTime + (height - note.second)
            val end = currentTime + (height - note.first)
            timeline.add(Pair(start, end))
        }

        currentTime += speed
    }

    if (timeline.isEmpty()) {
        return listOf()
    }

    // in: (top, bottom) (0, 24)
    // -> (top, bottom) (100, 76)
    // sort notes to make finding overlapping ones easier
    timeline.sortBy { note -> note.first }
    // combine overlapping notes
    val cleanTimeline = mutableListOf(timeline.first())
    var previousEnd: Double =
        timeline.firstOrNull()?.second ?: (height + 1) // bigger than anything possible
    for (note in timeline) {
        if (previousEnd >= note.first) {
            // this note is the same as the previous, thus we might have to extend the previous
            if (note.second < previousEnd) {
                // extend the previous one
                val last = cleanTimeline.removeLast()
                cleanTimeline.add(Pair(last.first, note.second))
                previousEnd = note.second
            }
        } else {
            // this is a new note
            cleanTimeline.add(note)
            previousEnd = note.second
        }
    }

    return cleanTimeline
}


/**
 * If speed is negative then it is invalid.
 */
private fun estimateSpeed(notes: List<Map<Int, List<Pair<Double, Double>>>>): Double {
    var speed = -1.0
    var startNote = Pair(Double.MAX_VALUE, Double.MAX_VALUE)
    var endNote = Pair(Double.MAX_VALUE, Double.MAX_VALUE)
    var searchKey = -1
    var startFrame = -1
    for (f in notes.indices) {
        val noteFrame = notes[f]
        // find frame with at least one note
        if (noteFrame.isNotEmpty()) {
            // find lowest up note (easiest to find in the next frame)
            for ((key, notes) in noteFrame) {
                notes.forEach { point ->
                    if (point.second < startNote.second) {
                        startNote = point
                        searchKey = key
                    }
                }
            }
        }
        if (searchKey != -1) {
            startFrame = f
            break
        }
    }

    // check if we found a valid startFrame
    if (startFrame in 0 until (notes.size - 1)) { // exclude one frame as we need one after the startFrame
        val noteFrame = notes[startFrame + 1]
        if (noteFrame.containsKey(searchKey)) {
            val keyNotes = noteFrame[searchKey]
            keyNotes?.forEach { point ->
                if (point.second < endNote.second) {
                    endNote = point
                }
            }
        }
    }

    if (startNote.second != Double.MAX_VALUE && endNote.second != Double.MAX_VALUE) {
        speed = endNote.second - startNote.second
    }

    return speed
}

fun convertNotesToTimestamps(
    // list of frame maps of keyIndex -> List of notes(top, bottom)
    notes: List<Map<Int, List<Pair<Double, Double>>>>,
    keyCount: Int,
    height: Double
): Timeline {
    //val speed = estimateSpeed(notes) // TODO: not perfect but probably good enough for now
    val speed = 12.3 / 2.0
    val keyFocused = convertIntoKeyFocused(notes, keyCount)
    val timeline = mutableListOf<List<Pair<Double, Double>>>()
    return getTimeline(keyFocused, speed, height, timeline)
}

fun getTimeline(
    keyFocused: List<List<List<Pair<Double, Double>>>>,
    speed: Double,
    height: Double,
    timeline: MutableList<List<Pair<Double, Double>>>
): Timeline {
    var counter = 0
    for (key in keyFocused) {
        val keyTimeline = getKeyTimeline(key, speed, height)
        //println("for key $counter, timeline: ${keyTimeline.joinToString(separator = ", ")}")
        timeline.add(keyTimeline)
        counter++
    }
    return Timeline(timeline)
}

fun convertIntoKeyFocused(
    notes: List<Map<Int, List<Pair<Double, Double>>>>,
    keyCount: Int
): List<List<List<Pair<Double, Double>>>> {
    // Structure: Key: frame: notes: note
    val frameCount = notes.size
    val keyFocused = mutableListOf<MutableList<MutableList<Pair<Double, Double>>>>()
    for (k in 0 until keyCount) {
        val keyList = mutableListOf<MutableList<Pair<Double, Double>>>()
        for (f in 0 until frameCount) {
            keyList.add(mutableListOf())
        }
        