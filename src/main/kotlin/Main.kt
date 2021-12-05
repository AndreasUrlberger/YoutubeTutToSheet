import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import nu.pattern.OpenCV
import org.opencv.core.*
import org.opencv.core.Core.*
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
    Main(args[0], args[1]).start(args[2])
}

class Main(private val filepathInput: String, private val filepathOutput: String) {
    //Instantiating the ImageCodecs class
    val imageCodecs = Imgcodecs()

    init {
        OpenCV.loadLocally()
    }

    fun start(command: String) {
        when (command) {
            "detectNotesInVideo" -> detectNotesInVideo()
            "createMidi" -> createMidiFromMarkers(filepathInput.toDouble(), (1080 * 0.25).toInt())
            "appendImages" -> appendImages()
            else -> println("does not recognize the command '$command'")
        }
    }

    private fun appendImages() {
        var imgOrinal = loadImage("./output/testFrame141.jpg") ?: throw IllegalArgumentException()
        val img1 = imgOrinal.submat(0, (imgOrinal.height() * 0.75).toInt(), 0, imgOrinal.width())
        val img2 = loadImage("./output/testFrame142.jpg") ?: throw IllegalArgumentException()
        val thresholds = doubleArrayOf(80.0, 85.0, 80.0)
        val channels1 = mutableListOf<Mat>()
        val channels2 = mutableListOf<Mat>()
        split(img1, channels1)
        split(img2, channels2)
        val thresh1 = computeAddedThresh(channels1, thresholds)
        val thresh2 = computeAddedThresh(channels2, thresholds)
        val result = Mat()
        matchTemplate(thresh1, thresh2, result, TM_SQDIFF)
        val mmr = Core.minMaxLoc(result)
        println("minMax: ${mmr.minLoc}")
        val cut = img2.submat(0, mmr.minLoc.y.roundToInt(), 0, img2.width())
        val out = Mat()
        vconcat(mutableListOf(cut, imgOrinal), out)
        saveImage("./output/result.jpg", result)
        saveImage("./output/appended.jpg", out)
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
        createAndPlayMidi(shiftedTimeline, 52.0)
    }

    private fun createMidiFromMarkers(playSpeed: Double, imgHeight: Int) {
        val markersString: String
        FileInputStream("./output/noteMarkers.json").use {
            markersString = it.readAllBytes().decodeToString()
        }
        val markers = Json.decodeFromString<NoteMarkers>(markersString)
        val speed = estimateSpeed(markers.notes, imgHeight)
        println("estimated Speed $speed")
        if (speed == -1.0) {
            throw IllegalStateException("could not get any speed information")
        }
        markers.notes.forEachIndexed { index, keyMarkers ->
            if (index == 23) {
                var (sum, weight) = estimateKeySpeed(keyMarkers, imgHeight)
                val keySpeed = sum / weight
                println("overall speed: $speed, keySpeed: $keySpeed, difference: ${keySpeed - speed}")
                println("key $index markers: ${keyMarkers.joinToString(separator = ", ")}")
                var differences = ""
                var lastMarker = -speed
                var currentOff = 0.0
                sum = 0.0
                var times = 0
                keyMarkers.forEach { marker ->
                    if (marker.isNotEmpty()) {
                        val end = marker.getOrNull(0)?.first ?: lastMarker
                        sum += end - lastMarker
                        times++
                        differences += ", ${end - currentOff}"
                        currentOff += speed
                        lastMarker = end
                    }
                }
                println("should have speed: ${sum / times}")
                println("key $index differences: $differences")
            }
        }
        val timeline = mutableListOf<List<Pair<Double, Double>>>()
        val actualTimeline =
            getTimeline(markers.notes, speed, (1080 * 0.25).toInt().toDouble(), timeline)
        actualTimeline.timeline.forEachIndexed { index, keyTimeline ->
            if (index == 23) {
                println("key $index timeline: ${keyTimeline.joinToString(separator = ", ")}")
            }
        }
        val singleTimeline = convertInSingleTimeline(actualTimeline)
        val shiftedTimeline = shiftTimelineToStart(singleTimeline)
        val fps = 15.0
        val bpm = 104.0
        val bps = bpm / 60.0
        val res = 480.0
        val pxPerSec = speed * fps
        val pxPerBeat = pxPerSec / bps
        val playSpeed = pxPerBeat / res
        createAndPlayMidi(shiftedTimeline, playSpeed)
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
            val frameWidth = 1920
            val frameHeight = (1080 * 0.25).toInt()
            val cap = VideoCapture(filepathInput)
            val keys = mutableListOf<Pair<Double, Double>>()
            val (_, keyboardWidth) = initKeys('a', -3, 'c', 5, keys)
            val pixelsPerInch = frameWidth / keyboardWidth
            val keyBorders = keys.asSequence().map { old ->
                Pair(
                    old.first * pixelsPerInch,
                    ((old.second * pixelsPerInch).coerceAtMost(frameWidth.toDouble()))
                )
            }.toList()
            val notes = mutableListOf<MutableList<MutableList<Pair<Double, Double>>>>()
            keyBorders.indices.forEach { _ -> notes.add(mutableListOf()) }

            val frame = Mat()
            var cut = Mat()

            cap.read(frame)
            var counter = 0
            while (!frame.empty()) {
                //if (counter % 10 == 0)
                println("processing frame #$counter")
                // Add a frame entry for each key
                notes.forEach { keyList -> keyList.add(mutableListOf()) }
                cut = frame.submat(frameHeight, frameHeight * 2, 0, frameWidth)
                detectNotesInImage(cut, keyBorders, notes, counter)

                for (i in 0 until 1) { // for every frame skip 3 more -> fps / 4
                    cap.read(frame)
                }
                cap.read(frame)
                counter++
            }

            cap.release()


            // Save data
            if (frameHeight <= 0) {
                println("Failed at processing the images")
            } else {
                val noteObject = NoteMarkers(notes)
                val noteString = Json.encodeToString(noteObject)
                FileOutputStream("./output/noteMarkers.json").use {
                    it.write(noteString.toByteArray())
                }

                val timeline = convertNotesToTimestamps(notes, frameHeight.toDouble())
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

    private fun createAndPlayMidi(timeline: List<Triple<Double, Int, Boolean>>, playSpeed: Double) {
        println("create midi")
        val sequencer = MidiSystem.getSequencer()
        sequencer.open()
        // Creating a sequence.
        val sequence = javax.sound.midi.Sequence(javax.sound.midi.Sequence.PPQ, 480)
        // PPQ(Pulse per ticks) is used to specify timing, type and 4 is the timing resolution.
        // Creating a track on our sequence upon which MIDI events would be placed
        val track = sequence.createTrack()

        val baseNote = 21 // the code of the lowest note on the piano, the A2
        val timelineSorted = timeline.sortedBy { it.first }
        // Add events
        timelineSorted.forEach { event ->
            println(
                "event at ${event.first.div(playSpeed)}, rounded ${
                    event.first.div(playSpeed).roundToInt()
                }"
            )
            if (event.third) {
                // on
                track.add(
                    makeEvent(
                        NOTE_ON,
                        1,
                        event.second + baseNote,
                        100,
                        event.first.div(playSpeed).toInt()
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
                        event.first.div(playSpeed).toInt()
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

    private fun detectNotesInImage(
        img: Mat,
        keyBorders: List<Pair<Double, Double>>,
        notes: MutableList<MutableList<MutableList<Pair<Double, Double>>>>,
        frameIndex: Int
    ) {
        val channels: MutableList<Mat> = mutableListOf()
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()

        val white = Scalar(255.0, 255.0, 255.0)
        line(img, Point(0.0, 0.0), Point(img.width().toDouble(), 0.0), white, 1)
        line(
            img,
            Point(0.0, img.height().minus(1).toDouble()),
            Point(img.width().toDouble(), img.height().minus(1).toDouble()),
            white,
            1
        )
        split(img, channels)

        val slices = mutableListOf<Mat>()
        val thresholds = doubleArrayOf(80.0, 85.0, 80.0)
        keyBorders.forEachIndexed { keyIndex, border ->
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
                            rectangle(
                                img,
                                Point(border.first, top - 1),
                                Point(border.second, bottom + 1),
                                Scalar(0.0, 0.0, 255.0),
                                2
                            )
                            // We only look at inner contours, so it is safe to extend them by one
                            notes[keyIndex][frameIndex].add(Pair(top - 1, bottom + 1))
                        }
                    }
                }
            }
        }
        //saveImage("./slices/slice_$frameIndex.jpg", img)
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
private fun estimateSpeed(notes: List<List<List<Pair<Double, Double>>>>, imgHeight: Int): Double {
    var speedSum = 0.0
    var weights = 0
    notes.forEach { key ->
        val (offset, weight) = estimateKeySpeed(key, imgHeight)
        speedSum += offset
        weights += weight
    }

    if (weights == 0) {
        return -1.0
    } else {
        return speedSum / weights
    }
}

private fun estimateKeySpeed(
    key: List<List<Pair<Double, Double>>>,
    imgHeight: Int
): Pair<Double, Int> {
    if (key.size < 2) {
        return 0.0 to 0
    }
    var offsetSum = 0.0
    var lastFrame = listOf<Pair<Double, Double>>()
    var divisor = 0 // the amount of times we actually calculated the cross offset
    for (frame in key) {
        if (frame.isNotEmpty() && lastFrame.isNotEmpty()) {
            val offset = getCrossOffset(lastFrame, frame, imgHeight)
            offsetSum += offset
            ++divisor
        }
        lastFrame = frame
    }

    return offsetSum to divisor
}

fun convertNotesToTimestamps(
    // list of frame maps of keyIndex -> List of notes(top, bottom)
    notes: List<List<List<Pair<Double, Double>>>>,
    height: Double
): Timeline {
    val speed =
        estimateSpeed(notes, height.toInt()) // TODO: not perfect but probably good enough for now
    println("estimated Speed $speed")
    if (speed == -1.0) {
        return Timeline(listOf())
    }
    val timeline = mutableListOf<List<Pair<Double, Double>>>()
    return getTimeline(notes, speed, height, timeline)
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
        keyFocused.add(keyList)
    }

    notes.forEachIndexed { fIndex, frame ->
        frame.forEach { (kIndex, key) ->
            key.sortedBy { note -> note.second }.forEach { note ->
                keyFocused[kIndex][fIndex].add(note)
            }
        }
    }

    return keyFocused
}

fun getCrossOffset(
    frame1: List<Pair<Double, Double>>,
    frame2: List<Pair<Double, Double>>,
    imgHeight: Int
): Double {
    val end = imgHeight
    var biggestCorrelation = 0.0
    val bestScores = mutableListOf<Int>()
    for (offset in 1 until end) {
        val correlation = calculateCrossCorrelation(frame1, frame2, offset)
        if (correlation > biggestCorrelation) {
            biggestCorrelation = correlation
            bestScores.clear()
            bestScores.add(offset)
        } else if (correlation == biggestCorrelation) {
            bestScores.add(offset)
        }
    }
    return bestScores.average()
}

fun calculateCrossCorrelation(
    frame1: List<Pair<Double, Double>>,
    frame2: List<Pair<Double, Double>>,
    offset: Int
): Double {
    val points = mutableListOf<Pair<Double, Boolean>>()
    frame1.forEach { note ->
        points.add(Pair(note.first + offset, true))
        points.add(Pair(note.second + offset, true))
    }
    frame2.forEach { note ->
        points.add(Pair(note.first, false))
        points.add(Pair(note.second, false))
    }

    var equalDistance = 0.0
    points.sortBy { it.first }
    var frame1Open = false
    var frame2Open = false
    var lastPoint = 0.0
    points.forEach { point ->
        if (frame1Open && frame2Open) { // only add to score if open fields overlap
            equalDistance += point.first - lastPoint
        }
        if (point.second) { // frame1
            frame1Open = !frame1Open
        } else {
            frame2Open = !frame2Open
        }

        lastPoint = point.first
    }

    val potentialOverlap =
        (frame1.sumOf { it.second - it.first } + frame2.sumOf { it.second - it.first }) / 2.0
    return equalDistance / potentialOverlap // gets a value between 0 and 1
}

@Serializable
data class Timeline(val timeline: List<List<Pair<Double, Double>>>)

@Serializable
data class NoteMarkers(val notes: MutableList<MutableList<MutableList<Pair<Double, Double>>>>)