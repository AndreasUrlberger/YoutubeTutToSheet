import nu.pattern.OpenCV
import org.opencv.core.*
import org.opencv.core.Core.*
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc.*
import org.opencv.videoio.VideoCapture
import java.io.File
import java.io.FileInputStream
import java.io.FileNotFoundException
import java.io.FileOutputStream
import java.nio.file.Files
import javax.sound.midi.MidiEvent
import javax.sound.midi.MidiSystem
import javax.sound.midi.ShortMessage
import javax.sound.midi.ShortMessage.NOTE_OFF
import javax.sound.midi.ShortMessage.NOTE_ON
import kotlin.io.path.Path
import kotlin.math.abs
import kotlin.math.roundToInt
import kotlin.system.exitProcess

private var IMAGES_TO_SKIP = 0
private var INPUT_FPS = 0.0
private fun fps() = INPUT_FPS / (IMAGES_TO_SKIP + 1)
private var BPM = 0.0
private var MIDI_RES = 0.0

private var VIDEO_PATH = ""

private var KEY_START = 'a'
private var KEY_START_NUM = -3
private var KEY_END = 'c'
private var KEY_END_NUM = 5

fun main(args: Array<String>) {
    OpenCV.loadLocally()
    when (args[0]) {
        "getOffsets" -> {
            VIDEO_PATH = args[1]
            if (Files.notExists(Path(VIDEO_PATH)))
                throw FileNotFoundException("Could not find video at '$VIDEO_PATH'")
            INPUT_FPS = args[2].toDouble()
            IMAGES_TO_SKIP = args[3].toInt()
            getVideoOffsets()
        }
        "mergeImages" -> {
            VIDEO_PATH = args[1]
            if (Files.notExists(Path(VIDEO_PATH)))
                throw FileNotFoundException("Could not find video at '$VIDEO_PATH'")
            IMAGES_TO_SKIP = args[2].toInt()
            mergeImages()
        }
        "longImageToMidi" -> {
            INPUT_FPS = args[1].toDouble()
            IMAGES_TO_SKIP = args[2].toInt()
            BPM = args[3].toDouble()
            MIDI_RES = args.getOrNull(4)?.toDouble() ?: 480.0
            longImageToMidi()
        }

        else -> println("does not recognize the command '${args[0]}'")
    }
}

private fun longImageToMidi() {
    val img = loadImage("./input/longImage.bmp")
    if (img.empty())
        throw IllegalArgumentException("Image is empty")
    val offsets = FileInputStream("./output/offsetsCorrected.txt").use {
        it.readAllBytes().decodeToString().splitToSequence(", ").map { elem -> elem.toDouble() }
            .toList()
    }
    if (offsets.isEmpty())
        throw IllegalArgumentException("The offsets cannot be empty")

    // detect notes
    val keys = mutableListOf<Pair<Double, Double>>()
    val (_, keyboardWidth) = initKeys(keys)
    val pixelsPerInch = img.width() / keyboardWidth
    val keyBorders = keys.asSequence().map { old ->
        Pair(
            old.first * pixelsPerInch,
            ((old.second * pixelsPerInch).coerceAtMost(img.width().toDouble()))
        )
    }.toList()
    val notes = mutableListOf<KeyEvent>()
    detectNotesInImage(img, keyBorders, notes)

    // create Timeline
    val shiftedTimeline = shiftTimelineToStart(notes)

    // create midi
    val bps = BPM / 60.0
    val pxPerSec = offsets.average() * fps()
    println("pixel per second $pxPerSec")
    val pxPerBeat = pxPerSec / bps
    val playSpeed = (pxPerBeat / MIDI_RES)
    createMidi(shiftedTimeline, playSpeed)
}

private fun getVideoOffsets() {
    val cap = VideoCapture(VIDEO_PATH)
    val frame = Mat()
    var oldFrame: Mat
    val offsets = mutableListOf<Double>()
    getFrame(cap, frame)
    if (!frame.empty()) {
        var counter = 0
        oldFrame = frame.submat(0, (frame.height() * 0.75).toInt(), 0, frame.width())
        getFrame(cap, frame)
        while (!frame.empty() && counter < 1_000_000) {
            if (counter % 20 == 0) {
                println("processing image #$counter")
            }
            if (counter % 100 == 0) {
                System.gc()
            }

            val thresholds = doubleArrayOf(80.0, 85.0, 80.0)
            val channels1 = mutableListOf<Mat>()
            val channels2 = mutableListOf<Mat>()
            split(oldFrame, channels1)
            split(frame, channels2)
            val thresh1 = computeAddedThresh(channels1, thresholds)
            val thresh2 = computeAddedThresh(channels2, thresholds)
            val result = Mat()
            matchTemplate(thresh1, thresh2, result, TM_SQDIFF)
            val mmr = minMaxLoc(result)
            offsets.add(mmr.minLoc.y)
            //println("$counter minMax: ${mmr.minLoc}")
            //saveImage("./output/thresh$counter.jpg", thresh1)

            oldFrame = frame.submat(0, (frame.height() * 0.75).toInt(), 0, frame.width())
            for (i in 0 until IMAGES_TO_SKIP) {
                getFrame(cap, frame)
            }
            getFrame(cap, frame)
            counter++
        }
    }
    cap.release()

    val median = median(offsets)
    val correctedOffsets =
        offsets.map { value -> if (abs(value - median) > median * 0.25) median else value }
            .toList()

    FileOutputStream("./output/offsetsCorrected.txt").use {
        it.write(correctedOffsets.joinToString(", ").toByteArray())
    }
}

private fun median(list: List<Double>) = list.sorted().let {
    if (it.size % 2 == 0)
        (it[it.size / 2] + it[(it.size - 1) / 2]) / 2
    else
        it[it.size / 2]
}

private fun mergeImages() {
    val offsets = FileInputStream("./output/offsetsCorrected.txt").use {
        it.readAllBytes().decodeToString().splitToSequence(", ").map { elem -> elem.toDouble() }
            .toList()
    }
    val cap = VideoCapture(VIDEO_PATH)
    val frame = Mat()
    val bigFrame = Mat()
    cap.read(bigFrame)
    Mat(bigFrame, Range((bigFrame.height() * 0.25).toInt(), bigFrame.height())).copyTo(bigFrame)
    getFrame(cap, frame)
    var counter = 0
    while (!frame.empty() && counter < offsets.size) {
        println("merge image #$counter")
        val cut = frame.submat(0, offsets[counter].plus(1).roundToInt(), 0, frame.width())
        vconcat(mutableListOf(cut, bigFrame), bigFrame)

        for (i in 0 until IMAGES_TO_SKIP) {
            getFrame(cap, frame)
        }
        getFrame(cap, frame)
        counter++
        if (counter % 100 == 0) {
            System.gc()
        }
    }
    cap.release()
    saveImage("./output/appended.bmp", bigFrame)
}

private fun getFrame(cap: VideoCapture, frame: Mat) {
    cap.read(frame)
    if (!frame.empty())
        Mat(
            frame,
            Range((frame.height() * 0.25).toInt(), (frame.height() * 0.50).toInt())
        ).copyTo(frame)
}

private fun initKeys(
    outKeys: MutableList<Pair<Double, Double>>
): Pair<Int, Double> {
    val keyCodes = mapOf('a' to 9, 'h' to 11, 'c' to 0, 'd' to 2, 'e' to 4, 'f' to 5, 'g' to 7)
    checkBorderKeys(KEY_START_NUM, KEY_END_NUM, KEY_START, KEY_END)

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
    for (key in (keyCodes[KEY_START] ?: 0)..(keyCodes.values.maxOrNull() ?: 0)) {
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
    for (scale in (KEY_START_NUM + 1) until KEY_END_NUM) {
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
    for (key in (keyCodes.values.minOrNull() ?: 0)..(keyCodes[KEY_END] ?: 0)) {
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

private fun shiftTimelineToStart(timeline: List<KeyEvent>): List<KeyEvent> {
    if (timeline.isEmpty()) {
        return timeline
    }
    val offset = timeline.minOf { it.pos }
    return timeline.map { KeyEvent(it.pos - offset, it.key, it.type, it.hand) }.toList()
}

private fun createMidi(timeline: List<KeyEvent>, playSpeed: Double) {
    println("create midi")
    val sequencer = MidiSystem.getSequencer()
    sequencer.open()
    // Creating a sequence.
    val sequence = javax.sound.midi.Sequence(javax.sound.midi.Sequence.PPQ, MIDI_RES.toInt())
    // PPQ(Pulse per ticks) is used to specify timing, type and 4 is the timing resolution.


    val baseNote = 21 // the code of the lowest note on the piano, the A2
    val timelineSorted = timeline.sortedBy { it.pos }

    // Creating a track on our sequence upon which MIDI events would be placed
    val track = sequence.createTrack()
    // Add events
    timelineSorted.forEach { event ->
        track.add(
            makeEvent(
                if (event.type) NOTE_ON else NOTE_OFF,
                event.hand,
                event.key + baseNote,
                100,
                event.pos.div(playSpeed).roundToInt()
            )
        )
    }

    // Setting our sequence so that the sequencer can run it on synthesizer
    sequencer.sequence = sequence

    // Specifies the beat rate in beats per minute.
    sequencer.tempoInBPM = BPM.toFloat()

    println("storing midi file")
    val file = File("./output/midi.mid")
    MidiSystem.write(sequence, 0, file)
    exitProcess(1)
}

private fun detectNotesInImage(
    img: Mat,
    keyBorders: List<Pair<Double, Double>>,
    notes: MutableList<KeyEvent>,
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
                    val (vertical, horizontal) = findExtrema(contours[x])
                    val (top, bottom) = vertical
                    val (left, right) = horizontal
                    if (right - left >= widthThreshold) {
                        rectangle(
                            img,
                            Point(border.first - sliceWidth * 0.2 + left, top - 1),
                            Point(border.first - sliceWidth * 0.2 + right, bottom + 1),
                            Scalar(0.0, 0.0, 255.0),
                            3
                        )
                        // Get correct hand
                        val middleX = ((bottom + top) / 2).toInt()
                        val middleY = ((left + right) / 2 + border.first - sliceWidth * 0.2).toInt()
                        val hand = getHand(img.get(middleX, middleY))

                        // We only look at inner contours, so it is safe to extend them by one
                        notes.add(KeyEvent(img.height() - (bottom + 1), keyIndex, true, hand))
                        notes.add(KeyEvent(img.height() - (top - 1), keyIndex, false, hand))
                    }
                }
            }
        }
    }
    saveImage("./slices/slice.bmp", img)
}

private fun getHand(colors: DoubleArray): Int {
    // (blue, green, red)
    return if (colors[0] > colors[1]) { // more blue than green
        1
    } else {
        0
    }
}

fun findExtrema(elem: MatOfPoint): Pair<Pair<Double, Double>, Pair<Double, Double>> {
    val bottom = elem.toArray().maxOf { point -> point.y }
    val top = elem.toArray().minOf { point -> point.y }
    val left = elem.toArray().minOf { point -> point.x }
    val right = elem.toArray().maxOf { point -> point.x }
    return Pair(Pair(top, bottom), Pair(left, right))
}

fun computeAddedThresh(channels: MutableList<Mat>, weights: DoubleArray): Mat {
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

    val addedThresh = Mat()
    bitwise_and(threshBlue, threshGreen, addedThresh)
    bitwise_and(addedThresh, threshRed, addedThresh)
    return addedThresh
}

fun makeEvent(command: Int, channel: Int, note: Int, velocity: Int, tick: Int): MidiEvent {

    // ShortMessage stores a note as command type, channel,
    // instrument it has to be played on and its speed.
    val a = ShortMessage()
    a.setMessage(command, channel, note, velocity)

    // A midi event consists of a short message(representing
    // a note) and the tick at which that note has to be played

    return MidiEvent(a, tick.toLong())
}


fun loadImage(imagePath: String): Mat {
    return Imgcodecs.imread(imagePath)
        ?: throw IllegalArgumentException("Could not find image at path '$imagePath'")
}

fun saveImage(path: String, image: Mat) {
    Imgcodecs.imwrite(path, image)
}

data class KeyEvent(val pos: Double, val key: Int, val type: Boolean, val hand: Int)