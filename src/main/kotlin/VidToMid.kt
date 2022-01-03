import nu.pattern.OpenCV
import org.opencv.core.*
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.videoio.VideoCapture
import org.opencv.videoio.Videoio.CAP_PROP_POS_FRAMES
import java.io.File
import java.io.FileInputStream
import java.io.FileNotFoundException
import java.io.FileOutputStream
import java.nio.file.Files
import javax.sound.midi.MidiEvent
import javax.sound.midi.MidiSystem
import javax.sound.midi.ShortMessage
import kotlin.io.path.Path
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.system.exitProcess


fun main() {
    val conv = VidToMid("input/arrival_short.mp4", VidSettings(""), false)
    conv.getOffsets(30.0, 2)
    conv.mergeImages(2)
    conv.imageToMidi(30.0, 2, 154.0)
}

class VidToMid(
    vidPath: String,
    var settings: VidSettings,
    var safeToStorage: Boolean,
) {

    init {
        OpenCV.loadLocally()
    }

    private val cap: VideoCapture
    private var appendedImage: Mat? = null

    init {
        if (Files.notExists(Path(vidPath)))
            throw FileNotFoundException("Could not find video at '$vidPath'")
        cap = VideoCapture(vidPath)
    }


    private var FRAMES_TO_SKIP = 0
    private var INPUT_FPS = 0.0
    private fun fps() = INPUT_FPS / (FRAMES_TO_SKIP + 1)
    private var BPM = 0.0
    private var MIDI_RES = 0.0

    private var KEY_START = 'a'
    private var KEY_START_NUM = -3
    private var KEY_END = 'c'
    private var KEY_END_NUM = 5

    private val realKeyDimensions = listOf(
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

    private val simpleKeyDimensions = listOf(
        // (left position relative to the start of last octave, width of the key, isWhiteKey)
        Triple(0.0, 1 / 7.0, true), // c
        // (left position relative to end of last white key, same for the right position, isWhiteKey)
        Triple(-0.35 / 6.5, 0.15 / 6.5, false), // cis
        Triple(1 / 7.0, 1 / 7.0, true), // d
        Triple(-0.15 / 6.5, 0.35 / 6.5, false), // dis
        Triple(2 / 7.0, 1 / 7.0, true), // e
        Triple(3 / 7.0, 1 / 7.0, true), // f
        Triple(-0.4 / 6.5, 0.1 / 6.5, false), // fis
        Triple(4 / 7.0, 1 / 7.0, true), // g
        Triple(-0.25 / 6.5, 0.25 / 6.5, false), // gis
        Triple(5 / 7.0, 1 / 7.0, true), // a
        Triple(-0.1 / 6.5, 0.4 / 6.5, false), // ais
        Triple(6 / 7.0, 1 / 7.0, true), // h
    ).map { Triple(it.first.times(6.5), it.second.times(6.5), it.third) }


    // Sheet Music Boss
    private val imageRangeSHB = Pair(0.25, 0.5)
    private val prepareImageSHB: (Mat) -> Mat = { img ->
        val thresholds = doubleArrayOf(80.0, 85.0, 80.0)
        val channels = mutableListOf<Mat>()
        Core.split(img, channels)
        computeAddedThresh(channels, thresholds)
    }
    private val extractNotesSMB: (Mat) -> Mat = { img ->
        val thresholds = doubleArrayOf(80.0, 85.0, 80.0)
        val slices = mutableListOf<Mat>()
        Core.split(img, slices)
        computeAddedThresh(slices, thresholds)
    }
    private val maxImageDiffSMB = 0.25
    private val keyboardDistOriginSMB = 0.47
    private val keyboardDistCoeffSMB = 0.043

    // Patrick Pietschmann
    private val imageRangePP = Pair(0.0, 0.5)
    private val prepareImagePP: (Mat) -> Mat = { img ->
        val thresh = extractPatrickNotes(img)
        val start = (thresh.height() * 0.25).toInt()
        val end = (thresh.height() * 0.75).toInt()
        thresh.submat(start, end, 0, thresh.width())
    }
    private val extractNotesPP: (Mat) -> Mat = ::extractPatrickNotes
    private val maxImageDiffPP = 0.50
    private val keyboardDistOriginPP = 0.47
    private val keyboardDistCoeffPP = 0.043

    // (Pair<StartPercent, EndPercent>)
    private var FRAME_LIMITS = imageRangePP

    // Lambda that modifies an image in a way that it makes it easier to find the notes
    private var prepareImage: (Mat) -> Mat = prepareImagePP
    private var extractNotes: (Mat) -> Mat = extractNotesPP

    private var MAX_IMAGE_DIFF = maxImageDiffPP
    private var KEYBOARD_DIST_COEFF = keyboardDistCoeffPP
    private var KEYBOARD_DIST_ORIGIN = keyboardDistOriginPP
    private var KEY_DIMENSIONS = simpleKeyDimensions

    fun getOffsets(fps: Double, framesToSkip: Int) {
        INPUT_FPS = fps
        FRAMES_TO_SKIP = framesToSkip
        getVideoOffsets()
    }

    fun mergeImages(framesToSkip: Int) {
        FRAMES_TO_SKIP = framesToSkip
        recMergeImages()
    }

    fun imageToMidi(fps: Double, framesToSkip: Int, bpm: Double, midiRes: Double = 480.0) {
        INPUT_FPS = fps
        FRAMES_TO_SKIP = framesToSkip
        BPM = bpm
        MIDI_RES = midiRes
        longImageToMidi()
    }

    fun loadSettingsFile(settings: VidSettings) {
        this.settings = settings
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
        val tunedKeyBorders = tuneKeyBorders(keys)
        val keyBorders = tunedKeyBorders.asSequence().map { old ->
            Pair(
                old.first * pixelsPerInch,
                ((old.second * pixelsPerInch).coerceAtMost(img.width().toDouble()))
            )
        }.toList()
        val notes = mutableListOf<KeyEvent>()
        detectNotesInImagePP(img, keyBorders, notes)

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
        cap.set(CAP_PROP_POS_FRAMES, 0.0) // make sure we are at the start
        val frame = Mat()
        var oldFrame: Mat
        val offsets = mutableListOf<Double>()
        getFrame(cap, frame, true)
        if (!frame.empty()) {
            oldFrame = prepareImage(frame)
            oldFrame = oldFrame.submat(0, (oldFrame.height() * 0.75).toInt(), 0, oldFrame.width())
            skipFrames(cap, FRAMES_TO_SKIP, frame)
            getFrame(cap, frame, true)
            var counter = 0
            while (!frame.empty() && counter < 1_000_000) {
                if (counter % 20 == 0) {
                    println("processing image #$counter")
                }
                if (counter % 100 == 0) {
                    System.gc()
                }

                val result = Mat()
                val prepFrame = prepareImage(frame)
                //saveImage("slices/slice$counter.jpg", prepFrame)
                Imgproc.matchTemplate(oldFrame, prepFrame, result, Imgproc.TM_SQDIFF)
                val mmr = Core.minMaxLoc(result)
                offsets.add(mmr.minLoc.y)
                println("$counter minMax: ${mmr.minLoc.y}")
                //saveImage("./output/thresh$counter.jpg", thresh1)

                oldFrame =
                    prepFrame.submat(0, (prepFrame.height() * 0.75).toInt(), 0, prepFrame.width())
                skipFrames(cap, FRAMES_TO_SKIP, frame)
                getFrame(cap, frame, true)
                counter++
            }
        }
        cap.release()

        val median = median(offsets)
        val correctedOffsets =
            offsets.map { value -> if (abs(value - median) > median * MAX_IMAGE_DIFF) median else value }
                .toList()

        FileOutputStream("./output/offsetsCorrected.txt").use {
            it.write(correctedOffsets.joinToString(", ").toByteArray())
        }
    }

    private fun skipFrames(cap: VideoCapture, framesToSkip: Int, placeholder: Mat? = null) {
        val img = placeholder ?: Mat()
        for (i in 0 until framesToSkip) {
            cap.read(img)
        }
    }

    private fun median(list: List<Double>) = list.sorted().let {
        if (it.size % 2 == 0)
            (it[it.size / 2] + it[(it.size - 1) / 2]) / 2
        else
            it[it.size / 2]
    }

    private fun recMergeImages() {
        val offsets = FileInputStream("output/offsetsCorrected.txt").use {
            it.readAllBytes().decodeToString().splitToSequence(", ").map { elem -> elem.toDouble() }
                .toList()
        }
        cap.set(CAP_PROP_POS_FRAMES, 0.0) // make sure we are at the start
        val frames = mutableListOf<Mat>()
        val startFrame = Mat()
        getFrame(cap, startFrame)
        Imgproc.cvtColor(startFrame, startFrame, Imgproc.COLOR_BGR2GRAY)
        adaptThresh(startFrame, 7, 7.0).copyTo(startFrame)
        frames.add(startFrame)
        val small = Mat()
        // save first image
        skipFrames(cap, FRAMES_TO_SKIP, small)
        for ((counter, offset) in offsets.withIndex()) {
            getFrame(cap, small)
            Imgproc.cvtColor(small, small, Imgproc.COLOR_BGR2GRAY)
            adaptThresh(small, 7, 7.0).copyTo(small)
            val cut = small.submat(0, offset.plus(1).roundToInt(), 0, small.width())
            println("adds frame for offset: $counter")
            frames.add(cut)
            skipFrames(cap, FRAMES_TO_SKIP, small)
            if (counter % 100 == 0) {
                System.gc()
            }
        }

        frames.reverse()

        val fullImage = Mat()
        println("now concat images")
        Core.vconcat(frames, fullImage)
        saveImage("output/appended.bmp", fullImage)
    }

    private fun adaptThresh(channel: Mat, blockSize: Int, C: Double): Mat {
        val adaptThresh = Mat()
        // BORDER_REPLICATE | #BORDER_ISOLATED
        Imgproc.adaptiveThreshold(
            channel,
            adaptThresh,
            255.0,
            Core.BORDER_REPLICATE,
            Imgproc.THRESH_BINARY_INV,
            blockSize,
            C
        )
        return adaptThresh
    }

    /**
     * Gets the next frame from the given VideoCapture and cuts it to the given
     * limits
     * @param cap The VideoCapture to take the frame from.
     * @param frame The Mat to fill.
     */
    private fun getFrame(cap: VideoCapture, frame: Mat, cut: Boolean = false) {
        cap.read(frame)
        if (cut && !frame.empty()) {
            val start = (frame.height() * FRAME_LIMITS.first).toInt()
            val end = (frame.height() * FRAME_LIMITS.second).toInt()
            Mat(frame, Range(start, end)).copyTo(frame)
        }
    }

    private fun initKeys(
        outKeys: MutableList<Pair<Double, Double>>
    ): Pair<Int, Double> {
        val keyCodes = mapOf('a' to 9, 'h' to 11, 'c' to 0, 'd' to 2, 'e' to 4, 'f' to 5, 'g' to 7)
        checkBorderKeys(KEY_START_NUM, KEY_END_NUM, KEY_START, KEY_END)

        var whiteKeys = 0
        var xOffset = 0.0
        var width = 0.0
        // lower single keys
        for (key in (keyCodes[KEY_START] ?: 0)..(keyCodes.values.maxOrNull() ?: 0)) {
            val dim = KEY_DIMENSIONS[key]
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
            for (key in KEY_DIMENSIONS.indices) {
                val dim = KEY_DIMENSIONS[key]
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
            val dim = KEY_DIMENSIONS[key]
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

    private fun tuneKeyBorders(borders: List<Pair<Double, Double>>): List<Pair<Double, Double>> {
        val width = borders.maxOf { it.second }
        return borders.map { (left, right) ->
            val aLeft = distort(left, KEYBOARD_DIST_ORIGIN, width, KEYBOARD_DIST_COEFF)
            val aRight = distort(right, KEYBOARD_DIST_ORIGIN, width, KEYBOARD_DIST_COEFF)
            Pair(aLeft, aRight)
        }
    }

    private fun distort(pos: Double, distortionOrigin: Double, width: Double, k: Double): Double {
        val x = pos / width
        val left = (distortionOrigin).pow(3) * k
        val right = (1 - distortionOrigin).pow(3) * k
        val newLength = 1 - (left + right)
        val distance = (x - distortionOrigin)
        val offset = (distance).pow(3) * k
        return (x - offset - left) / (newLength / width)
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
        val sequencer = MidiSystem.getSequencer()
        sequencer.open()
        // Creating a sequence.
        // PPQ(Pulse per ticks) is used to specify timing, type and 4 is the timing resolution.
        val sequence = javax.sound.midi.Sequence(javax.sound.midi.Sequence.PPQ, MIDI_RES.toInt())

        val baseNote = 21 // the code of the lowest note on the piano, the A2
        val timelineSorted = timeline.sortedBy { it.pos }

        // Creating a track on our sequence upon which MIDI events would be placed
        val track = sequence.createTrack()
        // Add events
        timelineSorted.forEach { event ->
            track.add(
                makeEvent(
                    if (event.type) ShortMessage.NOTE_ON else ShortMessage.NOTE_OFF,
                    //event.hand,
                    0,
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

    private fun detectNotesInImagePP(
        img: Mat,
        keyBorders: List<Pair<Double, Double>>,
        notes: MutableList<KeyEvent>,
    ) {
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY)
        keyBorders.forEachIndexed { keyIndex, border ->
            val bonusPx = 3
            val sliceWidth = (border.second - border.first)
            val widthThreshold = sliceWidth * 0.55
            val borderStart =
                (border.first - bonusPx).roundToInt().coerceIn(0, img.width())
            val borderEnd = (border.second + bonusPx).roundToInt().coerceIn(0, img.width())
            val slice = img.submat(0, img.height(), borderStart, borderEnd)
            //saveImage("slices/slice${keyIndex}s.jpg", slice)

            val contours = mutableListOf<MatOfPoint>()
            val hierarchy = Mat()
            val potNotes = mutableListOf<Rect>()

            Imgproc.findContours(
                slice,
                contours,
                hierarchy,
                Imgproc.RETR_CCOMP,
                Imgproc.CHAIN_APPROX_SIMPLE
            )
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
                val bounding = Imgproc.boundingRect(contours[x])
                if (bounding.width >= widthThreshold) {
                    potNotes.add(
                        Rect(
                            border.first.toInt() + bounding.x - bonusPx,
                            bounding.y,
                            bounding.width,
                            bounding.height
                        )
                    )
                }
            }

            val innerNotes = mutableListOf<Rect>()
            potNotes.forEach { one ->
                potNotes.forEach { other ->
                    if (one.contains(other.tl()) && one.contains(other.br())) {
                        innerNotes.add(other)
                    }
                }
            }
            potNotes.removeAll(innerNotes)
            for (note in potNotes) {
                /*val middleX = ((note.br().y + note.tl().y) / 2).toInt()
                val middleY = ((note.tl().x + note.br().x) / 2 + border.first - bonusPx).toInt()
                val hand = getHand(img.get(middleX, middleY))*/

                // We only look at inner contours, so it is safe to extend them by one
                val down = img.height() - (note.br().y + 1)
                val up = img.height() - (note.tl().y - 1)
                notes.add(KeyEvent(down, keyIndex, true, 0))
                notes.add(KeyEvent(up, keyIndex, false, 0))
                Imgproc.rectangle(img, note, Scalar(255.0, 255.0, 255.0), Core.FILLED)
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
        Imgproc.threshold(channels[0], threshBlue, limitBlue, 255.0, Imgproc.THRESH_BINARY)
        Imgproc.threshold(channels[1], threshGreen, limitGreen, 255.0, Imgproc.THRESH_BINARY)
        Imgproc.threshold(channels[2], threshRed, limitRed, 255.0, Imgproc.THRESH_BINARY)

        val addedThresh = Mat()
        Core.bitwise_and(threshBlue, threshGreen, addedThresh)
        Core.bitwise_and(addedThresh, threshRed, addedThresh)
        return addedThresh
    }

    private fun makeEvent(
        command: Int,
        channel: Int,
        note: Int,
        velocity: Int,
        tick: Int
    ): MidiEvent {
        val a = ShortMessage()
        a.setMessage(command, channel, note, velocity)
        return MidiEvent(a, tick.toLong())
    }

    private fun loadImage(imagePath: String): Mat {
        return Imgcodecs.imread(imagePath)
            ?: throw IllegalArgumentException("Could not find image at path '$imagePath'")
    }

    private fun saveImage(path: String, image: Mat) {
        Imgcodecs.imwrite(path, image)
    }

    private data class KeyEvent(val pos: Double, val key: Int, val type: Boolean, val hand: Int)
}