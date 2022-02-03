import nu.pattern.OpenCV
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.videoio.VideoCapture
import org.opencv.videoio.Videoio.CAP_PROP_FPS
import org.opencv.videoio.Videoio.CAP_PROP_POS_FRAMES
import java.io.File
import java.io.FileInputStream
import java.io.FileNotFoundException
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.util.*
import javax.sound.midi.*
import kotlin.io.path.Path
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.system.exitProcess


fun main(args: Array<String>) {
    val settings = PietschmannSettings(
        bpm = args[0].toDouble(),
        midiRes = 480.0,
        framesToSkip = 2,
        timeSignatureDenominator = 4,
        timeSignatureNominator = 4,
    )
    val conv = VidToMid("input/Dune_Part2.mp4", settings, true)
    println("now calculate offset")
    conv.getOffsets()
    println("now merge images")
    conv.mergeImages()
    println("now detect notes")
    //conv.rasterizeNotes()
    conv.imageToMidi()
    conv.release()
}

class VidToMid(
    vidPath: String,
    private var conf: VidSettings,
    private var saveToStorage: Boolean,
) {

    init {
        OpenCV.loadLocally()
    }

    private val cap: VideoCapture
    private var appendedImage: Mat = Mat()
    private val notes = mutableListOf<KeyEvent>()
    private val offsets = mutableListOf<Double>()

    init {
        if (Files.notExists(Path(vidPath)))
            throw FileNotFoundException("Could not find video at '$vidPath'")
        cap = VideoCapture(vidPath)
    }

    private fun fps() = cap.get(CAP_PROP_FPS) / (conf.framesToSkip + 1)

    fun release() {
        cap.release()
    }

    fun getOffsets() {
        getVideoOffsets()
    }

    fun mergeImages() {
        if (saveToStorage) {
            offsets.clear()
            val loadedOffsets = FileInputStream("output/offsets.txt").use {
                it.readAllBytes().decodeToString().splitToSequence(", ")
                    .map { elem -> elem.toDouble() }
                    .toList()
            }
            offsets.addAll(loadedOffsets)
        }
        recMergeImages()
    }

    fun imageToMidi() {
        if (saveToStorage) {
            loadImage("output/appended.bmp").copyTo(appendedImage)
            Imgproc.cvtColor(appendedImage, appendedImage, Imgproc.COLOR_BGR2GRAY)

            offsets.clear()
            val loadedOffsets = FileInputStream("output/offsets.txt").use {
                it.readAllBytes().decodeToString().splitToSequence(", ")
                    .map { elem -> elem.toDouble() }
                    .toList()
            }
            offsets.addAll(loadedOffsets)
        }
        longImageToMidi()
    }

    fun renderLines(img: Mat) {
        val keyBorders = mutableListOf<Pair<Double, Double>>()
        val (_, keyboardWidth) = initKeys(keyBorders)
        val pixelsPerInch = img.width() / keyboardWidth
        keyBorders.replaceAll { old ->
            Pair(
                old.first * pixelsPerInch,
                ((old.second * pixelsPerInch).coerceAtMost(img.width().toDouble()))
            )
        }

        val top = (img.height() - 1).toDouble()
        val widthWhite = keyBorders.maxOf { it.second - it.first }
        tuneKeyBorders(keyBorders)
        keyBorders.forEach { (left, right) ->
            if ((right - left) > widthWhite * 0.8) {
                Imgproc.rectangle(
                    img,
                    Point(left, 0.0),
                    Point(right, top),
                    Scalar(0.0, 255.0, 0.0),
                    1
                )
            }
        }

        saveImage("output/lines.jpg", img)
    }

    fun loadSettingsFile(settings: VidSettings) {
        this.conf = settings
    }

    private fun longImageToMidi() {
        if (appendedImage.empty())
            throw IllegalArgumentException("Image is empty")
        if (offsets.isEmpty())
            throw IllegalArgumentException("The offsets cannot be empty")

        // detect notes
        val keyBorders = mutableListOf<Pair<Double, Double>>()
        val (_, keyboardWidth) = initKeys(keyBorders)
        val pixelsPerInch = appendedImage.width() / keyboardWidth
        tuneKeyBorders(keyBorders)
        keyBorders.replaceAll { old ->
            Pair(
                old.first * pixelsPerInch,
                ((old.second * pixelsPerInch).coerceAtMost(appendedImage.width().toDouble()))
            )
        }
        detectNotesInImage(keyBorders)

        // create Timeline
        shiftTimelineToStart(notes)

        // create midi
        val bps = conf.bpm / 60.0
        val pxPerSec = median(offsets) * fps()
        println("pixel per second $pxPerSec")
        val pxPerBeat = pxPerSec / bps
        val playSpeed = (pxPerBeat / conf.midiRes)
        createMidi(playSpeed)
    }

    private fun getVideoOffsets() {
        cap.set(CAP_PROP_POS_FRAMES, 0.0) // make sure we are at the start
        offsets.clear()
        val frame = Mat()
        var oldFrame: Mat
        getFrame(frame, true)
        if (!frame.empty()) {
            oldFrame = conf.prepareImage(frame)
            oldFrame = oldFrame.submat(0, (oldFrame.height() * 0.75).toInt(), 0, oldFrame.width())
            skipFrames(conf.framesToSkip, frame)
            getFrame(frame, true)
            var counter = 0
            while (!frame.empty() && counter < 1_000_000) {
                if (counter % 20 == 0) {
                    println("processing image #$counter")
                }
                if (counter % 100 == 0) {
                    System.gc()
                }

                val result = Mat()
                val prepFrame = conf.prepareImage(frame)
                Imgproc.matchTemplate(oldFrame, prepFrame, result, Imgproc.TM_SQDIFF)
                val mmr = Core.minMaxLoc(result)
                offsets.add(mmr.minLoc.y)

                oldFrame =
                    prepFrame.submat(0, (prepFrame.height() * 0.75).toInt(), 0, prepFrame.width())
                skipFrames(conf.framesToSkip, frame)
                getFrame(frame, true)
                counter++
            }
        }
        val median = median(offsets)
        offsets.replaceAll { value -> if (abs(value - median) > median * conf.maxImageDiff) median else value }

        if (saveToStorage) {
            FileOutputStream("./output/offsets.txt").use {
                it.write(offsets.joinToString(", ").toByteArray())
            }
        }
    }

    private fun skipFrames(framesToSkip: Int, placeholder: Mat? = null) {
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
        if (offsets.isEmpty()) {
            throw IllegalStateException("offsets have not been calculated yet")
        }
        cap.set(CAP_PROP_POS_FRAMES, 0.0) // make sure we are at the start
        val frames = mutableListOf<Mat>()
        val startFrame = Mat()
        getFrame(startFrame)
        Imgproc.cvtColor(startFrame, startFrame, Imgproc.COLOR_BGR2GRAY)
        adaptThresh(startFrame, 7, 7.0).copyTo(startFrame)
        frames.add(startFrame)
        val small = Mat()
        // save first image
        skipFrames(conf.framesToSkip, small)
        for ((counter, offset) in offsets.withIndex()) {
            getFrame(small)
            Imgproc.cvtColor(small, small, Imgproc.COLOR_BGR2GRAY)
            adaptThresh(small, 7, 7.0).copyTo(small)
            val cut = small.submat(0, offset.plus(1).roundToInt(), 0, small.width())
            frames.add(cut)
            skipFrames(conf.framesToSkip, small)
            if (counter % 100 == 0) {
                println("adds frame for offset: $counter")
                System.gc()
            }
        }

        frames.reverse()

        println("now concat images")
        Core.vconcat(frames, appendedImage)
        if (saveToStorage) {
            saveImage("output/appended.bmp", appendedImage)
        }
    }

    /**
     * Gets the next frame from the VideoCapture and cuts it to the given
     * limits
     * @param frame The Mat to fill.
     */
    private fun getFrame(frame: Mat, cut: Boolean = false) {
        cap.read(frame)
        if (cut && !frame.empty()) {
            val start = (frame.height() * conf.frameLimits.first).toInt()
            val end = (frame.height() * conf.frameLimits.second).toInt()
            Mat(frame, Range(start, end)).copyTo(frame)
        }
    }

    private fun initKeys(
        outKeys: MutableList<Pair<Double, Double>>
    ): Pair<Int, Double> {
        val keyCodes = mapOf('a' to 9, 'h' to 11, 'c' to 0, 'd' to 2, 'e' to 4, 'f' to 5, 'g' to 7)
        checkBorderKeys(
            conf.keyStartNum,
            conf.keyEndNum,
            conf.keyStart,
            conf.keyEnd
        )

        var whiteKeys = 0
        var xOffset = 0.0
        var width = 0.0
        // lower single keys
        for (key in (keyCodes[conf.keyStart] ?: 0)..(keyCodes.values.maxOrNull() ?: 0)) {
            val dim = conf.keyDimensions[key]
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
        for (scale in (conf.keyStartNum + 1) until conf.keyEndNum) {
            for (key in conf.keyDimensions.indices) {
                val dim = conf.keyDimensions[key]
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
        for (key in (keyCodes.values.minOrNull() ?: 0)..(keyCodes[conf.keyEnd] ?: 0)) {
            val dim = conf.keyDimensions[key]
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

    private fun tuneKeyBorders(borders: MutableList<Pair<Double, Double>>) {
        val width = borders.maxOf { it.second }
        borders.replaceAll { (left, right) ->
            val aLeft = distort(left, conf.keyboardDistOrigin, width, conf.keyboardDistCoef)
            val aRight =
                distort(right, conf.keyboardDistOrigin, width, conf.keyboardDistCoef)
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

    private fun shiftTimelineToStart(notes: MutableList<KeyEvent>) {
        if (notes.isNotEmpty()) {
            val offset = notes.minOf { it.pos }
            notes.replaceAll { KeyEvent(it.pos - offset, it.key, it.type, it.hand) }
        }
    }

    private fun createMidi(playSpeed: Double) {
        val sequencer = MidiSystem.getSequencer()
        sequencer.open()
        // Creating a sequence.
        // PPQ(Pulse per ticks) is used to specify timing, type and 4 is the timing resolution.
        val sequence = Sequence(Sequence.PPQ, conf.midiRes.toInt())
        // Setting our sequence so that the sequencer can run it on synthesizer
        sequencer.sequence = sequence

        val baseNote = 21 // the code of the lowest note on the piano, the A2
        // Creating a track on our sequence upon which MIDI events would be placed
        val track = sequence.createTrack()
        val tempo = encodeTempo(bpm2Tempo(conf.bpm.toFloat()))
        val timeSignature = encodeTimeSignature(
            conf.timeSignatureDenominator,
            conf.timeSignatureNominator
        )
        val trackName = "Generated Track"
        track.add(makeMetaEvent(TRACK_NAME_MIDI_CODE, trackName.toByteArray(), trackName.length, 0))
        track.add(makeMetaEvent(SET_TEMPO_MIDI_CODE, tempo, tempo.size, 0))
        track.add(makeMetaEvent(TIME_SIGNATURE_MIDI_CODE, timeSignature, timeSignature.size, 0))

        // Add events
        val timelineSorted = notes.sortedBy { it.pos }
        timelineSorted.forEach { event ->
            track.add(
                makeEvent(
                    if (event.type) ShortMessage.NOTE_ON else ShortMessage.NOTE_OFF,
                    //event.hand,
                    event.key + baseNote,
                    event.pos.div(playSpeed / 6).roundToInt()
                )
            )
        }


        val file = File("./output/midi.mid")
        MidiSystem.write(sequence, 0, file)
        exitProcess(1)
    }

    private fun detectNotesInImage(
        keyBorders: List<Pair<Double, Double>>,
    ) {
        val img = Mat()
        appendedImage.copyTo(img)
        notes.clear()
        keyBorders.forEachIndexed { keyIndex, border ->
            val bonusPx = 3
            val sliceWidth = (border.second - border.first)
            val widthThreshold = sliceWidth * 0.55
            val borderStart =
                (border.first - bonusPx).roundToInt().coerceIn(0, img.width())
            val borderEnd = (border.second + bonusPx).roundToInt().coerceIn(0, img.width())
            val slice = img.submat(0, img.height(), borderStart, borderEnd)

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

        saveImage("./output/detectedNotes.bmp", img)
    }

    fun rasterizeNotes() {
        if (offsets.isEmpty()) {
            val loadedOffsets = FileInputStream("output/offsets.txt").use {
                it.readAllBytes().decodeToString().splitToSequence(", ")
                    .map { elem -> elem.toDouble() }
                    .toList()
            }
            offsets.addAll(loadedOffsets)
        }

        val img = loadImage("./output/detectedNotes.bmp")
        val bps = conf.bpm / 60.0
        val pxPerSec = median(offsets) * fps()
        println("pixel per second $pxPerSec")
        val pxPerBeat = pxPerSec / bps

        val offset = 1755
        println("px: ${pxPerBeat.roundToInt()}")
        val max = img.height().minus(offset).div(pxPerBeat)
        for (i in 0 until max.toInt()) {
            val tl = Point(
                0.0,
                -offset + img.height() - (i * pxPerBeat).coerceAtMost(img.height().toDouble())
            )
            val br = Point(
                img.width().toDouble(),
                -offset + img.height() - (i * pxPerBeat).coerceAtMost(img.height().toDouble())
            )
            println("tl $tl br $br")
            Imgproc.line(img, tl, br, Scalar(0.0, 255.0, 0.0), 1)
        }
        saveImage("output/beatImage.bmp", img)
    }

    private fun getHand(colors: DoubleArray): Int {
        // (blue, green, red)
        return if (colors[0] > colors[1]) { // more blue than green
            1
        } else {
            0
        }
    }

    private fun makeEvent(
        command: Int,
        note: Int,
        tick: Int
    ): MidiEvent {
        val a = ShortMessage()
        a.setMessage(command, 0, note, 100)
        return MidiEvent(a, tick.toLong())
    }

    /**
     * source: https://stackoverflow.com/a/58476094
     */
    private fun makeMetaEvent(
        type: Int,
        data: ByteArray,
        length: Int,
        instant: Long
    ): MidiEvent {
        val metaMessage = MetaMessage()
        try {
            metaMessage.setMessage(type, data, length)
        } catch (e: InvalidMidiDataException) {
        }
        return MidiEvent(metaMessage, instant)
    }

    private fun bpm2Tempo(bpm: Float): Int {
        return (60_000_000 / bpm).toInt()
    }

    private fun encodeTempo(tempo: Int): ByteArray {
        val buffer = ByteBuffer.allocate(4)
        buffer.order(ByteOrder.BIG_ENDIAN)
        val bytes = buffer.putInt(tempo).array()
        return Arrays.copyOfRange(bytes, 1, 4)
    }

    private fun encodeTimeSignature(nominator: Int, denominator: Int): ByteArray {
        if (nominator > 255 || denominator <= 0)
            throw java.lang.IllegalArgumentException("nominator must be in range [1,255].")
        if (denominator % 2 != 0)
            throw java.lang.IllegalArgumentException("denominator must be a power of two.")
        if (denominator > 255 || denominator <= 0)
            throw java.lang.IllegalArgumentException("denominator must be in range [1,255].")
        val denominatorPower = (31 - Integer.numberOfLeadingZeros(denominator)).toByte()
        val buffer = ByteBuffer.allocate(4)
        buffer.order(ByteOrder.BIG_ENDIAN)
        // numerator, denominator pow: 'n' -> Pow(2, n) = d, MIDI Clocks :'18' = 24, number 1/32 notes per 24 MIDI clocks:'16' = 16'
        buffer.put(nominator.toByte()).put(denominatorPower).put(24.toByte()).put(22.toByte())
        return buffer.array()
    }

    private data class KeyEvent(val pos: Double, val key: Int, val type: Boolean, val hand: Int)

    companion object {
        private const val TRACK_NAME_MIDI_CODE = 0x03
        private const val SET_TEMPO_MIDI_CODE = 0x51
        private const val TIME_SIGNATURE_MIDI_CODE = 0x58
    }
}