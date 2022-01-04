import org.opencv.core.Core
import org.opencv.core.Mat

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

// Sheet Music Boss
private val imageRange = Pair(0.25, 0.5)
private val prepareImage: (Mat) -> Mat = { img ->
    val thresholds = doubleArrayOf(80.0, 85.0, 80.0)
    val channels = mutableListOf<Mat>()
    Core.split(img, channels)
    computeAddedThresh(channels, thresholds)
}
private val extractNotes: (Mat) -> Mat = { img ->
    val thresholds = doubleArrayOf(80.0, 85.0, 80.0)
    val slices = mutableListOf<Mat>()
    Core.split(img, slices)
    computeAddedThresh(slices, thresholds)
}
private const val maxImageDiff = 0.25
private const val keyboardDistOrigin = 0.47
private const val keyboardDistCoef = 0.043


class SMBSettings(bpm: Double, midiRes: Double, framesToSkip: Int) : VidSettings(
    bpm = bpm,
    midiRes = midiRes,
    framesToSkip = framesToSkip,

    keyStart = 'a',
    keyStartNum = -3,
    keyEnd = 'c',
    keyEndNum = 5,

    prepareImage = prepareImage,
    extractNotes = extractNotes,
    maxImageDiff = maxImageDiff,
    keyboardDistCoef = keyboardDistCoef,
    keyboardDistOrigin = keyboardDistOrigin,
    keyDimensions = realKeyDimensions,
    frameLimits = imageRange
)