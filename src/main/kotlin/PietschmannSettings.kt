import org.opencv.core.Mat

private val imageRange = Pair(0.0, 0.5)
private val prepareImage: (Mat) -> Mat = { img ->
    val thresh = extractPatrickNotes(img)
    val start = (thresh.height() * 0.25).toInt()
    val end = (thresh.height() * 0.75).toInt()
    thresh.submat(start, end, 0, thresh.width())
}
private val extractNotes: (Mat) -> Mat = ::extractPatrickNotes
private const val maxImageDiff = 0.50
private const val keyboardDistOrigin = 0.47
private const val keyboardDistCoef = 0.043

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


class PietschmannSettings(
    bpm: Double = 120.0,
    midiRes: Double = 480.0,
    framesToSkip: Int = 2
) : VidSettings(
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
    keyDimensions = simpleKeyDimensions,
    frameLimits = imageRange
)
