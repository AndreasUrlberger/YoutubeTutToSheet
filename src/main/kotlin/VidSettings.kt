import org.opencv.core.Mat

enum class NoteHandCoupling {
    None,
    NoteColor,
    HandPosDetection
}

open class VidSettings(
    val bpm: Double = 120.0,
    val midiRes: Double = 480.0,
    val framesToSkip: Int = 2,
    val timeSignatureNominator: Int,
    val timeSignatureDenominator: Int,

    val noteHandCoupling: NoteHandCoupling = NoteHandCoupling.None,

    val keyStart: Char = 'a',
    val keyStartNum: Int = -3,
    val keyEnd: Char = 'c',
    val keyEndNum: Int = 5,

    val prepareImage: (Mat) -> Mat,
    val extractNotes: (Mat) -> Mat,
    val maxImageDiff: Double,
    val keyboardDistCoef: Double,
    val keyboardDistOrigin: Double,
    val keyDimensions: List<Triple<Double, Double, Boolean>>,
    val frameLimits: Pair<Double, Double>
)
