import kotlin.math.abs

object Utils {
    @JvmStatic
    fun binaryGreaterThanSearch(
        list: List<Double>,
        fromIndex: Int = 0,
        toIndex: Int = list.size,
        minValue: Double
    ): Int {
        var low = fromIndex
        var high = toIndex - 1

        while (low < high) {
            val mid = (low + high).ushr(1) // safe from overflows
            val midVal = list[mid]

            if (midVal > minValue) {
                high = mid
            } else if (midVal < minValue) {
                low = mid + 1
            } else {
                low = mid
                high = mid
            }
        }

        // low and high should end up on the same value
        return if (list[low] >= minValue) {
            low
        } else {
            -1
        }
    }

    @JvmStatic
    fun likeliestHand(xPosNote: Double, hands: List<Hand>): Hand {
        if (hands.isEmpty()) {
            throw IllegalArgumentException("hands list cannot be empty")
        }
        if (xPosNote < 0 || xPosNote >= 1) {
            throw IllegalArgumentException("The note location must be in range [0;1[")
        }

        var likeliestHand: Hand = hands.first()
        var minDistance = Double.POSITIVE_INFINITY
        for (hand in hands) {
            for (keypoint in getFingerTips(hand)) {
                val distance = (abs(keypoint.x - xPosNote))
                if (distance < minDistance) {
                    minDistance = distance
                    likeliestHand = hand
                }
            }
        }

        return likeliestHand
    }
}