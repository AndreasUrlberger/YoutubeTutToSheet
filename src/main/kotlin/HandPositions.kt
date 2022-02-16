import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.decodeFromStream
import java.io.FileInputStream
import java.io.FileNotFoundException
import java.nio.file.Files
import kotlin.io.path.Path


@Serializable
data class HandPositions(
    val frames: MutableList<MutableList<Hand>>, val pxOffset: MutableList<Int> = mutableListOf()
) {
    fun getFrame(index: Int) = HandFrame(frames[index])
    fun size() = frames.size
}

@Serializable
data class HandFrame(val hands: MutableList<Hand>) {
    fun getHand(index: Int) = hands[index]
    fun size() = hands.size
}

// estimated tell whether this hand's positions were given or constructed / estimated later on
@Serializable
data class Hand(
    val index: Int,
    val right: Boolean,
    val landmarks: MutableList<HandKeypoint>,
    val estimated: Boolean = false
) {
    fun getKeypoint(index: Int) = landmarks[index]
    fun size() = landmarks.size
}

@Serializable
data class HandKeypoint(val x: Float, val y: Float)


@OptIn(ExperimentalSerializationApi::class)
fun loadHandPositions(location: String): HandPositions {
    if (Files.notExists(Path(location)) || Files.isDirectory(Path(location)))
        throw FileNotFoundException("Could not find hand position file at \"$location\"")

    val handPos = Json.decodeFromStream<HandPositions>(FileInputStream("input/handPos.txt"))
    fillHoles(handPos)
    return handPos
}

fun fillHoles(positions: HandPositions, hands: Int = 2) {
    val handsToAdd = findMissingHands(positions, hands)
    // Add all missing hands
    handsToAdd.forEach { (frameIndex, hand) ->
        positions.frames[frameIndex].add(hand)
    }
}

private fun findMissingHands(
    positions: HandPositions,
    hands: Int
): List<Pair<Int, Hand>> {
    // Temporarily stores the missing frames as we cannot iterate over and edit the HandPositions at
    // the same time
    val handsToAdd = mutableListOf<Pair<Int, Hand>>()
    // The last known keyframe of a hand
    val lastHands = Array<Hand?>(hands) { null }
    // For how long a hand is missing
    val missingFrames = IntArray(hands)

    positions.frames.forEachIndexed { frameIndex, frame ->
        // Remembers which hands are missing in a frame
        val seenIndices = HashSet<Int>()
        // For all contained hands
        frame.forEach { hand ->
            val handIndex = hand.index % hands
            seenIndices.add(handIndex)
            val missedFrames = missingFrames[handIndex]
            if (missedFrames > 0) {
                lastHands[handIndex]?.let {
                    createMissingHands(it, hand, missedFrames, handIndex, frameIndex, handsToAdd)
                } ?: run {
                    createMissingHands(missedFrames, handIndex, hand, handsToAdd)
                }

                missingFrames[handIndex] = 0
            }
            lastHands[handIndex] = hand
        }

        // For all missing hands
        // update the amount of times a frame was missing consecutively
        for (index in 0 until hands) {
            if (!seenIndices.contains(index)) {
                missingFrames[index]++
            }
        }
    }

    return handsToAdd
}

private fun createMissingHands(
    missedFrames: Int,
    handIndex: Int,
    hand: Hand,
    handsToAdd: MutableList<Pair<Int, Hand>>
) {
    // First time this frame is seen -> artificially restore previous frames
    // Start from zero since there is no start frame
    for (step in 0 until missedFrames) {
        val missingHand = Hand(handIndex, hand.right, mutableListOf(), estimated = true)
        hand.landmarks.forEach {
            missingHand.landmarks.add(it)
        }
        handsToAdd.add(Pair(step, missingHand))
    }
}

private fun createMissingHands(
    startHand: Hand,
    endHand: Hand,
    missedFrames: Int,
    handIndex: Int,
    frameIndex: Int,
    handsToAdd: MutableList<Pair<Int, Hand>>
) {
    // Zip the start and end landmarks to make interpolating easier
    val combList = startHand.landmarks.zip(endHand.landmarks)
    val startFrameIndex = frameIndex - missedFrames
    // Linearly interpolate between the landmarks between the landmarks of the start
    // and end frame
    for (step in 0 until missedFrames) {
        val missingHand =
            Hand(handIndex, endHand.right, mutableListOf(), estimated = true)
        combList.forEach { (start, end) ->
            val differenceX = end.x - start.x
            val differenceY = end.y - start.y
            missingHand.landmarks.add(
                HandKeypoint(
                    x = start.x + (differenceX / (missedFrames + 1)) * (step + 1),
                    y = start.y + (differenceY / (missedFrames + 1)) * (step + 1)
                )
            )
        }
        handsToAdd.add(Pair(startFrameIndex + step, missingHand))
    }
}


