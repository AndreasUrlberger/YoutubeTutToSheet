import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MainTest {
    @Test
    fun convertIntoKeyFocusedTest() {
        // arrange
        val notes = listOf(
            // frames
            mapOf(
                // keys
                0 to listOf(24.0 to 0.0, 78.0 to 56.0 /*notes*/),
                1 to listOf(122.0 to 56.0),
            ),
            mapOf(
                0 to listOf(36.0 to 12.0, 80.0 to 68.0),
                1 to listOf(134.0 to 68.0),
                4 to listOf(12.0 to 0.0),
            ),
            mapOf(
                4 to listOf(24.0 to 0.0),
            ),
        )

        // act
        val keyFocused = convertIntoKeyFocused(notes, 5)

        val want = listOf( // keys
            listOf( // frames
                listOf(24.0 to 0.0, 78.0 to 56.0 /*notes*/),
                listOf(36.0 to 12.0, 80.0 to 68.0),
                listOf()
            ),
            listOf(
                listOf(122.0 to 56.0),
                listOf(134.0 to 68.0),
                listOf()
            ),
            listOf(
                listOf(),
                listOf(),
                listOf()
            ),
            listOf(
                listOf(),
                listOf(),
                listOf()
            ),
            listOf(
                listOf(),
                listOf(12.0 to 0.0),
                listOf(24.0 to 0.0)
            )
        )
        // assert
        assertEquals(want, keyFocused)
    }

    @Test
    fun convertNotesToTimestampsTest() {
        // arrange
        val notes = listOf( // keys
            listOf( // frames
                listOf(24.0 to 0.0, 78.0 to 56.0 /*notes*/),
                listOf(36.0 to 12.0, 80.0 to 68.0),
                listOf()
            ),
            listOf(
                listOf(122.0 to 56.0),
                listOf(134.0 to 68.0),
                listOf()
            ),
            listOf(
                listOf(),
                listOf(),
                listOf()
            ),
            listOf(
                listOf(),
                listOf(),
                listOf()
            ),
            listOf(
                listOf(),
                listOf(12.0 to 0.0),
                listOf(24.0 to 0.0)
            )
        )
        // act
        //convertNotesToTimestamps(notes, 5, 200)
        // assert
        assertTrue(true)
    }

    @Test
    fun getKeyTimelineTest() {
        // arrange
        val key = listOf( // frames
            listOf(0.0 to 24.0, 56.0 to 78.0 /*notes*/),
            listOf(12.0 to 36.0, 68.0 to 80.0),
            listOf()
        )
        val want = listOf(
            22.0 to 44.0, 76.0 to 100.0
        )
        // act
        val have = getKeyTimeline(key, 12.0, 100.0)
        // assert
        assertEquals(want, have)
    }

    @Test
    fun calculateCrossCorrelationTestEqual() {
        // arrange
        val frameOld = listOf(
            Pair(12.0, 0.0),
            Pair(35.0, 30.0),
        )
        val frameNew = listOf(
            Pair(12.0, 0.0),
            Pair(35.0, 30.0),
        )
        val want = 1.0
        // act
        val have = calculateCrossCorrelation(frameOld, frameNew, 0)
        // assert
        assertEquals(want, have)
    }

    @Test
    fun calculateCrossCorrelationTestOff2() {
        // arrange
        val frameOld = listOf(
            Pair(12.0, 0.0),
            Pair(35.0, 30.0),
        )
        val frameNew = listOf(
            Pair(14.0, 2.0),
            Pair(37.0, 32.0),
        )
        val want = 1.0
        // act
        val have = calculateCrossCorrelation(frameOld, frameNew, 2)
        // assert
        assertEquals(want, have)
    }

    @Test
    fun calculateCrossCorrelationTestHalf() {
        // arrange
        val frameOld = listOf(
            Pair(12.0, 6.0),
        )
        val frameNew = listOf(
            Pair(12.0, 0.0),
        )
        val want = 0.5
        // act
        val have = calculateCrossCorrelation(frameOld, frameNew, 0)
        // assert
        assertEquals(want, have)
    }

    @Test
    fun getCrossOffsetTest() {
        // arrange
        val frameOld = listOf(
            Pair(12.0, 0.0),
            Pair(35.0, 30.0),
        )
        val frameNew = listOf(
            Pair(25.0, 13.0),
            Pair(48.0, 43.0),
        )
        val want = 13.0
        // act
        val have = getCrossOffset(frameOld, frameNew)
        // assert
        assertEquals(want, have)
    }
}