import javax.sound.midi.MidiEvent
import javax.sound.midi.MidiSystem
import javax.sound.midi.Sequence
import javax.sound.midi.ShortMessage
import kotlin.system.exitProcess

fun main() {
    val numOfNotes = 20
    val sequencer = MidiSystem.getSequencer()
    sequencer.open()

    // Creating a sequence.
    val sequence = Sequence(Sequence.PPQ, 4)

    // PPQ(Pulse per ticks) is used to specify timing
    // type and 4 is the timing resolution.

    // Creating a track on our sequence upon which
    // MIDI events would be placed
    val track = sequence.createTrack()

    // Adding some events to the track
    var i = 5
    while (i < 4 * numOfNotes + 5) {
        // Add Note On event
        track.add(makeEvent(144, 1, i, 100, i))

        // Add Note Off event
        track.add(makeEvent(128, 1, i, 100, i + 2))
        i += 4
    }

    // Setting our sequence so that the sequencer can
    // run it on synthesizer
    sequencer.sequence = sequence

    // Specifies the beat rate in beats per minute.
    sequencer.tempoInBPM = 220.toFloat()

    // Sequencer starts to play notes
    sequencer.start()

    while (true) {

        // Exit the program when sequencer has stopped playing.
        if (!sequencer.isRunning) {
            sequencer.close()
            exitProcess(1)
        }
    }
}

fun makeEvent(command: Int, channel: Int, note: Int, velocity: Int, tick: Int): MidiEvent {

    // ShortMessage stores a note as command type, channel,
    // instrument it has to be played on and its speed.
    val a = ShortMessage()
    a.setMessage(command, channel, note, velocity);

    // A midi event is comprised of a short message(representing
    // a note) and the tick at which that note has to be played

    return MidiEvent(a, tick.toLong())
}

