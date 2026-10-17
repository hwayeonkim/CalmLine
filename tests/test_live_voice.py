import asyncio
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from calmline.live_voice import LiveReader, drifted, omitted, speech_parts


class CompletenessTests(unittest.TestCase):
    def test_reported_tail_omission(self):
        self.assertTrue(omitted("This is the fourth time. I've called.", "This is the fourth time."))

    def test_full_read_and_contractions(self):
        self.assertFalse(omitted("This is the fourth time. I've called.", "This is the fourth time. I have called."))
        self.assertFalse(omitted("We've had no heat.", "We have had no heat."))

    def test_middle_phrase_omission(self):
        self.assertTrue(omitted("No heat at Walton Avenue apartment 2C.", "No heat at apartment 2C."))

    def test_missing_transcript(self):
        self.assertTrue(omitted("I've called.", ""))

    def test_one_word_asr_variation(self):
        self.assertFalse(omitted("at 1820 Walton Avenue", "at eighteen twenty Walton Avenue"))

    def test_additions_are_still_detected(self):
        self.assertTrue(drifted("I've called.", "Your complaint has already been resolved, goodbye."))

    def test_actual_two_sentence_script(self):
        self.assertEqual(speech_parts("This is the fourth time. I've called."),
                         ["This is the fourth time.", "I've called."])

    def test_addresses_and_abbreviations(self):
        self.assertEqual(speech_parts("At 75 Hart. Street and 222 East 21st. Street Apt. 4B."),
                         ["At 75 Hart. Street and 222 East 21st. Street Apt. 4B."])


class ReaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_parts_stream_in_order_and_keep_transcript(self):
        class Reader(LiveReader):
            async def _speak_part(self, text):
                self.last_transcript = text
                yield text
        reader = Reader()
        with patch('calmline.live_voice.PLAIN', True):
            chunks = [x async for x in reader.speak("This is the fourth time. I've called.")]
        self.assertEqual(chunks, ["This is the fourth time.", "I've called."])
        self.assertEqual(reader.last_transcript, "This is the fourth time. I've called.")

    async def test_partial_model_transcript_flags_without_duplicate_playback(self):
        class Session:
            async def send_client_content(self, **kwargs):
                pass
            async def receive(self):
                yield SimpleNamespace(server_content=SimpleNamespace(
                    output_transcription=SimpleNamespace(text='This is the fourth time.'),
                    model_turn=SimpleNamespace(parts=[SimpleNamespace(inline_data=SimpleNamespace(data=b'\0' * 9600))]),
                    turn_complete=True))
        reader = LiveReader()
        reader.session = Session()
        with patch('calmline.live_voice.PLAIN', False), patch('librosa.resample', side_effect=lambda x, **kwargs: x):
            chunks = [x async for x in reader.speak("This is the fourth time. I've called.")]
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(reader.last_voice_warnings), 1)

    async def test_concurrent_scripts_do_not_interleave_parts(self):
        events = []
        class Reader(LiveReader):
            async def _speak_part(self, text):
                events.append(text)
                await asyncio.sleep(0)
                self.last_transcript = text
                yield text
        reader = Reader()
        async def consume(text):
            return [x async for x in reader.speak(text)]
        with patch('calmline.live_voice.PLAIN', True):
            await asyncio.gather(consume('First. Second.'), consume('Third. Fourth.'))
        self.assertEqual(events, ['First.', 'Second.', 'Third.', 'Fourth.'])


if __name__ == '__main__':
    unittest.main()
