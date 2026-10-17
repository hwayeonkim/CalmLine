import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from calmline.sample_cache import SampleCache, _Local, _h


class VoiceCacheRevisionTests(unittest.TestCase):
    def test_old_audio_is_bypassed_without_losing_stt_or_cards(self):
        with tempfile.TemporaryDirectory() as folder:
            wav = Path(folder) / 'sample.wav'
            wav.write_bytes(b'fixture')
            with patch('calmline.sample_cache.store', return_value=_Local(str(Path(folder) / 'cache'))):
                cache = SampleCache('heat', str(wav), 'v20', 'google')
            text = "This is the fourth time. I've called."
            cache.st.put(f'{cache.ns}/voice/{_h(text)}.pcm', b'\0\0' * 20)
            cache.put_stt([{'text': text}])
            cache.put_card(0, text, {'delivered': text})
            self.assertIsNone(cache.get_voice(text))
            self.assertEqual(cache.get_stt()['events'][0]['text'], text)
            self.assertEqual(cache.get_card(0, text)['card']['delivered'], text)
            cache.put_voice(text, np.ones(20, dtype=np.float32) * .25)
            np.testing.assert_allclose(cache.get_voice(text), .25, atol=1/32767)


if __name__ == '__main__':
    unittest.main()
