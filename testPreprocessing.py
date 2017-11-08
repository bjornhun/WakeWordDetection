import unittest
import random
import preprocessing
from scipy.io import wavfile

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.seg_length = 14400
        fs, x = wavfile.read("data/train/f029_04_067.wav")        
        self.seg = x[:14400]
        self.short_seg = x[:10000]
        self.long_seg = x


    def test_set_length(self):
        short_seg = preprocessing.set_length(self.short_seg)
        long_seg = preprocessing.set_length(self.long_seg)

        self.assertEqual(len(short_seg), self.seg_length)
        self.assertEqual(len(long_seg), self.seg_length)


    def test_get_coefficients(self):
        frames = preprocessing.get_coefficients(self.seg, 0)
        frames_short = preprocessing.get_coefficients(self.short_seg, 0)
        frames_long = preprocessing.get_coefficients(self.long_seg, 0)
        
        self.assertEqual(frames[0][0].shape[1], 13)
        self.assertEqual(frames_short[0][0].shape[1], 13)
        self.assertEqual(frames_long[0][0].shape[1], 13)
        self.assertEqual(frames[0][1], 0)
        self.assertEqual(frames_short[0][1], 0)
        self.assertEqual(frames_long[0][1], 0)


if __name__ == '__main__':
    unittest.main()