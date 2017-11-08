import unittest
import random
import preprocessing

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.seg_length = 14400
        self.seg = random.choices(range(-100, 100), k=self.seg_length)
        self.short_seg = random.choices(range(-100, 100), k=(self.seg_length - random.randint(0, self.seg_length)))
        self.long_seg = random.choices(range(-100, 100), k=(self.seg_length + random.randint(0, self.seg_length)))


    def test_set_length(self):

        short_seg_uni = preprocessing.set_length(self.short_seg)
        long_seg_uni = preprocessing.set_length(self.long_seg)

        self.assertEqual(len(short_seg_uni), self.seg_length)
        self.assertEqual(len(long_seg_uni), self.seg_length)
        self.assertEqual(len(long_seg_uni), len(short_seg_uni))


    def test_get_coefficients(self):
        frames = preprocessing.get_coefficients(self.seg)
        frames_short = preprocessing.get_coefficients(self.short_seg)
        frames_long = preprocessing.get_coefficients(self.long_seg)
        
        self.assertEqual(frames.shape[1], 14400)
        self.assertEqual(frames_short.shape[1], 14400)
        self.assertEqual(frames_long.shape[1], 14400)
        self.assertEqual(frames.shape[0], 1)
        self.assertEqual(frames_short.shape[0], 1)
        self.assertEqual(frames_long.shape[0], 1 + (len(self.long_seg) - 14400) // 1600)
        self.assertListEqual(list(frames[0]), list(self.seg[:14400]))
        self.assertListEqual(list(frames_long[0]), list(self.long_seg[:14400]))


if __name__ == '__main__':
    unittest.main()