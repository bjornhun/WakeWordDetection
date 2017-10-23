import unittest
import random
import MFCC

class TestMFCC(unittest.TestCase):
    def setUp(self):
        self.seg_length = 16000 * 5
        self.seg = random.choices(range(-100, 100), k=self.seg_length)

    def test_uniformize(self):
        short_seg = random.choices(range(-100, 100), k=(self.seg_length - random.randint(0, self.seg_length)))
        long_seg = random.choices(range(-100, 100), k=(self.seg_length + random.randint(0, self.seg_length)))
        short_seg_uni = MFCC.uniformize(short_seg)
        long_seg_uni = MFCC.uniformize(long_seg)

        self.assertEqual(len(short_seg_uni), self.seg_length)
        self.assertEqual(len(long_seg_uni), self.seg_length)
        self.assertEqual(len(long_seg_uni), len(short_seg_uni))

    def test_frame_partition(self):
        frames = MFCC.frame_partition(self.seg)
        self.assertEqual(frames.shape, (498, 400))
        self.assertListEqual(list(frames[0]), list(self.seg[:400]))

if __name__ == '__main__':
    unittest.main()