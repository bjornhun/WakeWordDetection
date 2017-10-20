import unittest
import random
import MFCC

class TestMFCC(unittest.TestCase):
    def setUp(self):
        self.seg_length = 16000 * 5
        self.short_seg = random.choices(range(-100, 100), k=(self.seg_length - random.randint(0, self.seg_length)))
        self.long_seg = random.choices(range(-100, 100), k=(self.seg_length + random.randint(0, self.seg_length)))
        self.short_seg_uni = MFCC.uniformize(self.short_seg)
        self.long_seg_uni = MFCC.uniformize(self.long_seg)

    def test_uniformize(self):
        self.assertEqual(len(self.short_seg_uni), self.seg_length)
        self.assertEqual(len(self.long_seg_uni), self.seg_length)
        self.assertEqual(len(self.long_seg_uni), len(self.short_seg_uni))

    def test_frame_partition(self):
        pass

if __name__ == '__main__':
    unittest.main()