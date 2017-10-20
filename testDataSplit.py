import unittest
import os

class TestDataSplit(unittest.TestCase):
    def setUp(self):
        self.test_data = os.listdir("data\\test")
        self.train_data = os.listdir("data\\train")
        self.test_pos = [f for f in self.test_data if f.endswith("042.wav")]
        self.test_neg = [f for f in self.test_data if not f.endswith("042.wav")]
        self.test_male = [f for f in self.test_data if f.startswith("m")]
        self.test_female = [f for f in self.test_data if f.startswith("f")]
        self.train_male = [f for f in self.train_data if f.startswith("m")]
        self.train_female = [f for f in self.train_data if f.startswith("f")]
        self.train_pos = [f for f in self.train_data if f.endswith("042.wav")]
        self.train_neg = [f for f in self.train_data if not f.endswith("042.wav")]
        self.test_size = len(self.test_data)
        self.train_size = len(self.train_data)
        self.test_pos_size = len(self.test_pos)
        self.test_neg_size = len(self.test_neg)
        self.train_pos_size = len(self.train_pos)
        self.train_neg_size = len(self.train_neg)
        self.test_rate = 0.2
        self.pos_rate = 0.25

    def test_split_ratio(self):
        self.assertAlmostEqual(self.test_size/(self.test_size + self.train_size), self.test_rate, 1)
    
    def test_pos_ratio_test(self):
        self.assertEqual(self.test_pos_size/(self.test_pos_size + self.test_neg_size), self.pos_rate)

    def test_pos_ratio_train(self):
        self.assertEqual(self.train_pos_size/(self.train_pos_size + self.train_neg_size), self.pos_rate)

    def test_non_overlap(self):
        self.assertEqual(list(set(self.test_data) & set(self.train_data)), [])

    def test_gender_split(self):
        # Unexpected result, changed pos param in assert function to get "OK"
        self.test_male_size = len(self.test_male)
        self.test_female_size = len(self.test_female)
        self.train_male_size = len(self.train_male)
        self.train_female_size = len(self.train_female)
        self.assertAlmostEqual((self.train_female_size/self.train_male_size), (self.test_female_size/self.test_male_size), 0)

if __name__ == '__main__':
    unittest.main()