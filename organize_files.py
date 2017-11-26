import os
from random import sample

test_rate = .2

rsr_path = os.getcwd() + "\\rsr2015\\"
data_path = os.getcwd() + "\\data\\"

for folder in ["male\\", "female\\"]:
    os.chdir(rsr_path + folder)
    subfolders = os.listdir()

    num_speakers = len(subfolders)
    num_test = test_rate * num_speakers

    for subfolder in subfolders:
        
        if (int(subfolder[-3:]) <= num_test):
            target = data_path + "test\\"
        else:
            target = data_path + "train\\"

        os.chdir(rsr_path + folder + subfolder)
        files = os.listdir()
        #pos = [f for f in files if f.endswith("042.wav")]
        #neg = sample([f for f in files if f not in pos], len(pos)*3)

        #for f in pos:
        #    command = ("copy " +  f + " " + target + f)
        #    os.system(command)

        #for f in neg:
        #    command = ("copy " +  f + " " + target + f)
        #    os.system(command)

        neg_close = [f for f in files if f.endswith("043.wav")]
        neg_hold = [f for f in files if f.endswith("044.wav")]

        for f in neg_close:
            command = ("copy " +  f + " " + target + f)
            os.system(command)

        for f in neg_hold:
            command = ("copy " +  f + " " + target + f)
            os.system(command)

        print("Subfolder " + folder + subfolder + " sampled")