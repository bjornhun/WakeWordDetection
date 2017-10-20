import os

rsr_path = "C:/Users/Bjornar/Documents/Workspace/WakeWordDetection/rsr2015/"
os.chdir(rsr_path)

for folder in ["male/", "female/"]:
    os.chdir(rsr_path + folder)
    print("Entered folder " + folder)
    subfolders = os.listdir()
    for subfolder in subfolders:
        os.chdir(rsr_path + folder + subfolder)
        files = [f for f in os.listdir() if f.endswith(".sph")]
        for f in files:
            f_out = f.replace(".sph", ".wav")
            os.system("sox " + f + " " + f_out)
            os.system("DEL " + f)
        print("Subfolder " + folder + subfolder + " converted")
