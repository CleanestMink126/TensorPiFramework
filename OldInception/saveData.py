import inception
from inception import transfer_values_cache
import cifar10,os
import matplotlib.pyplot as plt
import time
import rawpy
import os
from scipy import misc

path = "/home/gsteelman/Desktop/Machine Learning/TestingInception/Pictures/"
mydir = [path + "middle/",path + "left/",path + "right/"]
imageObj= []
clsObj = []
k = 0
for i, v in enumerate(mydir):
    files_txt = [f for f in os.listdir(v) if f.endswith('.jpg')]
    for j,infile in enumerate(files_txt):
        rgb = misc.imread(v + infile)
        #print(rgb)
        imageObj.append(rgb)
        clsObj.append(i)
        if j %10  == 9:



            folder = "data/cheese-10/batches-py/"

            identifier = "data_batch_" + str(k)
            k += 1
            number = ".pkl"


            cifar10._save_data(imageObj,identifier + "_img" + number)
            cifar10._save_data(clsObj,identifier + "_cls" + number)
            imageObj= []
            clsObj = []
            j = 0

images,classes,onehot = cifar10.load_training_data()
