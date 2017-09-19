import inception
from inception import transfer_values_cache
import cifar10,os
import matplotlib.pyplot as plt
import time
import rawpy
import os
from scipy import misc


data_path = "data/cheese-10/"

# URL for the data-set on the internet.
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

########################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.


# Number of channels in each image, 3 channels: Red, Green, Blue.

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_width * img_height * num_channels

# Number of classes.


########################################################################
# Various constants used to allocate arrays of the correct size.


# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.

########################################################################
# Private functions for downloading, unpacking and loading data-files.

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
