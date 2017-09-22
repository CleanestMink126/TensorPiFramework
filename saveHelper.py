########################################################################
#
#
# Implemented in Python 3.6
#
#
########################################################################
#
# This file originates from  part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import numpy as np
import pickle
import os
from dataset import one_hot_encoded
import random
from scipy import misc
from PIL import Image
import matplotlib.pyplot as plt


########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
class dataObject:
    '''This is a simple data containter so the data can be pickled with its values'''
    def __init__(self,traindata, classes):
        self.traindata = traindata
        self.classes = classes

class saverObject:
    '''This object handles all the saving and caching done in the model'''
    def __init__(self,maxSize, to_folder, from_folder,img_dimensions,class_names):
        self.batches_folder = to_folder#Where to cache data
        self.origin_folder = from_folder#where to read pictures from
        self.maxSize = maxSize#maximum amount of data in workspace at a time
        self.img_width = img_dimensions[0]
        self.img_height = img_dimensions[1]
        self.channels = img_dimensions[2]
        self.numImages = int(self.maxSize/(self.img_width*self.img_height + 1))#calculates the number of images so that max size applied
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.num_batches = 0
    def _get_file_path(self,filename=""):
        """
        Return the full path of a data-file for the data-set.

        If filename=="" then return the directory of the files.
        """

        return os.path.join(self.batches_folder, filename)


    def _unpickle(self,filename):
        """
        Unpickle the given file and return the data.

        Note that the appropriate dir-name is prepended the filename.
        """

        # Create full path for the file.
        file_path = self._get_file_path(filename)

        # print("Loading data: " + file_path)

        with open(file_path, mode='rb') as f:
            # In Python 3.X it is important to set the encoding,
            # otherwise an exception is raised here.
            data = pickle.load(f, encoding='bytes')

        return data


    def _convert_images(self,raw):
        """
        Convert images from the CIFAR-10 format and
        return a 4-dim array with shape: [image_number, height, width, channel]
        where the pixels are floats between 0.0 and 1.0.
        """
        # Convert the raw images from the data-files to floating-points.
        raw_float = np.array(raw)
        # print(len(raw_float[0]))

        # Reshape the array to 4-dimensions.
        images = raw_float.reshape([-1,self.img_width,self.img_height,self.channels])

        # Reorder the indices of the array.
        # images = images.transpose([3, 2, 4, 1])


        return images


    def _load_data(self,filename):
        """
        Load a pickled data-file from the data-set
        and return the converted images (see above) and the class-number
        for each image.
        """

        # Load the pickled data-file.
        data = self._unpickle(filename)
        # print(len(data.traindata))
         # raw_images = np.array(data.traindata)
        images = self._convert_images(data.traindata)

        return images, data.classes, one_hot_encoded(class_numbers=data.classes, num_classes=self.num_classes)

    def _save_data(self,data,classes, filename):

        file_path = self._get_file_path(filename) + ".pkl"
        if len(data) == len(classes):
            saveObject = dataObject(traindata=data,classes=classes)
        else:
            print("NOT SAME SIZE")
            return
        print("Saving data: " + file_path)

        with open(file_path, mode='wb') as f:
            # In Python 3.X it is important to set the encoding,
            # otherwise an exception is raised here.
            data = pickle.dump(saveObject,f)

        return data

    def resize(self, subfolders,targetSize = [30,20], path = None,offset = 0):
        '''This will resize the images in the desired subfolders to the targetSize'''
        if path is None:
            path = self.origin_folder


        for i, v in enumerate(subfolders):
            files_txt = [path+v + f for f in os.listdir(path+v) if f.endswith('.jpg')]
            for f in files_txt:
                im = Image.open(f)
                print(im.size)
                im = im.resize(targetSize,Image.ANTIALIAS)
                im.save(f,optimize=True,quality=100)




    def cache_pictures(self, subfolders, path = None,offset = 0):
        '''This function will take all the subfolders, iterate through the
        pictures in them, randomize the pictures while keeping track of the
        subfolders, and cache the pictures in appropriatly sized pickle
        folders in the ToFolder'''
        if path is None:
            path = self.origin_folder
        listPaths = []

        for i, v in enumerate(subfolders):
            files_txt = [[path+v + f, v]for f in os.listdir(path+v) if f.endswith('.jpg')]
            listPaths = listPaths + files_txt

        random.shuffle(listPaths)
        print(listPaths)
        # imageObj= np.zeros(shape = [self.numImages,self.img_width,self.img_height,self.channels])
        # clsObj = np.zeros(shape = [self.numImages])
        imageObj = []
        clsObj = []
        for i,v in enumerate(listPaths):
            rgb = misc.imread(v[0])
            # rgb = np.array(im.getdata()).reshape((self.img_width, self.img_height, self.channels))
            # print(rgb)
            # imageObj[i % self.numImages] = rgb
            # clsObj[i % self.numImages] = subfolders.index(v[1])
            imageObj.append(rgb)
            clsObj.append(subfolders.index(v[1]))


            if i % self.numImages == self.numImages-1:
                identifier = "data_batch_" + str(int(i/self.numImages)+ offset)
                # print(imageObj[1])
                self._save_data(data=imageObj,classes=clsObj,filename= identifier)
                # imageObj= np.zeros(shape = [self.numImages,self.img_width,self.img_height,self.channels])
                # clsObj = np.zeros(shape = [self.numImages])
                imageObj = []
                clsObj = []

        identifier = "data_batch_" + str(int(i/self.numImages + offset))
        self.num_batches = int(len(listPaths)/self.numImages + offset)
        print(imageObj[1])
        print(self.num_batches)
        self._save_data(data=imageObj,classes=clsObj,filename= identifier)
        imageObj= []
        clsObj = []


    def random_in_batch(self,batch_size):
        '''I'm gonna wait to implement this but it will return random values
        inside a batch'''
        return
    def random_batch(self,batch_size):
        '''This will continually return batches of the desired batch size
        from a random cache. Does not work as well if there are minimal chaches'''
        data,classes,one_hot_classes = self._load_data("data_batch_" + str(random.randint(0,self.num_batches))+ ".pkl")
        # print("why god")
        # print(classes)
        for i in range(0,len(classes),batch_size):
            # print("running")
            j = min(i+batch_size,len(classes))
            print(i)
            yield data[i:j],classes[i:j],one_hot_classes[i:j]
    def test_batch(self,batch_size,testName):
        '''Return batches from the desired cache name'''
        data,classes,one_hot_classes = self._load_data("data_batch_" + testName+ ".pkl")
        for i in range(0,self.numImages,batch_size):
            j = min(i+batch_size,len(classes))
            # print(data)
            yield data[i:j],classes[i:j],one_hot_classes[i:j]







########################################################################
