import saveHelper

from Conv02 import *
import tensorflow as tf
import os
from PIL import Image

maxsize = 10000000 #define max points in workspace
#then define our helperobject with image dimensions and classes
saveObj = saveHelper.saverObject(class_names=["middle","left","right"],
                                 img_dimensions=[30, 20, 1],
                                 from_folder="/home/gsteelman/Desktop/ML/Pictures/" ,
                                 to_folder="/home/gsteelman/Desktop/ML/AllData/LRtracker/" ,
                                  maxSize=maxsize)
# saveObj.resize(subfolders=["middle/","left/","right/"])
# saveObj.togray(subfolders=["middle/","left/","right/"],)
#
# saveObj.cache_pictures(subfolders=["left/","middle/","right/"])
# saveObj.cache_test_pictures(subfolders=["left/","middle/","right/"], path = "/home/gsteelman/Desktop/ML/Pictures/test/")
#add new model
myModel = CNNModel(saveObj,80)
filter_size1 = [5,5]          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = [5,5]           # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128
#Define our convolutional layers
x_pretty = pt.wrap(myModel.x_image)
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=16, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, name='layer_conv2').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name='layer_fc1').\
        softmax_classifier(num_classes=myModel.num_classes, labels=myModel.y_true)
#
# #Start session and init values
myModel.set_optimizer(y_pred, loss = loss)#set optimizer
session = tf.Session()
session.run(tf.global_variables_initializer())

# saver = tf.train.Saver()
# save_dir = 'checkpoints/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# save_path = os.path.join(save_dir, 'best_validation')
# saver.restore(sess=session, save_path=save_path)
#

#optimize the model for a set iterations
# myModel.optimize(num_iterations=10000,saveHelper=saveObj, session = session,batch_size = 64,numTest = 8)
# my_gen = saveObj.random_batch(1)
# for i in range(100):
#     next(my_gen,(None,None,None))
# data,dclass,_ = next(my_gen,(None,None,None))
# data = np.squeeze(data)
# im = Image.fromarray(data)
# print(dclass)
# im.show()
#
# myModel.print_test_accuracy(saveHelper=saveObj, session = session)
# #
# saver = tf.train.Saver()
# save_dir = 'checkpoints/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# save_path = os.path.join(save_dir, 'best_validation')
# saver.save(sess=session, save_path=save_path)
#     #
