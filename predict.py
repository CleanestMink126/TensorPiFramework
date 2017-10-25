import saveHelper

from Conv02 import *
import tensorflow as tf
import os
from PIL import Image
import v4l2capture
import select
import numpy as np
from io import StringIO


def init_model():

    maxsize = 10000000 #define max points in workspace
    #then define our helperobject with image dimensions and classes
    saveObj = saveHelper.saverObject(class_names=["middle","left","right"],
                                     img_dimensions=[30, 20, 1],
                                     from_folder="/home/gsteelman/Desktop/ML/Pictures/" ,
                                     to_folder="/home/gsteelman/Desktop/ML/AllData/LRtracker/" ,
                                      maxSize=maxsize)
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

    saver = tf.train.Saver()
    save_dir = 'checkpoints/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')
    saver.restore(sess=session, save_path=save_path)
    return myModel, session





def run_answers(myModel,session):
    video = v4l2capture.Video_device("/dev/video0")
    size_x, size_y = video.set_format(30, 20)
    print ("device chose {0}x{1} res".format(size_x, size_y))
    video.create_buffers(30)

    # Send the buffer to the device. Some devices require this to be done
    # before calling 'start'.
    video.queue_all_buffers()

    # Start the device. This lights the LED if it's a camera that has one.
    print("start capture")
    video.start()
    while True:
        select.select((video,), (), ())

                # The rest is easy :-)
        image_data = video.read_and_queue()
        imgDecode = np.frombuffer(image_data, dtype=np.uint8)
        # imgDecode = np.reshape(imgDecode, (-1,3))
        imgDecode = np.reshape(imgDecode, (size_y,size_x,3))
        im = Image.fromarray(imgDecode)
        im = im.resize((30,20),Image.ANTIALIAS)
        im = im.convert('L')
        im.show()
        im  =np.expand_dims(np.asarray(im), axis = 2)
        im  =np.expand_dims(im, axis = 0)
        # im.show()
        break
        # print(im,'-------------')
        # imgDecode = StringIO(imgDecode)
        # im = Image.open(imgDecode)

        # print(imgDecode)
        # im = Image.fromarray(imgDecode)
        # im.show()
        print(myModel.return_answers(im,session))

if __name__ =="__main__":
    myModel, session = init_model()
    run_answers(myModel,session)
