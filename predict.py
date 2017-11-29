import saveHelper

from Conv02 import *
import tensorflow as tf
import os
import cv2
import select
import numpy as np
from io import StringIO
import moveServo


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
    cap = cv2.VideoCapture(0)
    print cap.get(3)
    print cap.get(4)
    angle = 90
    direction = 3

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(gray, (30,20))
        im2 = myModel.saverObject._convert_images(im)

        # Our operations on the frame come here
        if ret:
            # Display the resulting frame
            #cv2.imshow('frame',im)
            # myModel.return_answers(im,session)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            ans =  myModel.return_answers(im2,session)
            if ans == 1 and angle < 180:
                angle += direction
                moveServo.SetAngle(angle)
            elif ans == 2 and angle > 0:
                angle -= direction
                moveServo.SetAngle(angle)


if __name__ =="__main__":
    myModel, session = init_model()
    run_answers(myModel,session)
