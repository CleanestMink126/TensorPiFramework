import saveHelper

from Conv02 import *
maxsize = 10000000
saveObj = saveHelper.saverObject(class_names=["middle","left","right"],
                                 img_dimensions=[80, 60, 3],
                                 from_folder="/home/gsteelman/Desktop/Machine Learning/CNN Framework/Pictures/" ,
                                 to_folder="/home/gsteelman/Desktop/Machine Learning/CNN Framework/data/cheese-10/batches-py/" ,
                                  maxSize=maxsize)
#####saveObj.resize(subfolders=["middle/","left/","right/"])
saveObj.cache_pictures(subfolders=["middle/","left/","right/"])
# saveObj.num_batches=73
print(next(saveObj.random_batch(20)))

myModel = CNNModel(saveObj,saveObj.numImages)
filter_size1 = [5,5]          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = [5,5]           # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128


layer_conv1, weights_conv1 = \
    new_conv_layer(input=myModel.x_image,
                   num_input_channels=myModel.num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)


layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv2)

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=myModel.num_classes,
                         use_relu=False)

myModel.set_optimizer(layer_fc2)
myModel.optimize(num_iterations=1000,saveHelper=saveObj, batch_size = 30)
myModel.print_test_accuracy(saveHelper=saveObj)

#
#     #
