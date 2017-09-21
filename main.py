import saveHelper

from Conv02 import *
maxsize = 10000000
saveObj = saveHelper.saverObject(class_names=["0","1","2","3","4","5","6","7","8","9"],
                                 img_dimensions=[28, 28, 1],
                                 from_folder="/home/gsteelman/Desktop/Machine Learning/MNIST/trainingSample/" ,
                                 to_folder="/home/gsteelman/Desktop/Machine Learning/CNN Framework/data/cheese-10/batches-py/" ,
                                  maxSize=maxsize)
#####saveObj.resize(subfolders=["middle/","left/","right/"])
saveObj.cache_pictures(subfolders=["0/","1/","2/","3/","4/","5/","6/","7/","8/","9/"])
# saveObj.num_batches=73
# print(next(saveObj.random_batch(20)))

myModel = CNNModel(saveObj,64)
filter_size1 = [5,5]          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = [5,5]           # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128

#
# layer_conv1, weights_conv1 = \
#     new_conv_layer(input=myModel.x_image,
#                    num_input_channels=myModel.num_channels,
#                    filter_size=filter_size1,
#                    num_filters=num_filters1,
#                    use_pooling=True)
#
#
# layer_conv2, weights_conv2 = \
#     new_conv_layer(input=layer_conv1,
#                    num_input_channels=num_filters1,
#                    filter_size=filter_size2,
#                    num_filters=num_filters2,
#                    use_pooling=True)
#
# layer_flat, num_features = flatten_layer(layer_conv2)
#
# layer_fc1 = new_fc_layer(input=layer_flat,
#                          num_inputs=num_features,
#                          num_outputs=fc_size,
#                          use_relu=True)
# layer_fc2 = new_fc_layer(input=layer_fc1,
#                          num_inputs=fc_size,
#                          num_outputs=myModel.num_classes,
#                          use_relu=False)
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


myModel.set_optimizer(loss,y_pred)
myModel.optimize(num_iterations=10000,saveHelper=saveObj, batch_size = 64)
myModel.print_test_accuracy(saveHelper=saveObj)

#
#     #
