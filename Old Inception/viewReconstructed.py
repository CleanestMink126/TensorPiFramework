import inception
from inception import transfer_values_cache
import cifar10,os
import matplotlib.pyplot as plt
import time


model = inception.Inception()

class_names = cifar10.load_class_names()
images_train, cls_train, labels_train = cifar10.load_training_data()
#images_test, cls_test, labels_test = cifar10.load_test_data(1)
# Get the first images from the test-set.

# images_scaled = images_train * 255.0

print('started')
start = time.time()
for image in images_train:
    transferImage = model.transfer_values(image = image)
    transferImage = transferImage.reshape((32, 64))
    plt.imshow(image)
    plt.show()
    plt.imshow(transferImage)
    plt.show()

end = time.time()
print((end - start)/16)
# print(timeit.timeit(iterate_transfer))
# start = timeit.default_timer()
# iterate_transfer()
# end = timeit.default_timer()
# print(end - start)
