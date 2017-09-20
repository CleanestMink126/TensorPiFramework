import saveHelper

import Conv02.py
maxsize = 10000000
saveObj = saveHelper.saverObject(class_names=["middle","left","right"],
                                 img_dimensions=[640, 480, 3],
                                 from_folder="/home/gsteelman/Desktop/Machine Learning/CNN Framework/Pictures/" ,
                                 to_folder="/home/gsteelman/Desktop/Machine Learning/CNN Framework/data/cheese-10/batches-py/" ,
                                  maxSize=maxsize)
# saveObj.cache_pictures(subfolders=["middle/","left/","right/"])
saveObj.random_batch(6)
