import saveHelper

maxsize = 10000000
saveObj = saveHelper.saverObject(class_names=["middle","left","right"],
                                 img_dimensions=[480, 640, 3],
                                 from_folder="Pictures/" ,
                                 to_folder="data/cheese-10/batches-py/" ,
                                  maxSize=maxsize)
saveObj.cache_pictures(subfolders=["left/","middle/","right/"])
