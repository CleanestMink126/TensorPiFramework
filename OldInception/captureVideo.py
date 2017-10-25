
import numpy as np
import cv2
import time
import datetime

savePictureBool = False
lastTime = time.time()
timeToNextPicture = .5
cap = cv2.VideoCapture(1)
fileFolder = "left/"

if not cap:
    ans = cap.open()
    print(ans)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()


    # Our operations on the frame come here
    if ret:
        # Display the resulting frame
        cv2.imshow('frame',frame)
        if savePictureBool:
            cD = str(datetime.datetime.now()).replace(" ","-")
            filePath = "Pictures/"+fileFolder+ cD+".jpg"
            cv2.imwrite(filePath, frame)
        elif time.time() - lastTime > timeToNextPicture:
            savePictureBool = True
            lastTime = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
