import cv2 as cv

from DataLoader import TestImagesLoader


loader = TestImagesLoader("data/0", rotate=cv.ROTATE_90_COUNTERCLOCKWISE)
img = next(loader)

cv.imshow("test", img)
while (k := cv.waitKey(0)) != 27:
    pass
