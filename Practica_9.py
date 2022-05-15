import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('monedas_3.jpg')
rimg = cv2.resize(img, (1000,600))
img_g = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
img_s = cv2.GaussianBlur(img_g,(7,7),0)

roi = cv2.selectROI(rimg)
template = rimg[int(roi[1]):int(roi[1]+roi[3]),int(roi[0]):int(roi[0]+roi[2])]
cv2.imshow("Template",template)
template_g = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
w, h = template_g.shape[::-1]

res = cv2.matchTemplate(img_s,template_g,cv2.TM_CCOEFF_NORMED)
threshold = 0.85
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(rimg, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv2.imwrite('res.jpg',rimg)
match = cv2.imread('res.jpg')
cv2.imshow('Matching',match)

cv2.waitKey(0)
cv2.destroyAllWindows()
