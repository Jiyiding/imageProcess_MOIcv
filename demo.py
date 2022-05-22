import cv2
import numpy as np
import pandas as pd

#本程序是完成梯度计算
moon = cv2.imread("picture/B9/text_gray.bmp", 0)
print("moon.shape", type(moon))
row, column = moon.shape
print("row= ", row)
moon_f = np.copy(moon)
cv2.imshow("moon_f", moon_f)
moon_f = moon_f.astype("float")

gradient = np.zeros((row, column))

for x in range(row - 1):
    for y in range(column - 1):
        print("moon_f= ",moon_f[x+1,y])
        gx = abs(moon_f[x + 1, y] - moon_f[x, y])
        gy = abs(moon_f[x, y + 1] - moon_f[x, y])
        gradient[x, y] = gx + gy



sharp = moon_f + gradient

sharp = np.where(sharp < 0, 0, np.where(sharp > 255, 255, sharp))

gradient = gradient.astype("uint8")
gradient=gradient*20
print("gradient= ", gradient)
print("moon_f", moon_f)
print("sharp", sharp)

sharp = sharp.astype("uint8")
cv2.imshow("moon", moon)
cv2.imshow("gradient", gradient)
cv2.imshow("sharp", sharp)
cv2.waitKey()
