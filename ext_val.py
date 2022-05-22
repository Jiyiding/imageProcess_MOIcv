# 本程序功能：获取图像某一区域极值
import cv2
import numpy as np
import pandas as pd

excel_dir = "excel/"
moon = cv2.imread("picture/B9/text_gray.bmp", 0)
#print("moon.shape=", moon.shape)
row, column = moon.shape
#print("row= ", row)
moon_f = np.copy(moon)
cv2.imshow("moon_f", moon_f)
# moon_f = moon_f.astype("float")

gradient = np.zeros((row, column))


def list_to_excel(list, s):
    dataframe = pd.DataFrame(list)
    dataframe.to_excel(excel_dir + s + '_list.xls')



val_max = []  #每行最大值
val_min = []  #每行最小值
Ry = []  #每行最大值列坐标
Ly = []  #每行最小值列坐标
Max_Zb = [] #最大值坐标
Min_Zb = [] #最小值坐标
x_min = 97
x_max = 290
for x in range(x_min, x_max):
    Max = 92
    Min = 92
    Max_val = moon_f[x, 128]
    Min_val = moon_f[x, 128]
    # 这个y的右边界是将ROI对称之后图像中暗区域包含进去，
    # 因为缺陷左右亮暗区域近似对称---尝试做仿真分析证明
    # y的取值很重要，尤其右边界
    for y in range(122, 255):
        if (moon_f[x, y + 1] > Max_val):
            Max = y + 1 #最大值列坐标
            Max_val = moon_f[x, y + 1]
        elif (moon_f[x, y + 1] < Min_val):
            Min = y + 1 #最小值列坐标
            Min_val = moon_f[x, y + 1]
    moon_f[x, Max] = 255
    moon_f[x, Min] = 255
    val_max.append(Max_val)  #最大值
    val_min.append(Min_val)  #最小值
    Ry.append(Max)  #最大值列坐标
    Ly.append(Min)  #最小值列坐标
    Max_Zb.append([x, Max]) #最大值坐标
    Min_Zb.append([x, Min]) #最小值坐标

list_to_excel(Ry, 'Max76')
list_to_excel(Ly, 'Min76')
# list_to_excel(Ry_max, 'Max_val')
# list_to_excel(Ly_min, 'Min_val')
All_Zb = np.hstack((Max_Zb, Min_Zb))
# img = cv2.drawContours(moon_f, All_Zb, 0, (0, 255, 0), 3)
cv2.imshow("moon_FF", moon_f)
#cv2.imwrite("picture/B9/moon_FF.bmp", moon_f)
# cv2.imshow("img", img)
'''sharp = moon_f + gradient

sharp = np.where(sharp < 0, 0, np.where(sharp > 255, 255, sharp))

gradient = gradient.astype("uint8")
gradient = gradient * 20
print("gradient= ", gradient)
print("moon_f", moon_f)
print("sharp", sharp)

sharp = sharp.astype("uint8")
cv2.imshow("moon", moon)
cv2.imshow("gradient", gradient)
cv2.imshow("sharp", sharp)'''
cv2.waitKey()
