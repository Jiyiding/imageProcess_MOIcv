# 本程序功能：在ext_val基础上，以10个像素为一组进行。
import cv2
import numpy as np
import pandas as pd

excel_dir = "excel/"
moon = cv2.imread("picture/B9/text_gray.bmp", 0)
print("moon.shape=", moon.shape)
row, column = moon.shape
print("row= ", row)
moon_f = np.copy(moon)
cv2.imshow("moon_f", moon_f)
# moon_f = moon_f.astype("float")


def list_to_excel(list, s):
    dataframe = pd.DataFrame(list)
    dataframe.to_excel(excel_dir + s + '_list.xls')

val_max = []  #每行最大值
val_min = []  #每行最小值
Ry = []  #每行最大值列坐标
Ly = []  #每行最小值列坐标
Max_Zb = [] #最大值坐标
Min_Zb = [] #最小值坐标
x_min = 126
x_max = 266
encope=1
for x in range(x_min, x_max):
    Max = 128  # 起始列，最大值所在列
    Min = 128  # 起始列，最小值所在列
    Max_val = moon_f[x, Max]
    Min_val = moon_f[x, Min]
    for y in range(128, 211):
        if (moon_f[x, y + 1] > Max_val):
            Max = y + 1
            Max_val = moon_f[x, y + 1]
        elif (moon_f[x, y + 1] < Min_val):
            Min = y + 1
            Min_val = moon_f[x, y + 1]
    # moon_f[x, Max] = 255
    # moon_f[x, Min] = 255
    val_max.append(Max_val)  # 最大值
    val_min.append(Min_val)  # 最小值
    Ry.append(Max)  # 最大值列坐标
    Ly.append(Min)  # 最小值列坐标
    Max_Zb.append([x, Max])  # 最大值坐标
    Min_Zb.append([x, Min])  # 最小值坐标
    #print("xxx=", x)
    if ((((x + 1 - x_min) % 10 == 0) or (x == x_max)) and (x > x_min)):
        ry_sum = 0
        ly_sum = 0
        print("x= ", x)
        #print("Ry=",Ry)
        print("encope= ",encope)
        print("Ry[encope:+10]= ",Ry[encope-10:])
        for ry in Ry[encope-10:]:
            print("jfoi",ry)
            ry_sum = ry_sum + ry / 10
        for ly in Ly[encope-10:]:
            ly_sum = ly_sum + ly / 10
        ry_sum = int(ry_sum)
        ly_sum = int(ly_sum)
        print("ry_sum= ", ry_sum)
       # print("type.ry_sum", type(ry_sum))
        for i in range(0, 9):
            moon_f[(x - i), ry_sum] = 255
            moon_f[(x - i), ly_sum] = 255
    encope=encope+1
# list_to_excel(Ry, 'Max')
# list_to_excel(Ly, 'Min')
# list_to_excel(Ry_max, 'Max_val')
# list_to_excel(Ly_min, 'Min_val')
All_Zb = np.hstack((Max_Zb, Min_Zb))
# img = cv2.drawContours(moon_f, All_Zb, 0, (0, 255, 0), 3)
cv2.imshow("moon_FF", moon_f)
cv2.imwrite("picture/B9/moon_FF.bmp", moon_f)
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
