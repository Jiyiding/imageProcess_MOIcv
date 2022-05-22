# 本程序功能：在ext_val10_76基础上，进行缩放，符合实际形状
import cv2
import numpy as np
import pandas as pd

excel_dir = "excel4/"
moon = cv2.imread("picture/B9/text_gray.bmp", 0)
# print("moon.shape=", m
# oon.shape)
row, column = moon.shape
# print("row= ", row)
moon_f = np.copy(moon)


# cv2.imshow("moon_f", moon_f)

def list_to_excel(list, s):
    dataframe = pd.DataFrame(list)
    dataframe.to_excel(excel_dir + s + '_list.xls')


def Lj_centre(list):
    Len = len(list)
    # print("Len= ", Len)
    for i in range(0, Len - 1):  # 前闭后开区间
        n = list[i + 1] - list[i]
        # print("(x_min + (i + 1) * gap - 1)= ", (x_min + (i + 1) * gap - 1))
        if n > 0:
            for j in range(0, n + 1):
                moon_f[(x_min + (i + 1) * gap - 1), (list[i] + j)] = 255
                # print("(x_min + (i + 1) * gap - 1)= ",(x_min + (i + 1) * gap - 1))
        elif n < 0:
            for k in range(n, 1):
                moon_f[(x_min + (i + 1) * gap - 1), (list[i] + k)] = 255
        elif n == 0:
            moon_f[(x_min + (i + 1) * gap - 1), list[i]] = 255


def Lj_top(list):
    L = len(list)
    lyy0 = Ly_sum[0]
    ryy0 = Ry_sum[0]
    lyy1 = Ly_sum[L - 1]
    ryy1 = Ry_sum[L - 1]
    # print("lyy0", lyy0)
    # print("ryy0", ryy0)
    for i in range(lyy0, ryy0 + 1):
        moon_f[x_min, i] = 255
        # print("i= ", i)
    for i in range(lyy1, ryy1 + 1):
        moon_f[282, i] = 255


val_max = []  # 每行最大值
val_min = []  # 每行最小值
Ly = []  # 每行最大值列坐标
Ry = []  # 每行最小值列坐标
Ly_7 = []  # 每行最大值坐标缩小70％
Ry_7 = []  # 每行最小值坐标缩小70％
Max_Zb = []  # 最大值坐标
Min_Zb = []  # 最小值坐标
Ly_sum = []  # gap个最大值平均列坐标
Ry_sum = []  # gap个最小值平均列坐标
gap = 5  # 每隔gap个像素点连接到一起
x_min = 83
x_max = 285
encope = 1
for x in range(x_min, x_max):
    Max = 122  # 起始列，最大值所在列
    Min = 122  # 起始列，最小值所在列
    Max_val = moon_f[x, Max]
    Min_val = moon_f[x, Min]
    for y in range(122, 230):  # 左右边界，右边界是理论得出的。
        if (moon_f[x, y + 1] > Max_val):
            Max = y + 1
            Max_val = moon_f[x, y + 1]
        elif (moon_f[x, y + 1] < Min_val):
            Min = y + 1
            Min_val = moon_f[x, y + 1]
    m = (Min - Max) / 2
    m = int(m * 0.3)
    print("m=", m)
    Max = Max + m
    Min = Min - m
    Max_val = moon_f[x, Max]
    Min_val = moon_f[x, Min]
    val_max.append(Max_val)  # 最大值
    val_min.append(Min_val)  # 最小值
    Ly.append(Max)  # 最大值列坐标,list存储
    Ry.append(Min)  # 最小值列坐标,list存储
    Max_Zb.append([x, Max])  # 最大值坐标
    Min_Zb.append([x, Min])  # 最小值坐标
    # test有详解
    # 每隔gap个点取平均值并连接在一起，
    if ((((x + 1 - x_min) % gap == 0) or (x == x_max)) and (x > x_min)):
        ly_sum = 0
        ry_sum = 0
        # print("x= ", x)
        for ly in Ly[encope - gap:]:  # 除去前encope-gap个元素，其他均取出
            ly_sum = ly_sum + ly / gap
        for ry in Ry[encope - gap:]:  # 除去前encope-gap个元素，其他均取出
            ry_sum = ry_sum + ry / gap
        ly_sum = int(ly_sum)
        ry_sum = int(ry_sum)
        Ly_sum.append(ly_sum)
        Ry_sum.append(ry_sum)
        # 最大值最小值转变成255
        for i in range(0, gap):  # 前闭后开,结合下边x-i，所以应该(0,gap)
            moon_f[(x - i), ly_sum] = 255
            moon_f[(x - i), ry_sum] = 255
    encope = encope + 1

# cv2.imshow("moon_FF", moon_f)
# cv2.imwrite("picture/B9/moon_FF.bmp", moon_f)

R_L = [Ry[i] - Ly[i] for i in range(len(Ry))]  # 每行最小值、最大值所在列坐标之差
# 对最大值坐标与最小值坐标分别缩小70％
'''for x in range(len(R_L)):
    R_L2 = R_L[x] / 2
    m = int(R_L2 * 0.3)
    print("m= ", m)
    print("Ly[x]_Max= ", Ly[x])
    Ly_7.append((Ly[x] + m))
    Ry_7.append((Ry[x] - m))
R_L_7 = [Ry_7[i] - Ly_7[i] for i in range(len(Ry_7))]  # 每行最小值、最大值所在列坐标之差的70％
list_to_excel(Ly_7, "Max_7")
list_to_excel(Ry_7, "Min_7")'''
list_to_excel(Ly, 'Max')
list_to_excel(Ry, 'Min')
list_to_excel(val_max, 'Max_val')
list_to_excel(val_min, 'Min_val')
list_to_excel(R_L, "Max_Min")
Ly.append(Ry)
# All_Zb = np.hstack((Max_Zb, Min_Zb))

# 两侧封边
Lj_centre(Ly_sum)
Lj_centre(Ry_sum)
cv2.imshow("moon_FF5", moon_f)
cv2.imwrite("picture/B9_7/moon_FF5.bmp", moon_f)

# 上下封顶
Lj_top(Ly_sum)
Lj_top(Ry_sum)
cv2.imshow("moon_FF55", moon_f)
cv2.imwrite("picture/B9_7/moon_FF55.bmp", moon_f)

cv2.waitKey()
