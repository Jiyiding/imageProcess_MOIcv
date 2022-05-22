# 本程序功能：在已获得ROI基础上，提取3D深度信息
import cv2
import numpy as np
import pandas as pd


def list_to_excel(list, s):
    dataframe = pd.DataFrame(list)
    dataframe.to_excel(excel_dir + s + '.xls')


excel_dir = "excel3D_70/"
sun_LK = cv2.imread("picture/B9_7/LK_grayb9_55.bmp", 0)
sun_ROI = cv2.imread("picture/B9_7/ROI_grayb9_55.bmp", 0)
sun_Gray = cv2.imread("picture/B9_7/text_gray.bmp", 0)
row, col = sun_LK.shape
# print("row= ", row)
sun_lk = np.copy(sun_LK)
sun_roi = np.copy(sun_ROI)
sun_gray = np.copy(sun_Gray)
cv2.imshow("sun_roi原始图", sun_roi)

sun_val_roi = np.ones((row, col)) * 0
sun_val_bias_roi = np.ones((row, col)) * 0
sun_val_grad_roi = np.ones((row, col)) * 0
sun_val_high_roi = np.ones((row, col)) * 0

# 逐行扫描：
# 1，定位；所需值---像素值大于零的列坐标
# 2，求深度值；所需值---差值-每行期望、梯度-每个列坐标左右邻坐标（注意像素值为0的边界）
X_roi = []  # 有效值所在行
Y_roi = [[]]  # 有效值所在列
val_roi = [[]]  # 依次保存有效值，按行分组
val_Exp = []  # 保存每行期望
val_bias_roi = [[]]  # 依次保存差值，按行分组
val_grad_roi = [[]]  # 依次保存梯度值，按行分组
val_high_roi = [[]]  # 最终深度值，按行分组
k_num = []  # 依次保存每行有效值数量
grad_L1 = 20  # 左梯度a权值
grad_L2 = 6  # 左梯度b权值
grad_R1 = 20  # 右梯度a权值
grad_R2 = 6  # 右梯度b权值
grad_U1 = 20  # 上梯度a权值
grad_U2 = 6  # 上梯度b权值
grad_D1 = 20  # 下梯度a权值
grad_D2 = 6  # 下梯度b权值
# 为什么像素值与期望的偏差权值为负数？
# 因为不管是像素值过大（亮）还是过小（暗）都是边缘处，只有越靠近期望，越是中心线处，深度越大。
K = -1  # 偏差权值;

# n = 1
for i in range(0, row):
    y_roi = []  # 每行有效值所在列
    val_x = []  # 每行有效值
    val_bias_x = []  # 每行偏差
    val_grad_x = []
    k = 0
    for j in range(0, col):
        if (sun_roi[i][j] > 0):
            y_roi.append(j)
            val_x.append(sun_roi[i][j])
            k = k + 1
            # 下一步就要求梯度变化；用列表不好算，还是用坐标，不考虑边界条件，用源灰度图计算
            # 如果 左和右/上和下 就重复了，所以就算 正方向前边减中心。不要L和U.
            val_L2 = int(sun_gray[i][j]) - int(sun_gray[i][j - 2])
            val_L1 = int(sun_gray[i][j]) - int(sun_gray[i][j - 1])
            val_R1 = int(sun_gray[i][j + 1]) - int(sun_gray[i][j])
            val_R2 = int(sun_gray[i][j + 2]) - int(sun_gray[i][j])
            val_U1 = int(sun_gray[i][j]) - int(sun_gray[i - 1][j])
            val_U2 = int(sun_gray[i][j]) - int(sun_gray[i - 2][j])
            val_D1 = int(sun_gray[i + 1][j]) - int(sun_gray[i][j])
            val_D2 = int(sun_gray[i + 2][j]) - int(sun_gray[i][j])

            val_L = abs(val_L1 * grad_L1 + val_L2 * grad_L2)
            val_R = abs(val_R1 * grad_R1 + val_R2 * grad_R2)
            val_U = abs(val_U1 * grad_U1 + val_U2 * grad_U2)
            val_D = abs(val_D1 * grad_D1 + val_D2 * grad_D2)
            # 就不要L和U了.
            grad = val_R+val_D
            # print("grad= ", grad)
            val_grad_x.append(grad)
    k_num.append(k)
    if (k >= 1):  # 如果所在行存在有效值
        X_roi.append(i)
        Y_roi.append(y_roi)
        val_grad_roi.append(val_grad_x)
        val_roi.append(val_x)
        L = len(val_x)
        X_sum = sum(val_x)
        val_exp = int(X_sum / L)
        val_Exp.append(val_exp)
        for m in val_x:
            bias = abs(m - val_exp) * K
            val_bias_x.append(bias)  # 保存每行像素值与期望值之差
        val_bias_roi.append(val_bias_x)

# 要开始遍历梯度和偏差的二维列表了，看看是用列表方便还是转化为数组方便！！！
'''print("X_roi.L= ", len(X_roi))
np_grad_roi = np.array(val_grad_roi)
np_bias_roi = np.array(val_bias_roi)
print("val_Grad.shape= ", np_grad_roi.shape)
print("val_bias.shape= ", np_bias_roi.shape)'''
del (val_roi[0])
del (Y_roi[0])
del (val_grad_roi[0])
del (val_bias_roi[0])
print("Y_roi.len= ", len(Y_roi))
print("val_grad_roi.len= ", len(val_grad_roi))
print("val_bias_roi.len= ", len(val_bias_roi))

# 开始遍历梯度值、偏差值的二维链表，赋值并保存
for i in range(len(val_grad_roi)):
    if ((len(val_grad_roi[i])) != (len(val_bias_roi[i]))):
        print("出错了！！！梯度与偏差在第 %d行数量不等", i)
    val_high_x = []
    di = X_roi[i]  # 遍历有效值属于哪一行
    for j in range(len(val_grad_roi[i])):
        val_high = val_grad_roi[i][j] + val_bias_roi[i][j]
        dj = Y_roi[i][j]  # 遍历有效值属于哪一列
        sun_roi[di][dj] = val_high  # 赋值给图像
        val_high_x.append(val_high)
    val_high_roi.append(val_high_x)
del (val_high_roi[0])

print("X_roi[0]= ",X_roi[0])
for i in range(row):
    if ((i >= X_roi[0]) and (i <= X_roi[len(X_roi) - 1])):
        #for j in range(col):
        e = 0
        for a in (Y_roi[i - 82]):
            #print("i-82= ",(i-82))
            #print("a=",a)
            sun_val_roi[i][a] = val_roi[i - 82][e]
            sun_val_grad_roi[i][a] = val_grad_roi[i - 82][e]
            sun_val_bias_roi[i][a] = -(val_bias_roi[i - 82][e])
            sun_val_high_roi[i][a] = val_high_roi[i - 82][e]
            e = e + 1

cv2.imshow("22D_sun_val_roi", sun_val_roi)
cv2.imshow("sun_val_grad_roi", sun_val_grad_roi)
cv2.imshow("sun_val_bias_roi", sun_val_bias_roi)
cv2.imshow("sun_val_high_roi", sun_val_high_roi)
cv2.imwrite("picture/B9_7/sun_val_roi.bmp", sun_val_roi)
cv2.imwrite("picture/B9_7/sun_val_grad_roi.bmp", sun_val_grad_roi)
cv2.imwrite("picture/B9_7/sun_val_bias_roi.bmp", sun_val_bias_roi)
cv2.imwrite("picture/B9_7/sun_val_high_roi.bmp", sun_val_high_roi)

list_to_excel(X_roi, "X_roi")
list_to_excel(val_Exp, "val_Exp")
list_to_excel(k_num, "k_num")
list_to_excel(Y_roi, "2D_Y_roi")
list_to_excel(val_roi, "2D_val_roi")
list_to_excel(val_grad_roi, "2D_val_grad_roi")
list_to_excel(val_high_roi, "2D_val_high_roi")
list_to_excel(val_bias_roi, "2D_val_bias_roi")

cv2.imshow("sun_roi深度图", sun_roi)
# cv2.imwrite("picture/B9_7/roi_3D_7.bmp", sun_roi)
sun_Guss_3d = cv2.GaussianBlur(sun_roi, (5, 5), 0)
for i in range(1):
    sun_Guss_3d = cv2.GaussianBlur(sun_Guss_3d, (3, 3), 0)

cv2.imshow("sun_Guss_3d", sun_Guss_3d)
cv2.imwrite("picture/B9_7/sun_Guss_3d_7.bmp", sun_Guss_3d)
# 怎么办！！！不够平滑啊！调权值？调滤波？还是怎样？
print("恭喜你，通过了！！！")

cv2.waitKey()
