import numpy as np
import cv2
import pandas as pd

def list_to_excel(list, s):
    dataframe = pd.DataFrame(list)
    dataframe.to_excel(excel_dir + s + '.xls')

excel_dir = "excel3D_70/"
a = [[1, 2, 3, 4], [33, 3, 2, 2]]
for s in (a[0]):
    print(s)
#list_to_excel(a, "test")

'''
imgray = cv2.imread("picture\B9\edgeb9.bmp")
img = cv2.imread("picture\B9\M.bmp")
# imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(imgray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("contours= ", contours)
img = cv2.drawContours(img, contours, 0, (0, 255, 0), 3)
cv2.imshow("img", img)
cv2.waitKey()
'''
# 证明这样写是对的！！
'''x_min = 83
x_max = 285
encope = 1
gap = 6
List = []
l = len(List)
for x in range(x_min, x_max):
    List.append(x)
    if ((((x + 1 - x_min) % gap == 0) or (x == x_max)) and (x > x_min)):
        print("x= ", x)
        Sum = 0
        for i in List[encope - gap:]:
            Sum = Sum + i / gap
        print("list.i=", int(Sum))
    encope = encope + 1
'''
# 实现两个列表对应元素相减
'''
R = [1, 2, 3, 4, 5, 6, 7]
L = [2, 2, 2, 2, 2, 2, 2]
c=[R[i]-L[i] for i in range(len(R))]
print("R-L= ", c)'''
'''
k = 1
j = 2
W = [[]]
A = [1, 20, 3]
B = [2, 2, 4, 4]
for m in A:
    m = m - 1
    print("m= ", m)

W.append(A)
print("W= ", W)
W.append(B)
print("W= ", W)
print("W.shape", type(W))
del (W[0])
print("W1==A", W[0])
print("W2==B",W[1])
print("A.sum= ", sum(A))


print("W= ", W)
for i in range(len(A)):
    print("i= ", i)
'''


