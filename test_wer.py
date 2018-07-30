import numpy as np
def lcs(x, y, lenx, leny):
    a = np.zeros([lenx + 1, leny + 1])
    for i in range(lenx + 1):
        a[i][0] = i
    for j in range(leny + 1):
        a[0][j] = j
    for i in range(1,lenx + 1):
        for j in range(1,leny + 1):
            if x[i - 1] == y[j - 1]:
                a[i, j] = a[i - 1, j - 1]
            else:
                a[i, j] = a[i - 1, j - 1] + 1
            a[i][j] = min(a[i][j], a[i][j - 1] + 1)
            a[i][j] = min(a[i][j], a[i - 1][j] + 1)
    return 1-(a[lenx, leny] / leny)

a=[1,2,3]
b=[1,2,3,4]
c=lcs(a,b,3,4)
print(c)