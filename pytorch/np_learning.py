import numpy as np


def matrix_multiple_row():
    # 矩阵乘法解释
    # 矩阵乘法 C 第n行 = A 第n行 -> B所有行的变换(线性组合)相加
    A = np.array([[1, 0, -3, 1]]).reshape(2, 2)
    B = np.array([1, 2, 3, 3, 4, 5]).reshape(2, 3)
    print(A)
    print(B)
    print(A @ B)
    """
    [[ 1  0]
     [-3  1]]
    [[1 2 3]
     [3 4 5]]
    [[ 1  2  3]
     [ 0 -2 -4]]
     
     1  2  3    =  1*r1+0*r2 =  1*[1 2 3] + 0 *[3 4 5]
     0 -2 -4    = -3*r1+1*r2 = -3*[1 2 3] + 1 *[3 4 5]
    """
    print("==========================================")
    A = np.array([[1, 0, -3, 1, 1, 1]]).reshape(3, 2)
    B = np.array([1, 2, 3, 3, 4, 5]).reshape(2, 3)
    print(A)
    print(B)
    print(A @ B)
    """
    [[ 1  0]
     [-3  1]
     [ 1  1]]
    [[1 2 3]
     [3 4 5]]
    [[ 1  2  3]
     [ 0 -2 -4]
     [ 4  6  8]]

     1  2  3    =  1*r1+0*r2 =  1*[1 2 3] + 0 *[3 4 5]
     0 -2 -4    = -3*r1+1*r2 = -3*[1 2 3] + 1 *[3 4 5]
     4  6  8    =  1*r1+1*r2 =  1*[1 2 3] + 1 *[3 4 5]
    """


def matrix_multiple_col():
    A = np.array([[1, 0, -3, 1]]).reshape(2, 2)
    B = np.array([1, 2]).reshape(-1)
    print(A)
    print(B)
    print(A @ B)
    """
    A:[
        [ 1  0]
        [-3  1]
      ]
    b [1  1]
    C = b[0]*c1+B[1]*C2 =1*[1,-3] + 2*[0,1] = [1,-1]
    """
    A = np.array([[1, 0, -3, 0]]).reshape(2, 2)
    B = np.array([1, 2, 3, 4]).reshape(2, 2)
    print(A)
    print(B)
    print(A @ B)
    """
    A:[
        [ a  b]
        [ c  d]
        [ e  f]
      ]
    B:[
        [ m  n]
        [ p  q]
      ]
    C列1 = [a c e]*m+[b d f]*n = 相加求和
    C列2 = [a c e]*n+[b d f]*q = 相加求和
    """


def np_zero():
    print(np.zeros((4,5)))



if __name__ == '__main__':
    # matrix_multiple_row()
    #matrix_multiple_col()
    np_zero()
