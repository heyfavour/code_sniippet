import numpy as np


def svd(A):
    U, S, V_T = np.linalg.svd(A)
    # print(U.shape)#AA_T的特征向量
    # print(S)  # A_TA 的特征值根号形式
    # print(np.linalg.eig(A.T @ A))
    # print(V_T.shape)#A_TA的特征向量
    k = S.shape[0]
    E = np.zeros_like(A)
    E[:k, :k] = np.diag(S)
    print(np.allclose(U @ E @ V_T, A))
    return U, S, V_T


def low_rank_k(A, U, S, V_T, k):
    S_k = np.zeros_like(A)
    S_k[:k, :k] = np.diag(S[:k])
    A_k = U @ S_k @ V_T
    return A_k


if __name__ == '__main__':
    A = np.random.randn(50, 50)
    U, S, V_T = svd(A)
    print(S.shape)
    print(np.linalg.matrix_rank(A))
    A_k = low_rank_k(A, U, S, V_T, 40)
    print(np.linalg.matrix_rank(A_k))
    #print(A)
    #print(A_k)
    error = np.linalg.norm(A - A_k)
    print(error)
