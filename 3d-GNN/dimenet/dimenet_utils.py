# Shameless steal from: https://github.com/klicperajo/dimenet

import numpy as np
from scipy import special as sp
from scipy.optimize import brentq

try:
    import sympy as sym
except ImportError:
    sym = None


def Jn(r, n):
    # sp.jv 贝塞尔函数
    return np.sqrt(np.pi / (2 * r)) * sp.jv(n + 0.5, r)  # 振幅系数*第一类整数阶bessel函数


def Jn_zeros(n, k):
    zerosj = np.zeros((n, k), dtype='float32')
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype='float32')
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            """
            通过 brentq 函数来求解 Jn 函数的零点
            在量子力学中，Bessel 函数的零点出现在径向方程的解中
            (points[j], points[j + 1]) 确定了自变量 r 的取值范围
            i 是n+0.5解
            """
            foo = brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]
    # zerosj 数组中存储了 Jn 函数的前 n 个零点的值
    return zerosj


def spherical_bessel_formulas(n):
    # 计算了sin(x)/x的1阶导数、2阶导数、3阶导数，n阶倒数
    x = sym.symbols('x')

    f = [sym.sin(x) / x]  # sin(x)/x Dirichlet 核函数
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        f += [sym.simplify(b * (-x) ** i)]
        a = sym.simplify(b)
    # f 是n个sin/x不断求导组合的结果
    return f


def bessel_basis(n, k):
    zeros = Jn_zeros(n, k)  # n*k的贝塞尔函数零点矩阵
    # 为一组基底计算归一化系数
    normalizer = []  # 存储归一化系数
    for order in range(n):  # 0-1
        normalizer_tmp = []  # k个数
        for i in range(k):  # 0-k
            # jn(每个零点,i+1) 计算在球谐函数展开中的一个基底
            # 正实数根和阶数传递给 Jn 函数，可以计算出特定阶数下的球谐函数在这些根位置的值
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1) ** 2]
        normalizer_tmp = 1 / np.array(normalizer_tmp) ** 0.5
        normalizer += [normalizer_tmp]
    # normalizer -> n  k  归一化实数根的值 可以理解为只是一个系数
    f = spherical_bessel_formulas(n)  # n个函数
    x = sym.symbols('x')
    bess_basis = []
    # 此时有 zeros n*k零点矩阵  normalizer n*k 归一化系数矩阵 f n个函数
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            # normalizer[order][i] * f[order].subs(x, zeros[order, i] * x)
            # 归一化系数 * f(x) x=零点矩阵*x
            # 1.41421360172718 = 4.44288319/3.1415927
            bess_basis_tmp += [
                sym.simplify(normalizer[order][i] * f[order].subs(x, zeros[order, i] * x))
            ]
        bess_basis += [bess_basis_tmp]
    # bess_basis n行j列的矩阵  行是函数的阶数  列是系数不同的函数方程
    return bess_basis


def sph_harm_prefactor(k, m):
    # 球谐函数归一化系数
    # (2n+1)*[(k-m)!]/(4*pi*(k+m)!)
    return ((2 * k + 1) * np.math.factorial(k - abs(m)) / (4 * np.pi * np.math.factorial(k + abs(m)))) ** 0.5


def associated_legendre_polynomials(k, zero_m_only=True):
    # https://zhuanlan.zhihu.com/p/562322945
    # 计算关联勒让德多项式
    z = sym.symbols('z')  # 角度变量，通常在球坐标系中表示为余纬度
    P_l_m = [[0] * (j + 1) for j in range(k)]  # 用于存储关联勒让德多项式的值

    P_l_m[0][0] = 1  # 设置 l = 0 和 m = 0 时的关联勒让德多项式的值为1
    # [P_l_m 1维 2维 3维 n维]
    if k > 0:
        P_l_m[1][0] = z  # 置 l = 1 和 m = 0 时的关联勒让德多项式的值为 z

        for j in range(2, k):
            # 动态规划求勒让德多项式
            # P_{n+1}(x) = [(2n+1)/(n+1)]*x*P_{n} - [n/(n+1)]*P_{n-1}(x)
            P_l_m[j][0] = sym.simplify(((2 * j - 1) * z * P_l_m[j - 1][0] - (j - 1) * P_l_m[j - 2][0]) / j)
        if not zero_m_only:
            for i in range(1, k):
                P_l_m[i][i] = sym.simplify((1 - 2 * i) * P_l_m[i - 1][i - 1])
                if i + 1 < k:
                    P_l_m[i + 1][i] = sym.simplify(
                        (2 * i + 1) * z * P_l_m[i][i])
                for j in range(i + 2, k):
                    P_l_m[j][i] = sym.simplify(
                        ((2 * j - 1) * z * P_l_m[j - 1][i] -
                         (i + j - 1) * P_l_m[j - 2][i]) / (j - i))
    # 1 1
    # 2 Z
    # 3 1.5*x^2 -0.5
    # 4 4.375*x^4-3.75*x^2+0.375
    # ...
    # n -> [1 2 3 4 5 ...n ] 每行首列是勒让德多项式
    return P_l_m


def real_sph_harm(k, zero_m_only=True, spherical_coordinates=True):
    # 实球谐函数
    if not zero_m_only:
        S_m = [0]
        C_m = [1]
        for i in range(1, k):
            x = sym.symbols('x')
            y = sym.symbols('y')
            S_m += [x * S_m[i - 1] + y * C_m[i - 1]]
            C_m += [x * C_m[i - 1] - y * S_m[i - 1]]
    # P_l_m -> [1 2 3 4 5 ...n ] 每行首列是勒让德多项式
    P_l_m = associated_legendre_polynomials(k, zero_m_only)
    #
    if spherical_coordinates:
        theta = sym.symbols('theta')
        z = sym.symbols('z')
        # cos(theta) 代替 z
        for i in range(len(P_l_m)):
            for j in range(len(P_l_m[i])):
                if type(P_l_m[i][j]) != int:
                    P_l_m[i][j] = P_l_m[i][j].subs(z, sym.cos(theta))
        if not zero_m_only:
            phi = sym.symbols('phi')
            for i in range(len(S_m)):
                S_m[i] = S_m[i].subs(x, sym.sin(theta) * sym.cos(phi)).subs(y, sym.sin(theta) * sym.sin(phi))
            for i in range(len(C_m)):
                C_m[i] = C_m[i].subs(x, sym.sin(theta) * sym.cos(phi)).subs(y, sym.sin(theta) * sym.sin(phi))
    # k个 1 3 5 7 2k+1个0
    Y_func_l_m = [['0'] * (2 * j + 1) for j in range(k)]
    # 此时P_l_m  P_l_m -> [1 2 3 4 5 ...n ] 每行首列是勒让德多项式 sph_harm_prefactor(i, 0) 球谐函数归一化系数
    for i in range(k):
        # 球谐函数归一化系数 * 勒让德多项式
        Y_func_l_m[i][0] = sym.simplify(sph_harm_prefactor(i, 0) * P_l_m[i][0])
    # Y_func_l_m 归一化的勒让德多项式
    if not zero_m_only:
        for i in range(1, k):
            for j in range(1, i + 1):
                Y_func_l_m[i][j] = sym.simplify(
                    2 ** 0.5 * sph_harm_prefactor(i, j) * C_m[j] * P_l_m[i][j])
        for i in range(1, k):
            for j in range(1, i + 1):
                Y_func_l_m[i][-j] = sym.simplify(
                    2 ** 0.5 * sph_harm_prefactor(i, -j) * S_m[j] * P_l_m[i][j])
    return Y_func_l_m


if __name__ == '__main__':
    # spherical_bessel_formulas(5)
    bessel_basis(7, 6)
