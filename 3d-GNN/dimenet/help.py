
def draw_bassel(n):
    """
    画出贝塞尔n和n+0.5阶的差异
    Returns:

    """
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.special as sp

    # 选择阶数
    n_integer = n  # 整数阶
    n_half_integer = n+0.5  # 半整数阶

    # 生成一组x值
    x = np.linspace(0, 10, 100)

    # 计算对应的贝塞尔函数值
    j_integer = sp.jv(n_integer, x)
    j_half_integer = sp.jv(n_half_integer, x)

    # 绘制图像
    plt.figure(figsize=(8, 6))
    plt.plot(x, j_integer, label=f'J_{n_integer}(x)')
    plt.plot(x, j_half_integer, label=f'J_{n_half_integer}(x)')
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Bessel Functions')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    draw_bassel(0)