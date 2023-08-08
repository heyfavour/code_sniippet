# https://blog.manyacan.com/archives/2041/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft, ifft


def generate_y():
    # 样本采样率(每秒钟采几个样)
    sr = 100
    # 样本采样时间间隔
    T_s = 1. / sr  # T_s=0.01
    # 样本采样的时间长度为5s，样本采样的个数为500，所以说最小频率=1/(N*T_s)=1/(500*0.01)=0.2Hz
    t = np.arange(0, 5, T_s)
    y_0 = 3 * np.sin(2 * np.pi * 1 * t)  # ω=2π/T=2πf,sin(ωt)
    y_1 = np.sin(2 * np.pi * 4.3 * t)
    y_2 = 0.5 * np.sin(2 * np.pi * 7 * t)
    y_3 = y_0 + y_1 + y_2
    return y_0, y_1, y_2, y_3, t, sr


# y_0, y_1, y_2, y_3, t, sr = generate_y()


def draw_y3(y_0, y_1, y_2, y_3, t):
    fig, ax = plt.subplots(2, 2, figsize=(6, 6), sharex='col')
    fig.subplots_adjust(hspace=0.2, wspace=0.3)

    ax[0, 0].plot(t, y_0)
    ax[0, 1].plot(t, y_1)
    ax[1, 0].plot(t, y_2)
    ax[1, 1].plot(t, y_3)

    ax[1, 0].set_xlabel('Time')
    ax[1, 1].set_xlabel('Time')
    ax[0, 0].set_ylabel('Amplitude')
    ax[1, 0].set_ylabel('Amplitude')

    ax[0, 0].set_title('$y_0=3\sin(2 \pi \cdot t)$')
    ax[0, 1].set_title('$y_1=\sin(2 \pi \cdot 4.3t)$')
    ax[1, 0].set_title('$y_2=0.5\sin(2 \pi \cdot 7t)$')
    ax[1, 1].set_title('$y_3=y_0+y_1+y_2$')

    plt.show()


# draw_y3(y_0, y_1, y_2, y_3, t)


def dofft(y_3, t, sr):
    # 傅里叶变换结果，返回长度=1/2奈奎斯频率/最小频率=1/2*100/0.2=250，250*2=500
    y_3_fft = fft(y_3)
    N = len(t)  # 500 采样个数
    # 采样数据的idx
    n = np.arange(N)
    # 总的采样时间
    T = N / sr  # 5
    # 频率区间：奈奎斯频率/2；n/T*sr = n/(N*T_s)=n*w；刚好奈奎斯频率限制和最小刻度值一起给出了频率空间，直接查看freq就懂了
    freq = n / T
    # 实际幅度
    y_3_fft_norm = y_3_fft / N * 2  # x_k = c_k/N = 2/N * A
    return freq, y_3_fft, y_3_fft_norm


# freq, y_3_fft, y_3_fft_norm = dofft(y_3,t,sr)


def draw_fft(freq, y_3_fft, y_3_fft_norm):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=150)  # 画板
    ax[0].stem(freq, np.abs(y_3_fft), 'b', markerfmt=' ', basefmt='-b')  # 100 y_3_fft
    ax[0].set_xlabel('Freq(Hz)')
    ax[0].set_ylabel('FFT Amplitude |X(freq)|')

    ax[1].stem(freq, np.abs(y_3_fft_norm), 'b', markerfmt=' ', basefmt='-b')
    # ax[1].set_xlim(0, 10)
    ax[1].set_xlabel('Freq(Hz)')
    ax[1].set_ylabel('FFT Amplitude |X(freq)|')

    ax[1].annotate(r'$y_0=3\sin(2 \pi \cdot t)$',
                   xy=(1, 2.8),
                   xytext=(+30, -30),
                   textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2")
                   )

    ax[1].annotate(
        r'$y_1=\sin(2 \pi \cdot 4.3t)$',
        xy=(4.2, .4),
        xytext=(+10, 50),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-.5")
    )

    ax[1].annotate(
        r'$y_2=0.5\sin(2 \pi \cdot 7t)$',
        xy=(7, .4),
        xytext=(+10, 30),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-.5")
    )

    plt.show()


# draw_fft(freq, y_3_fft, y_3_fft_norm)

#########################################################################################噪声去除
def make_noise_clean():
    # 采样频率
    sr = 1000
    # 采样时间间隔
    ts = 1. / sr
    # 样本采样点
    t = np.arange(0, 2, ts)  # 在2s的时间内，每隔(1./sr)进行一次采样
    # 原始信号
    f_clean = 5 + 2 * np.sin(2 * np.pi * 10 * t + 3) + 5 * np.sin(2 * np.pi * 30 * t)  # ω=2π/T=2πf,sin(ωt)=sin(2πf·t)
    # 信号噪音
    f_noise = f_clean + 3 * np.random.randn(len(t))
    return f_clean, f_noise, t, sr


# f_clean, f_noise, t, sr = make_noise_clean()


def draw_noise(f_clean, f_noise, t):
    # 绘制信号
    fig, ax = plt.subplots(figsize=(12, 3))

    ax.plot(t, f_noise, linewidth=.5, color='c', label='Noisy')
    ax.plot(t, f_clean, linewidth=.5, color='r', label='Clean')

    ax.set_xlabel('Sampling Time')
    ax.set_ylabel('Amplitude')
    ax.legend()

    plt.show()


# draw_noise(f_clean, f_noise, t)


def noise_fft(f_noise, sr):
    # 傅里叶变换
    """
    1、采样频率sr = 1000
    2、采样时间间隔 = 1/sr = 0.001
    3、采样点个数 = 2000
    4、最小频率 = 1/(N*T_s)=1/(2000/1000) = 0.5Hz
    """

    # 傅立叶变换结果，返回长度=1/2奈奎斯频率/最小频率=1/2*1000/0.2=250，再加上负频率，250*2=500
    X = fft(f_noise)
    N = len(X)

    # 采样数据的idx
    n = np.arange(N)

    # 总的采样时间
    T = N / sr

    # 频率区间：奈奎斯频率/2；n/T=n/N*sr=n/(N*T_s)=n*w；刚好奈奎斯频率限制和最小刻度值一起给出了频率空间，直接查看freq就懂了
    freq = n / T
    X_norm = X / N * 2  # 为何获取真实幅度的时候要✖(2/N)？
    return freq, X, X_norm, N


# freq, X, X_norm, N = noise_fft(f_noise, sr)


def draw_noise_fft(freq, X, X_norm):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].stem(freq, np.abs(X), 'b', markerfmt=' ', basefmt='-b')
    # ax[0].set_xlim(0, 10)
    ax[0].set_xlabel('Freq(Hz)')
    ax[0].set_ylabel('FFT Amplitude |X(freq)|')

    # 实际幅度
    ax[1].stem(freq, np.abs(X_norm), 'b', markerfmt=' ', basefmt='-b')
    ax[1].set_xlim(-1, 100)
    # ax[1].set_xlim(-1, 40)
    ax[1].set_xlabel('Freq(Hz)')
    ax[1].set_ylabel('FFT Amplitude |X(freq)|')
    ax[1].annotate(
        r'5(Constant terms)',
        xy=(0, 5),
        xytext=(+10, 50),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-.5")
    )
    ax[1].annotate(
        r'$2\sin(2 \pi \cdot 10 t+3)$',
        xy=(10, 2),
        xytext=(+10, 30),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-.5")
    )
    ax[1].annotate(
        r'$5\sin(2 \pi \cdot 30 t)$',
        xy=(30, 4),
        xytext=(+10, 30),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-.5")
    )

    plt.show()


# draw_noise_fft(freq, X, X_norm)


def noise_clean(X):
    freq_clean = pd.Series(X).apply(lambda x: x if np.abs(x) > 1000 else 0).to_numpy()
    return freq_clean


# freq_clean = noise_clean(X)


def draw_clean(freq_clean, N, freq):
    fig, ax = plt.subplots(figsize=(10, 4))
    # freq_clean = freq_clean / N * 2
    ax.stem(freq, np.abs(freq_clean), 'b', markerfmt=' ', basefmt='-b')

    ax.set_xlim(-1, 100)
    ax.set_xlabel('Freq(Hz)')
    ax.set_ylabel('FFT Amplitude |X(freq)|')
    ax.annotate(
        r'5(Constant terms)',
        xy=(0, 5 * N / 2),
        xytext=(+10, 50),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-.5")
    )
    ax.annotate(
        r'$2\sin(2 \pi \cdot 10 t+3)$',
        xy=(10, 2 * N / 2),
        xytext=(+10, 30),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-.5")
    )
    ax.annotate(
        r'$5\sin(2 \pi \cdot 30 t)$',
        xy=(30, 4 * N / 2),
        xytext=(+10, 30),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-.5")
    )

    plt.show()


# draw_clean(freq_clean,N,freq)

def ifft_noise(freq_clean, f_clean, t):
    # 逆傅里叶变换
    ix = ifft(freq_clean)

    # 绘制信号
    fig, ax = plt.subplots(figsize=(12, 3))

    ax.plot(t, ix, linewidth=.5, color='c', label='IFFT')
    ax.plot(t, f_clean, linewidth=.5, color='r', linestyle='-.', label='Raw signal', alpha=0.7)

    ax.set_xlabel('Sampling Time')
    ax.set_ylabel('Amplitude')
    ax.legend()

    plt.show()


# ifft_noise(freq_clean,f_clean)

def DFT_slow(x):
    x = np.asarray(x, dtype=float)  # ensure the data type
    N = x.shape[0]  # get the x array length 1024
    n = np.arange(N)  # 1d array 1024
    k = n.reshape((N, 1))  # 2d array, 1024 x 1, aka column array
    M = np.exp(-2j * np.pi * k * n / N)  # 1024*1024
    return np.dot(M, x)  # [a,b] . [c,d] = ac + bd, it is a sum


x = np.random.random(10)
print(np.allclose(DFT_slow(x), fft(x)))
