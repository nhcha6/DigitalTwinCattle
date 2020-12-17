# the script contains functions to run multilevel discrete wavelet transformation on selected data. It will be integrated
# into the other scripts which generate data and plots. Currently copying data across from the other
# scripts.

import pywt
import matplotlib.pyplot as plt

def extrapolate_heat(num_days):
    state_hot_ex = []
    state_other_ex = []
    for i in range(num_days):
        state_hot_ex += eating_hot
        state_other_ex += eating_other

    return state_hot_ex, state_other_ex

def DWT_level_2(signal1, signal1_name, signal2, signal2_name, signal3, signal3_name):
    coefs1 = pywt.wavedec(signal1, 'bior2.2', level=2)
    cA2o, cD2o, cD1o = coefs1

    coefs2 = pywt.wavedec(signal2, 'bior2.2', level=2)
    cA2h, cD2h, cD1h = coefs2

    coefs3 = pywt.wavedec(signal3, 'bior2.2', level=2)
    cA2c, cD2c, cD1c = coefs3

    plt.figure(figsize=(8, 7))
    plt.subplot(4, 1, 1)
    plt.plot(signal1, label=signal1_name)
    plt.plot(signal2, label=signal2_name)
    plt.plot(signal3, label=signal3_name)
    plt.title("Signal")
    plt.legend(loc="upper right")

    plt.subplot(4, 1, 2)
    plt.plot(cA2o)
    plt.plot(cA2h)
    plt.plot(cA2c)
    plt.title("Second Approximate")

    plt.subplot(4, 1, 3)
    plt.plot(cD2o)
    plt.plot(cD2h)
    plt.plot(cD2c)
    plt.title("Second Detail")

    plt.subplot(4, 1, 4)
    plt.plot(cD1o)
    plt.plot(cD1h)
    plt.plot(cD1c)
    plt.title("First Detail")

    plt.tight_layout()

#print(pywt.wavelist())
