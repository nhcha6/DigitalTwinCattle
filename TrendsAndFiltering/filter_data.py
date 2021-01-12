# the script contains functions to run multilevel discrete wavelet transformation on selected data. It will be integrated
# into the other scripts which generate data and plots. Currently copying data across from the other
# scripts.

import pywt
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import kaiserord, lfilter, firwin, freqs, butter, filtfilt


def extrapolate_heat(num_days, state_hot, state_other):
    state_hot_ex = []
    state_other_ex = []
    for i in range(num_days):
        state_hot_ex += state_hot
        state_other_ex += state_other

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

def fourier_transform(signal, title):
    # declare sample rate
    fs = 24

    # extract signal
    signal_fft = np.fft.fft(signal)

    # frequency
    fft_fre = np.fft.fftfreq(n=signal_fft.size, d=1 / fs)

    # plot magnitude of positive frequencies
    plt.figure()
    plt.plot(fft_fre[0:48], abs(signal_fft.real[0:48]))
    plt.title("Fourier Transform " + title)
    plt.xlabel("Frequency (1/days)")
    plt.ylabel("Magnitude")
#print(pywt.wavelist())

def fil_lp_filter(cutoff_hz_list, width_hz_list, signal, signal_name):
    i = 0
    plt.figure(figsize=(len(width_hz_list)*5, len(cutoff_hz_list)*2))
    plt.suptitle("Low Pass vs Unfiltered Signal (" + signal_name + ")")
    for cutoff_hz in cutoff_hz_list:
        for width_hz in width_hz_list:
            i+=1
            # sample rate is 24 per day
            sample_rate = 24

            # The Nyquist rate of the signal.
            nyq_rate = sample_rate / 2.0
            t = np.arange(len(signal)) / sample_rate

            # The desired width of the transition from pass to stop,
            # relative to the Nyquist rate.
            width = width_hz/nyq_rate

            # The desired attenuation in the stop band, in dB.
            ripple_db = 60.0

            # Compute the order and Kaiser parameter for the desired FIR filter.
            N, beta = kaiserord(ripple_db, width)

            # Use firwin with a Kaiser window to create a lowpass FIR filter. Taps are
            # the coefficients for the filter
            taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

            # Use lfilter to filter x with the FIR filter.
            filtered_signal = lfilter(taps, 1.0, signal)

            # The phase delay of the filtered signal.
            delay = 0.5 * (N-1) / sample_rate

            # plot filter response
            # plt.figure()
            # w, h = freqz(taps, worN=8000)
            # plt.plot((w / np.pi) * nyq_rate, np.absolute(h), linewidth=2)
            # plt.xlabel('Frequency (Hz)')
            # plt.ylabel('Gain')
            # plt.title('Filter Frequency Response')
            # plt.ylim(-0.05, 1.05)
            # plt.grid(True)


            # Plot the original signal
            plt.subplot(len(cutoff_hz_list), len(width_hz_list), i)
            plt.plot(t, signal, label='signal')
            # Plot the filtered signal, shifted to compensate for the phase delay.
            plt.plot(t-delay, filtered_signal, 'r-', label = 'low pass')
            # Plot just the "good" part of the filtered signal.  The first N-1
            # samples are "corrupted" by the initial conditions.
            plt.xlabel('days')
            plt.title('Cutoff = ' + str(cutoff_hz) + 'Hz, Width = ' + str(width_hz) + 'Hz', fontsize=10)
            plt.grid()
            plt.tight_layout()

    return filtered_signal[int(delay*24):]

def butter_lp_filter(cutoff_hz_list, order_list, signal, signal_name, plot=True):
    i = 0
    if plot:
        plt.figure(figsize=(len(order_list) * 5, len(cutoff_hz_list) * 2))
        plt.suptitle("Low Pass IIR vs Unfiltered Signal (" + signal_name + ")")
    for fc in cutoff_hz_list:
        for order in order_list:
            i+=1
            fs = 24
            t = np.arange(len(signal)) / fs
            wc = fc/(fs/2) # normalise relative to nyquist frequency

            # create filter and run
            b, a = butter(order, wc, 'low')
            filtered_signal = filtfilt(b,a,signal)

            if plot:
                plt.subplot(len(cutoff_hz_list), len(order_list), i)
                plt.plot(t, signal, label='signal')
                plt.plot(t, filtered_signal, 'r-',  label='low_pass')

                plt.xlabel('days')
                plt.title('Cutoff = ' + str(fc) + 'Hz, Order = ' + str(order), fontsize=10)
                plt.grid()
                plt.tight_layout()

    return filtered_signal