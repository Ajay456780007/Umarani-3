import pywt
import numpy as np
import wfdb


def Haar_wavelet_features(signal):
    new_list = []

    cA, cB = pywt.dwt(signal, "haar")

    # new_list.append(cA)

    haar_features = np.array(cA)

    return haar_features

# signal = wfdb.rdrecord(
#     "../Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/01000/01000_hr")
#
# final_signal = signal.p_signal
# print(final_signal.shape)
# out = Haar_wavelet_features(final_signal)
# print(out.shape)
