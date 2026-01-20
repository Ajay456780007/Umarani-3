# import matplotlib.pyplot as plt
# import neurokit2 as nk
# import numpy as np
# import wfdb
#
# file_path = "../Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/00000/00001_hr"
#
# signal = wfdb.rdrecord(file_path)
# for i in range(signal.p_signal.shape[1]):
#     print(i)
#     ecg_signal = signal.p_signal[:, i]
#     fs = signal.fs
#     x_axis = np.arange(ecg_signal.shape[0]) / fs
#
#     plt.plot(x_axis, ecg_signal)
#
#     plt.pause(0.1)
#     plt.close()
from typing import List

import numpy as np
import wfdb
# from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
import neurokit2 as nk


def HRV_Features(nn_intervals: List[float], pnni_as_percent: bool = True):
    """
    Notes
    -----
    Here are some details about feature engineering...
    - **mean_nni**: The mean of RR-intervals.
    - **sdnn** : The standard deviation of the time interval between successive normal heart beats \
    (i.e. the RR-intervals).
    - **sdsd**: The standard deviation of differences between adjacent RR-intervals
    - **rmssd**: The square root of the mean of the sum of the squares of differences between \
    adjacent NN-intervals. Reflects high frequency (fast or parasympathetic) influences on hrV \
    (*i.e.*, those influencing larger changes from one beat to the next).
    - **median_nni**: Median Absolute values of the successive differences between the RR-intervals.
    - **nni_50**: Number of interval differences of successive RR-intervals greater than 50 ms.
    - **pnni_50**: The proportion derived by dividing nni_50 (The number of interval differences \
    of successive RR-intervals greater than 50 ms) by the total number of RR-intervals.
    - **nni_20**: Number of interval differences of successive RR-intervals greater than 20 ms.
    - **pnni_20**: The proportion derived by dividing nni_20 (The number of interval differences \
    of successive RR-intervals greater than 20 ms) by the total number of RR-intervals.
    - **range_nni**: difference between the maximum and minimum nn_interval.
    - **cvsd**: Coefficient of variation of successive differences equal to the rmssd divided by \
    mean_nni.
    - **cvnni**: Coefficient of variation equal to the ratio of sdnn divided by mean_nni.
    - **mean_hr**: The mean Heart Rate.
    - **max_hr**: Max heart rate.
    - **min_hr**: Min heart rate.
    - **std_hr**: Standard deviation of heart rate.
    """
    diff_nni = np.diff(nn_intervals)
    length_int = len(nn_intervals) - 1 if pnni_as_percent else len(nn_intervals)
    # Basic statistics
    mean_nni = np.mean(nn_intervals)
    median_nni = np.median(nn_intervals)
    range_nni = max(nn_intervals) - min(nn_intervals)

    sdsd = np.std(diff_nni)
    rmssd = np.sqrt(np.mean(diff_nni ** 2))

    nni_50 = sum(np.abs(diff_nni) > 50)
    pnni_50 = 100 * nni_50 / length_int
    nni_20 = sum(np.abs(diff_nni) > 20)
    pnni_20 = 100 * nni_20 / length_int

    # Feature found on GitHub and not in documentation
    cvsd = rmssd / mean_nni

    # Features only for long term recordings
    sdnn = np.std(nn_intervals, ddof=1)
    cvnni = sdnn / mean_nni

    # Heart Rate equivalent features
    heart_rate_list = np.divide(60000, nn_intervals)
    mean_hr = np.mean(heart_rate_list)
    min_hr = min(heart_rate_list)
    max_hr = max(heart_rate_list)
    std_hr = np.std(heart_rate_list)

    time_domain_features = {
        'mean_nni': mean_nni,
        'sdnn': sdnn,
        'sdsd': sdsd,
        'nni_50': nni_50,
        'pnni_50': pnni_50,
        'nni_20': nni_20,
        'pnni_20': pnni_20,
        'rmssd': rmssd,
        'median_nni': median_nni,
        'range_nni': range_nni,
        'cvsd': cvsd,
        'cvnni': cvnni,
        'mean_hr': mean_hr,
        "max_hr": max_hr,
        "min_hr": min_hr,
        "std_hr": std_hr,
    }
    f1_new = []
    for value in time_domain_features.values():
        array = np.array(value)
        f1_new.append(array)
    arr = np.array(f1_new)
    arr = np.nan_to_num(arr)
    return arr


# signal = wfdb.rdrecord(
#     "../Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/01000/01000_hr")
#
# ecg_signal = signal.p_signal[:, 1]  # Lead II
# fs = signal.fs
#
# signals, info = nk.ecg_process(ecg_signal, sampling_rate=fs)
# r_peaks = info["ECG_R_Peaks"]
#
# nn_intervals = np.diff(r_peaks) / fs * 1000  # ms
#
# out = HRV_Features(nn_intervals)
#
# print(out.shape)