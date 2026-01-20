import numpy as np
from scipy.signal import find_peaks

import neurokit2 as nk


def find_pqrst_peaks(filtered_signal):
    # r_peaks, _ = find_peaks(filtered_signal, distance=50, height=np.max(filtered_signal) * 0.5)
    # if len(r_peaks) > 0:
    #     r_peak = r_peaks[0]
    #     p_peaks, _ = find_peaks(filtered_signal[:r_peak], distance=50,
    #                             height=np.mean(filtered_signal) + np.std(filtered_signal))
    #     s_peaks, _ = find_peaks(-filtered_signal[r_peak:], distance=50,
    #                             height=-np.mean(filtered_signal) - np.std(filtered_signal))
    #     q_peaks, _ = find_peaks(-filtered_signal[:r_peak], distance=50,
    #                             height=-np.mean(filtered_signal) - np.std(filtered_signal))
    #     t_peaks, _ = find_peaks(filtered_signal[r_peak:], distance=50,
    #                             height=np.mean(filtered_signal) + np.std(filtered_signal))
    # else:
    #     # p_peaks, q_peaks, s_peaks, t_peaks = [], [], [], []
    cleaned_ecg = nk.ecg_clean(filtered_signal, sampling_rate=500, method="pantompkins1985")
    _, rpeaks = nk.ecg_peaks(cleaned_ecg, sampling_rate=500, method="pantompkins1985")
    signals, waves = nk.ecg_delineate(cleaned_ecg, rpeaks, sampling_rate=500, method="dwt")
    p_peaks = waves["ECG_P_Peaks"]
    q_peaks = waves["ECG_Q_Peaks"]
    s_peaks = waves["ECG_S_Peaks"]
    t_peaks = waves["ECG_T_Peaks"]
    r_peaks = rpeaks["ECG_R_Peaks"]

    p_peaks = [m for m in p_peaks if str(m) != "nan"]
    q_peaks = [m for m in q_peaks if str(m) != "nan"]
    r_peaks = [m for m in r_peaks if str(m) != "nan"]
    s_peaks = [m for m in s_peaks if str(m) != "nan"]
    t_peaks = [m for m in t_peaks if str(m) != "nan"]

    return p_peaks, q_peaks, r_peaks, s_peaks, t_peaks


def split_peaks(signal, r_peak_index, p_peak_index, s_peak_index, q_peak_index, t_peak_index):
    r_peak_splitted, p_peak_splitted = [], []
    s_peak_splitted, q_peak_splitted = [], []
    t_peak_splitted = []

    peak_groups = [
        (r_peak_index, r_peak_splitted),
        (p_peak_index, p_peak_splitted),
        (s_peak_index, s_peak_splitted),
        (q_peak_index, q_peak_splitted),
        (t_peak_index, t_peak_splitted)
    ]

    # iterate over leads
    for lead in range(len(signal)):
        lead_signal = signal[lead]

        # iterate over peak types
        for peak_list, output in peak_groups:
            lead_peaks = peak_list[lead]

            for i in range(len(lead_peaks) - 1):
                current_peak = (lead_peaks[i] + lead_peaks[i]) // 2
                next_peak = (lead_peaks[i] + lead_peaks[i + 1]) // 2

                segment = lead_signal[current_peak + 1: next_peak]
                output.append(segment.tolist())

    return r_peak_splitted, p_peak_splitted, s_peak_splitted, q_peak_splitted, t_peak_splitted
