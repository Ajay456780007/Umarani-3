import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import wfdb

from Feature_Extraction.HRV_Features import HRV_Features
from Feature_Extraction.Stastical_Features import Stastical_features
from Sub_Functions.Read_data2 import bandpass_filter
from preprocessing.PQRST_peak import find_pqrst_peaks, split_peaks
from preprocessing.LMS_filter import LMSFilter, filter_signal_lms


def cut_and_save_segments(
        signal, x_axis, peak_indices,
        sample_idx, lead_idx, peak_name,
        base_dir="Image_Results/Segments"
):
    if len(peak_indices) < 3:
        return  # need prev, current, next

    save_dir = f"{base_dir}/{peak_name}/Sample{sample_idx}/Lead{lead_idx}"
    os.makedirs(save_dir, exist_ok=True)

    for k in range(1, len(peak_indices) - 1):
        prev_peak = peak_indices[k - 1]
        curr_peak = peak_indices[k]
        next_peak = peak_indices[k + 1]

        start = (prev_peak + curr_peak) // 2
        end = (curr_peak + next_peak) // 2

        segment_signal = signal[start:end]
        segment_time = x_axis[start:end]

        if len(segment_signal) == 0:
            continue

        plt.figure(figsize=(6, 3))
        plt.plot(segment_time, segment_signal, color='black')

        # mark peak in the MIDDLE of the segment
        plt.scatter(
            x_axis[curr_peak],
            signal[curr_peak],
            label=peak_name,
            zorder=5
        )

        # plt.axvline(x_axis[curr_peak], color='red', linestyle='--', alpha=0.5)

        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        # plt.title(f"{peak_name} centered segment")
        plt.grid(alpha=0.3)

        plt.savefig(
            f"{save_dir}/{peak_name}_segment_{k}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()


def plot_signal_with_pqrst(x, signal, p, q, r, s, t, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(x, signal, color='black')

    # Mark peaks
    if len(p) > 0:
        plt.scatter(x[p], signal[p], color='blue', label='P', zorder=5)
    if len(q) > 0:
        plt.scatter(x[q], signal[q], color='purple', label='Q', zorder=5)
    if len(r) > 0:
        plt.scatter(x[r], signal[r], color='red', label='R', zorder=5)
    if len(s) > 0:
        plt.scatter(x[s], signal[s], color='orange', label='S', zorder=5)
    if len(t) > 0:
        plt.scatter(x[t], signal[t], color='green', label='T', zorder=5)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(alpha=0.3)
    # plt.ylim([-0.2, 0.8])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


data = pd.read_csv(
    "Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv")

base_path = "Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

signal_paths = data["filename_hr"].values[5:]

complete_signal_paths = []

for path in signal_paths:
    # splitted_path = path.split(".")
    # sp = splitted_path.pop[-1]
    # path = "".join(sp)
    new_path = os.path.join(base_path, path)

    complete_signal_paths.append(new_path)

for index, complete_path in enumerate(complete_signal_paths):
    new_record = wfdb.rdrecord(complete_path)

    signal = new_record.p_signal
    fs = new_record.fs

    x_axis = np.arange(0, len(signal) / fs, 1 / fs)

    for lead in range(12):
        bandpass_filter1 = bandpass_filter(signal[:, lead], fs)
        LMS_filter1 = filter_signal_lms(bandpass_filter1, fs)

        p_peaks, q_peaks, r_peaks, s_peaks, t_peaks = find_pqrst_peaks(LMS_filter1)

        peak_groups = [
            (p_peaks, "P"),
            (q_peaks, "Q"),
            (r_peaks, "R"),
            (s_peaks, "S"),
            (t_peaks, "T")
        ]

        for peak_list, peak_name in peak_groups:
            cut_and_save_segments(
                LMS_filter1,
                x_axis,
                peak_list,
                sample_idx=index,
                lead_idx=lead + 1,
                peak_name=peak_name
            )

        # plt.figure(figsize=(12, 6))
        # plt.plot(x_axis, LMS_filter1, label="LMS Filtered Signal", alpha=0.4)
        # plt.plot(x_axis, bandpass_filter1, label='Bandpass Filtered Signal', linewidth=2)
        # plt.xlabel("Time (s)")
        # plt.ylabel("Amplitude")
        # plt.legend()
        # os.makedirs(f"Image_Results/LMS_filter/Sample{index}/", exist_ok=True)
        # plt.savefig(f"Image_Results/LMS_filter/Sample{index}/Lead{i + 1}.png")
        # plt.close()

        # plt.figure(figsize=(12,6))
        # plt.plot(x_axis,signal[:,i],alpha=0.4)
        # plt.xlabel("Time (s)")
        # plt.ylabel("Amplitude")
        # os.makedirs(f"Image_Results/Original_signal/Sample{index}/", exist_ok=True)
        # plt.savefig(f"Image_Results/Original_signal/Sample{index}/Lead{i + 1}.png")
        # plt.close()
        #
        # plt.figure(figsize=(12, 6))
        # plt.plot(x_axis, signal[:, i], label='Original Signal', alpha=0.7)
        # plt.plot(x_axis, bandpass_filter1, label='Bandpass Filtered Signal', linewidth=2)
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.legend()
        # # plt.grid(True)
        # os.makedirs(f"Image_Results/Bandpass_filter/Sample{index}/", exist_ok=True)
        # plt.savefig(f"Image_Results/Bandpass_filter/Sample{index}/Lead{i + 1}.png")
        # plt.close()
        # plt.show()
