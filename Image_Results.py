import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
from Feature_Extraction.HRV_Features import HRV_Features
from Feature_Extraction.Stastical_Features import Stastical_features
from Sub_Functions.Read_data import bandpass_filter
from preprocessing.PQRST_peak import find_pqrst_peaks, split_peaks
from preprocessing.LMS_filter import LMSFilter, filter_signal_lms

def plot_12_leads_original(signal, fs, save_path):
    """
    signal: shape (N, 12)
    fs: sampling frequency
    save_path: path to save the image
    """
    num_leads = signal.shape[1]
    x_axis = np.arange(signal.shape[0]) / fs  # Time in seconds

    fig, axes = plt.subplots(6, 2, figsize=(16, 18), sharex=True)
    axes = axes.flatten()

    for lead in range(num_leads):
        axes[lead].plot(x_axis, signal[:, lead], color='black', linewidth=1)
        axes[lead].set_title(f"Lead {lead + 1}")
        axes[lead].set_ylabel("mV")
        axes[lead].grid(alpha=0.3)

    for ax in axes[-2:]:
        ax.set_xlabel("Time (s)")

    # fig.suptitle("12-Lead Original ECG Signal", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_signal_with_pqrst(x, signal, p, q, r, s, t, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(x, signal, label='Filtered ECG', color='black')

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
    plt.ylabel("Amplitude (mV) ")
    plt.legend()
    plt.grid(alpha=0.3)
    # plt.ylim([-0.2, 0.8])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


data = pd.read_csv(
    "Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv")

base_path = "Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

signal_paths = data["filename_hr"].values[:5]

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

    os.makedirs(f"Image_Results/Original_signal/Sample{index}", exist_ok=True)

    save_12lead_path = (
        f"Image_Results/Original_signal/Sample{index}/12lead.png"
    )

    plot_12_leads_original(signal, fs, save_12lead_path)

    # x_axis = np.arange(len(signal))
    #
    # plt.figure(figsize=(12, 6))
    # plt.plot(x_axis, signal, alpha=0.4)
    # plt.xlabel("Time (ms)")
    # plt.ylabel("Voltage (mV)")
    # os.makedirs(f"Image_Results/Original_signal/Sample{index}/", exist_ok=True)
    # plt.savefig(f"Image_Results/Original_signal/Sample{index}/sample1.png")
    # plt.close()
    #
    # for i in range(12):
    #     bandpass_filter1 = bandpass_filter(signal[:, i], fs)
    #     LMS_filter1 = filter_signal_lms(bandpass_filter1, fs)
    #
    #     p_peaks, q_peaks, r_peaks, s_peaks, t_peaks = find_pqrst_peaks(LMS_filter1)
    #
    #     save_dir = f"Image_Results/PQRST/Sample{index}"
    #     os.makedirs(save_dir, exist_ok=True)
    #
    #     save_path = f"{save_dir}/Lead{i + 1}.png"
    #
    #     plot_signal_with_pqrst(
    #         x_axis,
    #         LMS_filter1,
    #         p_peaks,
    #         q_peaks,
    #         r_peaks,
    #         s_peaks,
    #         t_peaks,
    #         save_path
    #     )
    #
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(x_axis, LMS_filter1, label="LMS Filtered Signal", alpha=0.4)
    #     plt.plot(x_axis, bandpass_filter1, label='Bandpass Filtered Signal', linewidth=2)
    #     plt.xlabel("Time (ms)")
    #     plt.ylabel("Voltage(mV)")
    #     plt.legend()
    #     os.makedirs(f"Image_Results/LMS_filter/Sample{index}/", exist_ok=True)
    #     plt.savefig(f"Image_Results/LMS_filter/Sample{index}/Lead{i + 1}.png")
    #     plt.close()
    #
    #     # plt.figure(figsize=(12, 6))
    #     # plt.plot(x_axis, signal[:, i], alpha=0.4)
    #     # plt.xlabel("Time (ms)")
    #     # plt.ylabel("Voltage (mV)")
    #     # os.makedirs(f"Image_Results/Original_signal/Sample{index}/", exist_ok=True)
    #     # plt.savefig(f"Image_Results/Original_signal/Sample{index}/Lead{i + 1}.png")
    #     # plt.close()
    #     #
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(x_axis, signal[:, i], label='Original Signal', alpha=0.7)
    #     plt.plot(x_axis, bandpass_filter1, label='Bandpass Filtered Signal', linewidth=2)
    #     plt.xlabel('Time (ms)')
    #     plt.ylabel('Voltage (mV)')
    #     plt.legend()
    #     # plt.grid(True)
    #     os.makedirs(f"Image_Results/Bandpass_filter/Sample{index}/", exist_ok=True)
    #     plt.savefig(f"Image_Results/Bandpass_filter/Sample{index}/Lead{i + 1}.png")
    #     plt.close()
    #     plt.show()
