import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import neurokit2 as nk

# ---------------------------------------------------------
# Utility: clean peak indices (handles None & NaN)
# ---------------------------------------------------------
def clean_peaks(peaks):
    clean = []
    for p in peaks:
        if p is None:
            continue
        if isinstance(p, float) and np.isnan(p):
            continue
        clean.append(int(p))
    return np.array(clean, dtype=int)

# ---------------------------------------------------------
# LOAD METADATA
# ---------------------------------------------------------
data = pd.read_csv(
    "Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
    "ptbxl_database.csv"
)

base_path = (
    "Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

signal_paths = data["filename_hr"].values[:5]

# ---------------------------------------------------------
# PROCESS EACH SAMPLE
# ---------------------------------------------------------
for sample_idx, rel_path in enumerate(signal_paths):

    record_path = os.path.join(base_path, rel_path)
    record = wfdb.rdrecord(record_path)

    signal = record.p_signal          # shape: (N, 12)
    fs = record.fs

    sample_dir = f"Image_Results/PQRST/Sample{sample_idx}"
    os.makedirs(sample_dir, exist_ok=True)

    # -----------------------------------------------------
    # PROCESS EACH LEAD
    # -----------------------------------------------------
    for lead in range(signal.shape[1]):

        # RAW ECG (for plotting)
        ecg_raw = signal[:, lead]

        # CLEANED ECG (for detection only)
        ecg_cleaned = nk.ecg_clean(ecg_raw, sampling_rate=fs)

        # ---------------- R-PEAK DETECTION ----------------
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
        r_peaks = rpeaks["ECG_R_Peaks"].astype(int)

        if len(r_peaks) < 2:
            continue

        # ---------------- PQRST DELINEATION ----------------
        _, waves = nk.ecg_delineate(
            ecg_cleaned,
            rpeaks,
            sampling_rate=fs,
            method="dwt"
        )

        p_peaks = clean_peaks(waves["ECG_P_Peaks"])
        q_peaks = clean_peaks(waves["ECG_Q_Peaks"])
        s_peaks = clean_peaks(waves["ECG_S_Peaks"])
        t_peaks = clean_peaks(waves["ECG_T_Peaks"])

        # ---------------- PLOTTING ON RAW ECG ----------------
        plt.figure(figsize=(16, 5))
        plt.plot(ecg_raw, color="black", linewidth=1, label="Raw ECG")

        if len(p_peaks) > 0:
            plt.scatter(p_peaks, ecg_raw[p_peaks], color="blue", s=40, label="P")
        if len(q_peaks) > 0:
            plt.scatter(q_peaks, ecg_raw[q_peaks], color="green", s=40, label="Q")
        if len(r_peaks) > 0:
            plt.scatter(r_peaks, ecg_raw[r_peaks], color="red", s=60, label="R")
        if len(s_peaks) > 0:
            plt.scatter(s_peaks, ecg_raw[s_peaks], color="purple", s=40, label="S")
        if len(t_peaks) > 0:
            plt.scatter(t_peaks, ecg_raw[t_peaks], color="orange", s=40, label="T")

        plt.xlabel("Samples")
        plt.ylabel("Voltage (mV)")
        # plt.title(f"Sample {sample_idx} – Lead {lead + 1} (RAW ECG with PQRST)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(sample_dir, f"Lead{lead + 1}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    print(f"Saved PQRST plots for Sample {sample_idx}")
# along with this also cut the signals and extract the correct Time ms in x_axis and save the images indside
# Image_Results/Splitted/sample0/P/lead1,lead2,,Q,R,ST     for lead in range(len(signal)):
#         lead_signal = signal[lead]
#
#         # iterate over peak types
#         for peak_list, output in peak_groups:
#             lead_peaks = peak_list[lead]
#
#             for i in range(len(lead_peaks) - 1):
#                 current_peak = (lead_peaks[i] + lead_peaks[i]) // 2
#                 next_peak = (lead_peaks[i] + lead_peaks[i + 1]) // 2
#
#                 segment = lead_signal[current_peak + 1: next_peak]  -- this is the cutting logic , give me teh complete corrected code