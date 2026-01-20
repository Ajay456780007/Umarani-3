# # import numpy as np
# # import matplotlib.pyplot as plt
# # import neurokit2 as nk
# # import wfdb
# # # ---------------------------------------------------------
# # # 1. LOAD OR PROVIDE ECG SIGNAL
# # # ---------------------------------------------------------
# # # Example: replace this with your PTB ECG signal (1 lead)
# # # ecg_signal shape: (5000,) or any length
# # fs = 500  # Sampling frequency (Hz)
# #
# # # Dummy signal for testing (REMOVE this when using real data)
# # # ecg_signal = nk.ecg_simulate(duration=10, sampling_rate=fs, heart_rate=70)
# # path ="Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/02000/02008_hr"
# # signal = wfdb.rdrecord(path)
# #
# # ecg_signal =signal.p_signal[:,11]
# # # ---------------------------------------------------------
# # # 2. R-PEAK DETECTION (ROBUST METHOD)
# # # ---------------------------------------------------------
# # signals, info = nk.ecg_process(ecg_signal, sampling_rate=fs)
# # r_peaks = info["ECG_R_Peaks"]
# #
# # # ---------------------------------------------------------
# # # 3. PQRST EXTRACTION FUNCTION (YOUR METHOD)
# # # ---------------------------------------------------------
# # def extract_pqrst(signal, r_peaks, fs=500):
# #     p, q, s, t = [], [], [], []
# #
# #     for r in r_peaks:
# #         # Boundary protection
# #         if r - int(0.2 * fs) < 0 or r + int(0.4 * fs) >= len(signal):
# #             continue
# #
# #         # Windows based on ECG physiology
# #         p_win = signal[r - int(0.2 * fs): r - int(0.1 * fs)]
# #         q_win = signal[r - int(0.05 * fs): r]
# #         s_win = signal[r: r + int(0.06 * fs)]
# #         t_win = signal[r + int(0.15 * fs): r + int(0.4 * fs)]
# #
# #         # Peak detection
# #         p.append(np.argmax(p_win) + r - int(0.2 * fs))
# #         q.append(np.argmin(q_win) + r - int(0.05 * fs))
# #         s.append(np.argmin(s_win) + r)
# #         t.append(np.argmax(t_win) + r + int(0.15 * fs))
# #
# #     return np.array(p), np.array(q), np.array(r_peaks), np.array(s), np.array(t)
# #
# # # ---------------------------------------------------------
# # # 4. EXTRACT PQRST
# # # ---------------------------------------------------------
# # p, q, r, s, t = extract_pqrst(ecg_signal, r_peaks, fs)
# #
# # # ---------------------------------------------------------
# # # 5. VISUALIZATION FUNCTION
# # # ---------------------------------------------------------
# # def plot_pqrst(signal, p, q, r, s, t):
# #     plt.figure(figsize=(12, 5))
# #     plt.plot(signal, color="black", linewidth=1, label="ECG")
# #
# #     plt.scatter(p, signal[p], color="blue", s=40, label="P")
# #     plt.scatter(q, signal[q], color="green", s=40, label="Q")
# #     plt.scatter(r, signal[r], color="red", s=60, label="R")
# #     plt.scatter(s, signal[s], color="purple", s=40, label="S")
# #     plt.scatter(t, signal[t], color="orange", s=40, label="T")
# #
# #     plt.title("ECG Signal with P-Q-R-S-T Detection")
# #     plt.xlabel("Samples")
# #     plt.ylabel("Amplitude")
# #     plt.legend()
# #     plt.grid(True)
# #     plt.tight_layout()
# #     plt.show()
# #
# # # ---------------------------------------------------------
# # # 6. PLOT RESULT
# # # ---------------------------------------------------------
# # plot_pqrst(ecg_signal, p, q, r, s, t)
# #
# # # here the cut the signal
#
#
# import numpy as np
# import neurokit2 as nk
# import wfdb
# import matplotlib.pyplot as plt
#
# # ---------------------------------------------------------
# # Utility: clean peak indices (handles None & NaN)
# # ---------------------------------------------------------
# def clean_peaks(peaks):
#     clean = []
#     for p in peaks:
#         if p is None:
#             continue
#         if isinstance(p, float) and np.isnan(p):
#             continue
#         clean.append(int(p))
#     return np.array(clean, dtype=int)
#
# # ---------------------------------------------------------
# # 1. LOAD RAW ECG SIGNAL (NO FILTERING)
# # ---------------------------------------------------------
# fs = 500
#
# path = (
#     "Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
#     "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
#     "records500/02000/02000_hr"
# )
#
# record = wfdb.rdrecord(path)
#
# # Choose ONE lead (example: lead index 9)
# ecg_raw = record.p_signal[:, 0]   # <-- ORIGINAL RAW ECG
#
# plt.plot(np.arange(len(ecg_raw)),ecg_raw,color="black")
# plt.show()
# # ---------------------------------------------------------
# # 2. CLEAN ONLY FOR DETECTION (NOT FOR PLOTTING)
# # ---------------------------------------------------------
# ecg_cleaned = nk.ecg_clean(ecg_raw, sampling_rate=fs)
#
# # ---------------------------------------------------------
# # 3. R-PEAK DETECTION
# # ---------------------------------------------------------
# _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
# r_peaks = rpeaks["ECG_R_Peaks"].astype(int)
#
# # ---------------------------------------------------------
# # 4. PQRST DELINEATION (WAVELET METHOD)
# # ---------------------------------------------------------
# _, waves = nk.ecg_delineate(
#     ecg_cleaned,
#     rpeaks,
#     sampling_rate=fs,
#     method="dwt"
# )
#
# # ---------------------------------------------------------
# # 5. EXTRACT P Q R S T PEAKS
# # ---------------------------------------------------------
# p_peaks = clean_peaks(waves["ECG_P_Peaks"])
# q_peaks = clean_peaks(waves["ECG_Q_Peaks"])
# s_peaks = clean_peaks(waves["ECG_S_Peaks"])
# t_peaks = clean_peaks(waves["ECG_T_Peaks"])
#
# # ---------------------------------------------------------
# # 6. PLOT RAW ECG + PQRST (MATCHES YOUR FIRST IMAGE)
# # ---------------------------------------------------------
# plt.figure(figsize=(16, 5))
#
# # RAW ECG waveform (NO filtering)
# plt.plot(ecg_raw, color="black", linewidth=1, label="Raw ECG")
#
# # Overlay PQRST peaks ON RAW SIGNAL
# plt.scatter(p_peaks, ecg_raw[p_peaks], color="blue",   s=40, label="P")
# plt.scatter(q_peaks, ecg_raw[q_peaks], color="green",  s=40, label="Q")
# plt.scatter(r_peaks, ecg_raw[r_peaks], color="red",    s=60, label="R")
# plt.scatter(s_peaks, ecg_raw[s_peaks], color="purple", s=40, label="S")
# plt.scatter(t_peaks, ecg_raw[t_peaks], color="orange", s=40, label="T")
#
# plt.xlabel("Samples")
# plt.ylabel("Voltage (mV)")
# plt.title("PQRST Peaks Overlaid on RAW ECG Signal")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

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

    record = wfdb.rdrecord(os.path.join(base_path, rel_path))
    signal = record.p_signal     # (N, 12)
    fs = record.fs

    # -----------------------------------------------------
    # PROCESS EACH LEAD
    # -----------------------------------------------------
    for lead in range(signal.shape[1]):

        ecg_raw = signal[:, lead]
        ecg_cleaned = nk.ecg_clean(ecg_raw, sampling_rate=fs)

        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
        r_peaks = rpeaks["ECG_R_Peaks"].astype(int)

        if len(r_peaks) < 2:
            continue

        _, waves = nk.ecg_delineate(
            ecg_cleaned, rpeaks, sampling_rate=fs, method="dwt"
        )

        p = clean_peaks(waves["ECG_P_Peaks"])
        q = clean_peaks(waves["ECG_Q_Peaks"])
        s = clean_peaks(waves["ECG_S_Peaks"])
        t = clean_peaks(waves["ECG_T_Peaks"])
        r = r_peaks

        peak_groups = {
            "P": p,
            "Q": q,
            "R": r,
            "S": s,
            "T": t
        }

        # -------------------------------------------------
        # CUT & SAVE SEGMENTS WITH TIME (ms)
        # -------------------------------------------------
        for wave_name, peaks in peak_groups.items():

            if len(peaks) < 2:
                continue

            for i in range(len(peaks) - 1):

                start = peaks[i]
                end = peaks[i + 1]

                if end <= start:
                    continue

                segment = ecg_raw[start:end]
                time_ms = np.arange(start, end) * (1000 / fs)

                save_dir = (
                    f"Image_Results/Splitted/Sample{sample_idx}/"
                    f"{wave_name}/Lead{lead + 1}"
                )
                os.makedirs(save_dir, exist_ok=True)

                plt.figure(figsize=(4, 2))
                plt.plot(time_ms, segment, color="black", linewidth=1)
                plt.xlabel("Time (ms)")
                plt.ylabel("Voltage (mV)")
                # plt.title(f"{wave_name} | Lead {lead+1}")
                plt.grid(True)
                plt.tight_layout()

                plt.savefig(
                    os.path.join(save_dir, f"{wave_name}_{i}.png"),
                    dpi=300
                )
                plt.close()

    print(f"Finished Sample {sample_idx}")
