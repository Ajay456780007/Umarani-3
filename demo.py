import neurokit2 as nk
import numpy as np
import wfdb
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load PTB-XL ECG record
# --------------------------------------------------
record = wfdb.rdrecord(
    "Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
    "records500/03000/03000_hr"
)

ecg_signal = record.p_signal[:, 0]
SAMPLING_RATE = record.fs

# --------------------------------------------------
# R-peak detection
# --------------------------------------------------
signals, info = nk.ecg_peaks(ecg_signal, sampling_rate=SAMPLING_RATE)
r_peaks = info["ECG_R_Peaks"].astype(int)

# --------------------------------------------------
# PQRST delineation
# --------------------------------------------------
signal_cwt, waves_peak = nk.ecg_delineate(ecg_signal,
                                         r_peaks,
                                         sampling_rate=SAMPLING_RATE,
                                         method="cwt",
                                         show=True,
                                         show_type='peaks')


# --------------------------------------------------
# Clean peaks safely
# --------------------------------------------------
def clean_peaks(peaks):
    peaks = np.array(peaks, dtype=float)
    peaks = peaks[~np.isnan(peaks)]
    return peaks.astype(int)


p_peaks = clean_peaks(waves_peak["ECG_P_Peaks"])
q_peaks = clean_peaks(waves_peak["ECG_Q_Peaks"])
s_peaks = clean_peaks(waves_peak["ECG_S_Peaks"])
t_peaks = clean_peaks(waves_peak["ECG_T_Peaks"])

# --------------------------------------------------
# Plot ECG with PQRST peaks (first 4 seconds)
# --------------------------------------------------
time = np.arange(len(ecg_signal)) / SAMPLING_RATE

plt.figure(figsize=(14, 5))
plt.plot(time[:4 * SAMPLING_RATE], ecg_signal[:4 * SAMPLING_RATE], label="ECG")

plt.scatter(time[p_peaks], ecg_signal[p_peaks], c="green", label="P", s=30)
plt.scatter(time[q_peaks], ecg_signal[q_peaks], c="cyan", label="Q", s=30)
plt.scatter(time[r_peaks], ecg_signal[r_peaks], c="red", label="R", s=30)
plt.scatter(time[s_peaks], ecg_signal[s_peaks], c="magenta", label="S", s=30)
plt.scatter(time[t_peaks], ecg_signal[t_peaks], c="orange", label="T", s=30)

plt.xlim(0, 4)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.title("Accurate PQRST Extraction using NeuroKit2 (PTB-XL)")
plt.legend()
plt.grid(True)
plt.show()
