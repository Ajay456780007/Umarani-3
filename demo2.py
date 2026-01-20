import neurokit2 as nk
import matplotlib.pyplot as plt
import wfdb
import numpy as np

# Load PTB-XL ECG data
path = "Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/02000/02006_hr"
record = wfdb.rdrecord(path)
ecg_signal = record.p_signal[:, 2]  # Lead III (index 2)

# Clean the signal using Pan-Tompkins filtering (5-15Hz)
cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=500, method="pantompkins1985")

# Detect R-peaks using Pan-Tompkins algorithm
_, rpeaks = nk.ecg_peaks(cleaned_ecg, sampling_rate=500, method="pantompkins1985")

# Delineate full PQRST complex using DWT
signals, waves = nk.ecg_delineate(cleaned_ecg, rpeaks, sampling_rate=500, method="dwt")

# **FIXED**: Extract peak locations - NeuroKit2 returns lists, not numpy arrays
p_peaks = np.array(waves["ECG_P_Peaks"])
q_peaks = np.array(waves["ECG_Q_Peaks"])
r_peaks = np.array(rpeaks["ECG_R_Peaks"])
s_peaks = np.array(waves["ECG_S_Peaks"])
t_peaks = np.array(waves["ECG_T_Peaks"])

# Create enhanced scatter plot visualization
fig, ax = plt.subplots(figsize=(12, 6))
time = np.arange(len(cleaned_ecg)) / 500  # Convert samples to seconds

# Plot cleaned ECG signal
ax.plot(time, cleaned_ecg, color='black', linewidth=0.8, alpha=0.7, label='Cleaned ECG')

# Define peaks with colors, markers, and sizes
peaks_data = [
    (p_peaks, 'limegreen', 'o', 40, 'P'),    # P-wave: circle
    (q_peaks, 'orange', 'v', 35, 'Q'),       # Q-wave: triangle down
    (r_peaks, 'red', '^', 60, 'R'),          # R-peak: triangle up, largest
    (s_peaks, 'blue', 's', 35, 'S'),         # S-wave: square
    (t_peaks, 'magenta', 'D', 45, 'T')       # T-wave: diamond
]

# Scatter all peaks with different markers and sizes
for peaks, color, marker, size, label in peaks_data:
    valid_peaks = peaks[~np.isnan(peaks)]  # Remove NaN values
    if len(valid_peaks) > 0:
        ax.scatter(time[valid_peaks.astype(int)], cleaned_ecg[valid_peaks.astype(int)],
                  c=color, marker=marker, s=size,
                  edgecolors='white', linewidth=0.8,
                  label=label, zorder=5, alpha=0.9)

# Customize the plot
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Amplitude (mV)', fontsize=12)
ax.set_title('ECG PQRST Peak Detection (PTB-XL Dataset)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=10)
ax.margins(x=0.01)

plt.xlim(0.50,1.29)
plt.tight_layout()
plt.show()

# Print detection statistics
print(f"R-peaks detected: {len(r_peaks[~np.isnan(r_peaks)])}")
print(f"P-peaks detected: {len(p_peaks[~np.isnan(p_peaks)])}")
print(f"Q-peaks detected: {len(q_peaks[~np.isnan(q_peaks)])}")
print(f"S-peaks detected: {len(s_peaks[~np.isnan(s_peaks)])}")
print(f"T-peaks detected: {len(t_peaks[~np.isnan(t_peaks)])}")


