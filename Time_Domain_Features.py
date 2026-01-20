import numpy as np
import wfdb
from scipy.stats import skew, kurtosis


def extract_time_domain_features(ecg_signal):
    # mean_val = np.mean(ecg_signal)
    # std_val = np.std(ecg_signal)
    # skew_val = skew(ecg_signal)
    variance = np.var(ecg_signal)
    kurt_val = kurtosis(ecg_signal)
    coefficient_of_variation = np.std(ecg_signal) / np.mean(ecg_signal)
    hjorthActivity = variance
    hjorthMobility = np.sqrt(np.var(np.gradient(ecg_signal)) / np.var(ecg_signal))
    line_length = sum([abs(ecg_signal[i] - ecg_signal[i - 1]) for i in range(1, len(ecg_signal))])
    non_linear_energy = sum(
        [(ecg_signal[i]) ** 2 - ecg_signal[i + 1] * ecg_signal[i - 1] for i in range(1, len(ecg_signal) - 1)])

    time_domain_features = np.hstack(
        [variance, kurt_val, coefficient_of_variation, hjorthActivity, hjorthMobility,
         line_length, non_linear_energy])

    # time_domain_features = np.array([mean_val, std_val, skew_val, kurt_val], dtype=object)

    return time_domain_features

# signal = wfdb.rdrecord(
#     "../Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/01000/01000_hr")
#
# final_signal = signal.p_signal
# print(final_signal.shape)
# out = extract_time_domain_features(final_signal)
# print(out)
# print(out.shape)
