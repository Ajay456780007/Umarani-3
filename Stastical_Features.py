import numpy as np
import pandas as pd
import wfdb
from scipy.stats import skew, kurtosis


def statistical_features_1lead(signal):
    return np.array([
        np.mean(signal),
        np.std(signal),
        np.var(signal),
        skew(signal),
        kurtosis(signal),
        np.median(signal),
        np.min(signal),
        np.max(signal),
        np.max(signal) - np.min(signal),
        np.sqrt(np.mean(np.square(np.array(signal))))
    ])


def Stastical_features(final_signal):
    features_all_leads = []

    lead_signal = final_signal
    feats = statistical_features_1lead(lead_signal)
    features_all_leads.append(feats)

    features_all_leads = np.array(features_all_leads)

    feat1 = np.average(features_all_leads, axis=0)

    return feat1


# signal = wfdb.rdrecord(
#     "../Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/01000/01000_hr")
#
# final_signal = signal.p_signal
# print(final_signal.shape)
# out = Stastical_features(final_signal)
# print(out.shape)