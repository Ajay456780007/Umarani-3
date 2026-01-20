import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import os
from scipy.signal import butter, filtfilt
from preprocessing.LMS_filter import filter_signal_lms
from preprocessing.PQRST_peak import find_pqrst_peaks, split_peaks
from Feature_Extraction.Stastical_Features import Stastical_features
from Feature_Extraction.HRV_Features import HRV_Features
from Feature_Extraction.Time_Domain_Features import extract_time_domain_features
from Feature_Extraction.Haar_Wavelet import Haar_wavelet_features
from Feature_Extraction.Deep_VGGish import extract_vggish_features
import ast


def bandpass_filter(sig, fs, low=0.5, high=40):
    b, a = butter(4, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, sig, axis=0)


def Read_data(DB):
    data = pd.read_csv(
        "Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv")

    base_path = "Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

    signal_paths = data["filename_hr"].values[:50]

    complete_signal_paths = []

    for path in signal_paths:
        # splitted_path = path.split(".")
        # sp = splitted_path.pop[-1]
        # path = "".join(sp)
        new_path = os.path.join(base_path, path)

        complete_signal_paths.append(new_path)
    full_features = []
    full_label = []
    for index2, complete_path in enumerate(complete_signal_paths):
        print("Processing:", index2)
        new_record = wfdb.rdrecord(complete_path)

        signal = new_record.p_signal

        fs = new_record.fs

        bandpass_filter1 = bandpass_filter(signal, fs)

        out_LMS = []
        for leads in range(bandpass_filter1.shape[1]):
            LMS_filter1 = filter_signal_lms(bandpass_filter1[:, leads], fs)
            out_LMS.append(LMS_filter1)

        LMS_out = np.array(out_LMS)

        normalized_out = (LMS_out - LMS_out.mean(axis=0)) / LMS_out.std(axis=0)
        final_r_peak, final_q_peak, final_p_peak, final_s_peak, final_t_peak = [], [], [], [], []
        for leads1 in range(normalized_out.shape[0]):
            p_peaks, q_peaks, r_peaks, s_peaks, t_peaks = find_pqrst_peaks(normalized_out[leads1, :])
            final_r_peak.append(r_peaks)
            final_p_peak.append(p_peaks)
            final_q_peak.append(q_peaks)
            final_s_peak.append(s_peaks)
            final_t_peak.append(t_peaks)

        # out11 =[final_p_peak,final_q_peak,final_r_peak,final_s_peak,final_t_peak]
        final_r_peak_splitted, final_q_peak_splitted, final_p_peak_splitted, final_s_peak_splitted, final_t_peak_splitted = [], [], [], [], []

        r_peak_splitted, p_peak_splitted, s_peak_splitted, q_peak_splitted, t_peak_splitted = split_peaks(
            normalized_out, final_r_peak, final_p_peak, final_s_peak, final_q_peak, final_t_peak)

        peak_splits = [r_peak_splitted, p_peak_splitted, s_peak_splitted, q_peak_splitted, t_peak_splitted]

        hrv_features = []
        haar_wavelet_feat = []
        Time_domain_feat = []
        statistical_feat = []
        Deep_VGGish_feat = []
        final_labels = []

        max_len1 = max(len(m) for m in final_p_peak)
        max_len2 = max(len(m) for m in final_q_peak)
        max_len3 = max(len(m) for m in final_r_peak)
        max_len4 = max(len(m) for m in final_s_peak)
        max_len5 = max(len(m) for m in final_t_peak)

        max_len = max(max_len1, max_len2, max_len3, max_len4, max_len5)
        max_len = 1210
        Deep_Vggish = extract_vggish_features(signal)
        for index11, splits in enumerate(peak_splits):
            temp_hrv_features = []
            temp_haar_wavelet_feat = []
            temp_time_domain_feat = []
            temp_statistical_feat = []
            temp_deep_vggish_feat = []

            for s in splits:
                if len(s) > 0:
                    hrv = HRV_Features(s)
                    padded_signal = s[:max_len] + [0] * (max_len - len(s))
                    haar = Haar_wavelet_features(padded_signal)
                    tdf = extract_time_domain_features(s)
                    stats = Stastical_features(s)

                    temp_haar_wavelet_feat.append(haar)
                    temp_time_domain_feat.append(tdf)
                    temp_statistical_feat.append(stats)
                    # temp_deep_vggish_feat.append(Deep_Vggish)
                    temp_hrv_features.append(hrv)

            haar_wavelet_feat.append(np.average(temp_haar_wavelet_feat, axis=0))
            hrv_features.append(np.average(temp_hrv_features, axis=0))
            Time_domain_feat.append(np.average(temp_time_domain_feat, axis=0))
            statistical_feat.append(np.average(temp_statistical_feat, axis=0))
            Deep_VGGish_feat.append(np.array(Deep_Vggish))

        haar_wavelet_feat = np.array(haar_wavelet_feat)
        hrv_features = np.array(hrv_features)
        Time_domain_feat = np.array(Time_domain_feat)
        statistical_feat = np.array(statistical_feat)

        final_hrv_feat = hrv_features.reshape(-1)
        final_haar_features = haar_wavelet_feat.reshape(-1)
        final_tdf_features = Time_domain_feat.reshape(-1)
        final_stats_features = statistical_feat.reshape(-1)
        # final_deep_vggish_features = np.average(np.array(Deep_VGGish_feat), axis=0)
        final_deep_vigish_features = np.array(np.average(Deep_VGGish_feat, axis=0))

        final_concated_features = np.concatenate(
            [final_hrv_feat, final_haar_features, final_tdf_features, final_stats_features, final_deep_vigish_features],
            axis=0)

        full_features.append(final_concated_features)
        import ast

        MI_CODES = {"AMI", "IMI", "MI"}
        count = 0
        for scp_str in data["scp_codes"]:
            scp_dict = ast.literal_eval(scp_str)
            if count < 50:
                if any(code in MI_CODES for code in scp_dict.keys()):
                    full_label.append(1)
                else:
                    full_label.append(0)
            count = count + 1

    os.makedirs(f"data_loader/{DB}/", exist_ok=True)
    np.save(f"data_loader/{DB}/Features.npy", np.array(full_features))
    np.save(f"data_loader/{DB}/Labels.npy", np.array(full_label))
