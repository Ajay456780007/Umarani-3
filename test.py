# # # import pandas as pd
# # # import numpy as np
# # # import wfdb
# # # import ast
# # #
# # #
# # # def load_raw_data(df, sampling_rate, path):
# # #     if sampling_rate == 100:
# # #         data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
# # #     else:
# # #         data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
# # #     data = np.array([signal for signal, meta in data])
# # #     return data
# # #
# # #
# # # path = "Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
# # # sampling_rate = 100
# # #
# # # # load and convert annotation data
# # # Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
# # # Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
# # #
# # # # Load raw signal data
# # # X = load_raw_data(Y, sampling_rate, path)
# # #
# # # # Load scp_statements.csv for diagnostic aggregation
# # # agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
# # # agg_df = agg_df[agg_df.diagnostic == 1]
# # #
# # #
# # # def aggregate_diagnostic(y_dic):
# # #     tmp = []
# # #     for key in y_dic.keys():
# # #         if key in agg_df.index:
# # #             tmp.append(agg_df.loc[key].diagnostic_class)
# # #     return list(set(tmp))
# # #
# # #
# # # # Apply diagnostic superclass
# # # Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
# # #
# # # # Split data into train and test
# # # test_fold = 10
# # # # Train
# # # X_train = X[np.where(Y.strat_fold != test_fold)]
# # # y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# # # # Test
# # # X_test = X[np.where(Y.strat_fold == test_fold)]
# # # y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
# #
# # import numpy as np
# # import wfdb
# # import matplotlib.pyplot as plt
# #
# # path = "Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/01000/01004_hr"
# #
# # out = wfdb.rdrecord(path)
# #
# # signal = out.p_signal
# #
# # print(out.fs)
# #
# # print(signal.shape)
# #
# # plt.plot(np.arange(out.fs*10),signal)
# # plt.show()
# #
# #
# import ast
#
# import numpy as np
# import pandas as pd
# full_label=[]
# data = pd.read_csv(
#         "Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv")
# MI_CODES = {"AMI", "IMI", "MI"}
# count =0
# for scp_str in data["scp_codes"]:
#     scp_dict = ast.literal_eval(scp_str)
#     if count<50:
#         if any(code in MI_CODES for code in scp_dict.keys()):
#             full_label.append(1)
#         else:
#             full_label.append(0)
#     count =count+1
#
# np.save(f"data_loader/DB1/Labels.npy",np.array(full_label))
import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(
    "Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv"
)

base_path = "Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

signal_paths = data["filename_hr"].values[:5]

save_dir = "ECG_12Lead_Images"
os.makedirs(save_dir, exist_ok=True)

for index, path in enumerate(signal_paths):

    complete_path = os.path.join(base_path, path)
    record = wfdb.rdrecord(complete_path)

    signal = record.p_signal        # shape: (samples, 12)
    fs = record.fs

    time = np.arange(signal.shape[0]) / fs

    lead_names = record.sig_name    # ['I', 'II', 'III', ... ]

    fig, axes = plt.subplots(12, 1, figsize=(15, 20), sharex=True)

    for lead in range(12):
        axes[lead].plot(time, signal[:, lead])
        axes[lead].set_ylabel(lead_names[lead])
        axes[lead].grid(True)

    axes[-1].set_xlabel("Time (seconds)")

    plt.suptitle(f"12-Lead ECG Record {index}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(save_dir, f"ECG_Record_{index}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")
