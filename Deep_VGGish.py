# import numpy as np
# import torch
# import torchaudio
# from torchvision.models import resnet101
# import wfdb
# from scipy import signal
# from scipy.ndimage import zoom
# import matplotlib.pyplot as plt
#
# # 1. Load pretrained ResNet101 (stage 1)
# resnet = resnet101(pretrained=True)
# resnet.fc = torch.nn.Identity()  # Remove classifier for features
# resnet.eval()
#
# # 2. Load VGGish (stage 2)
# from torchvggish import vggish, vggish_input
#
# vggish_model = vggish()
# vggish_model.eval()
#
#
# def resnet_ecg_preprocess(ecg_signal, fs=500):
#     """ECG (5000,12) -> ResNet image (224,224,3)"""
#     # Average 12 leads
#     ecg_mono = np.mean(ecg_signal, axis=1)
#
#     # Spectrogram -> image
#     f, t, Sxx = signal.spectrogram(ecg_mono, fs=fs, nperseg=256)
#     Sxx = np.log(Sxx + 1e-10)
#
#     # Resize to 224x224x1
#     target_size = (224, 224)
#     Sxx_resized = zoom(Sxx, (224 / Sxx.shape[0], 224 / Sxx.shape[1]))
#
#     # Convert to 3-channel RGB
#     img = np.stack([Sxx_resized] * 3, axis=-1)
#     img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)  # (1,3,224,224)
#
#     # ResNet normalization
#     normalize = torch.nn.Sequential(
#         torch.nn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     )
#     return normalize(img)
#
#
# def Deep_VGGish_features(signal):
#     """Two-stage: ResNet101 -> VGGish"""
#
#     # Stage 1: ResNet101 deep features
#     print("Stage 1: ResNet101...")
#     resnet_input = resnet_ecg_preprocess(signal)
#     with torch.no_grad():
#         resnet_feat = resnet(resnet_input)  # (1, 2048)
#         resnet_feat = resnet_feat.mean(dim=0).numpy()  # (2048,)
#
#     # Stage 2: VGGish deep features
#     print("Stage 2: VGGish...")
#     ecg_mono = np.mean(signal, axis=1).astype(np.float32)
#     log_mel = vggish_input.waveform_to_examples(ecg_mono, 500)  # (T,96,64)
#     log_mel = torch.from_numpy(log_mel).unsqueeze(0)  # (1,T,96,64)
#
#     with torch.no_grad():
#         vggish_feat = vggish_model(log_mel)  # (1,T,128)
#         vggish_feat = vggish_feat.mean(dim=1).squeeze().numpy()  # (128,)
#
#     # Combine: 2048 + 128 = 2176D deep features
#     deep_features = np.concatenate([resnet_feat, vggish_feat])
#     print(f"Final deep features shape: {deep_features.shape}")
#     return deep_features
#
#
# # Install first: pip install torch torchaudio torchvision torchvggish scipy wfdb
# # Test
# signal = wfdb.rdrecord("records500/01000/01000_hr").p_signal
# print(f"ECG shape: {signal.shape}")
#
# out = Deep_VGGish_features(signal)
# print(f"Deep features: {out.shape}")


import numpy as np
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
# import tensorflow_hub as hub
import wfdb
from keras.applications import ResNet101
from keras.models import Model
from torch import hub
from torchvggish import vggish_input, vggish_params, vggish

# res = ResNet101(weights="imagenet", include_top=True)
# ResNet_model = Model(inputs=res.input, outputs=res.layers[2].output)

# model = hub.load('https://tfhub.dev')

model = vggish()
model.eval().to('cpu')


def extract_vggish_features(sig_path, device='cpu'):

    batch_data = vggish_input.waveform_to_examples(sig_path, 512)

    tensor_data = torch.tensor(batch_data).float().to(device)

    with torch.no_grad():
        embeddings = model(tensor_data)

    return embeddings.detach().cpu().numpy().flatten()

# signal = wfdb.rdrecord(
#     "../Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/01000/01000_hr")
#
# final_signal = signal.p_signal
# print(final_signal.shape)
# out = Deep_VGGish_features(final_signal)
# print(out.shape)
