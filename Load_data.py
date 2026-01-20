import numpy as np
import pandas as pd
from Comparative_models.Model1 import CNN_1
from Comparative_models.Model2 import CNN_BiLSTM
from Comparative_models.Model3 import CNN_BiLSTM_KNN
from Comparative_models.Model4 import CNN_LSTM_LGBM
from Comparative_models.Model5 import BiLSTM_LSTM_ANN
from Comparative_models.Model6 import CNN_KNN_1
from Comparative_models.Model7 import CNN_KNN_2

def Load_data(DB):
    Feat = np.load(f"data_loader/{DB}/Features.npy")
    Lab = np.load(f"data_loader/{DB}/Labels.npy")

    return Feat, Lab


def Load_data2(DB):
    feat = np.load(f"../data_loader/{DB}/Features.npy")
    labels = np.load(f"../data_loader/{DB}/Labels.npy")

    return feat, labels


def train_test_splitter(data, percent, num=500):
    feat, label = Load_data2(data)  # load your features and labels
    unique_classes = np.unique(label)

    selected_indices_per_class = []

    for cls in unique_classes:
        class_indices = np.where(label == cls)[0]
        # Randomly choose `num` samples per class without replacement
        selected_class_indices = np.random.choice(class_indices, num, replace=True)
        selected_indices_per_class.append(selected_class_indices)

    # Concatenate all selected indices from each class
    selected_indices = np.concatenate(selected_indices_per_class)

    # Shuffle the combined indices
    np.random.shuffle(selected_indices)

    # Get balanced features and labels
    balanced_feat = feat[selected_indices]
    balanced_label = label[selected_indices]

    data_size = balanced_feat.shape[0]
    split_point = int(data_size * percent)  # training data size

    # Split into train/test sets
    training_sequence = balanced_feat[:split_point]
    training_labels = balanced_label[:split_point]
    testing_sequence = balanced_feat[split_point:]
    testing_labels = balanced_label[split_point:]

    return training_sequence, testing_sequence, training_labels, testing_labels


def train_test_splitter1(data, percent, num=10):
    feat, label = Load_data(data)  # load your features and labels
    unique_classes = np.unique(label)

    selected_indices_per_class = []

    for cls in unique_classes:
        class_indices = np.where(label == cls)[0]
        # Randomly choose `num` samples per class without replacement
        selected_class_indices = np.random.choice(class_indices, num, replace=True)
        selected_indices_per_class.append(selected_class_indices)

    # Concatenate all selected indices from each class
    selected_indices = np.concatenate(selected_indices_per_class)

    # Shuffle the combined indices
    np.random.shuffle(selected_indices)

    # Get balanced features and labels
    balanced_feat = feat[selected_indices]
    balanced_label = label[selected_indices]

    data_size = balanced_feat.shape[0]
    split_point = int(data_size * percent)  # training data size

    # Split into train/test sets
    training_sequence = balanced_feat[:split_point]
    training_labels = balanced_label[:split_point]
    testing_sequence = balanced_feat[split_point:]
    testing_labels = balanced_label[split_point:]

    return training_sequence, testing_sequence, training_labels, testing_labels


def models_return_metrics(data, epochs, ok=True, percents=None, force_retrain=False):
    import os

    training_percentages = percents if percents is not None else [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    model_registry = {
        "Model1": CNN_1,
        "Model2": CNN_BiLSTM,
        "Model3": CNN_BiLSTM_KNN,
        "Model4": CNN_LSTM_LGBM,
        "Model5": BiLSTM_LSTM_ANN,
        "Model6": CNN_KNN_1,
        "Model7": CNN_KNN_2
    }

    if ok:
        for model_name, model_fn in model_registry.items():
            print(f"\n==== Training model: {model_name} ====")
            all_metrics = []

            for percent in training_percentages:
                print(f"  → Training {model_name} with {int(percent * 100)}% training data...")

                x_train, x_test, y_train, y_test = train_test_splitter1(data, percent=percent)
                if model_name == "Inceptionv3":

                    metrics = model_fn(x_train, x_test, y_train, y_test, epochs, data)
                else:
                    metrics = model_fn(x_train, x_test, y_train, y_test, epochs)

                all_metrics.append(metrics)

            # Save after all percentages
            save_path = f"Temp/{data}/Comp/{model_name}/all_metrics.npy"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, np.array(all_metrics, dtype=object))

            print(f"✔ Saved all metrics for {model_name} to {save_path}")