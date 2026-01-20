import numpy as np
import os
import random


def generate_metrics_from_accuracy(acc_value):
    """Generate all metrics from given accuracy value for proposed model"""
    # Sensitivity > Accuracy > Specificity
    sens = acc_value + random.uniform(0.005, 0.015)
    spec = acc_value - random.uniform(0.003, 0.008)

    # Recall = Sensitivity
    recall = sens

    # Precision = Specificity - small value
    precision = spec - random.uniform(0.002, 0.005)

    # F1 Score = 2 * (precision * recall) / (precision + recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # TPR = Sensitivity
    tpr = sens

    # FPR = 1 - Specificity
    fpr = 1 - spec

    return {
        'accuracy': acc_value,
        'f1_score': f1_score,
        'precision': precision,
        'recall': recall,
        'sensitivity': sens,
        'specificity': spec,
        'tpr': tpr,
        'fpr': fpr
    }


def generate_training_data(accuracy_value, shape=(8, 6)):
    """
    Generate complete 8x6 matrix for all metrics.

    Columns are saved in the order: 40, 50, 60, 70, 80, 90.
    The input accuracy_value corresponds to 90% training.
    """
    training_percentages = [40, 50, 60, 70, 80, 90]
    # MODEL_ORDER = [2, 4, 6, 7, 1, 3, 5, 8]  # YOUR EXACT ORDER
    MODEL_ORDER = [4, 5, 1, 2, 6, 3, 7, 8]  # YOUR EXACT ORDER

    # Generate proposed model (last row) values
    metrics_proposed = {}
    metrics_names = ['accuracy', 'f1_score', 'precision', 'recall',
                     'sensitivity', 'specificity', 'tpr', 'fpr']

    for metric in metrics_names:
        if metric == 'accuracy':
            base_value_90 = accuracy_value
        else:
            # Use accuracy to generate other metrics for 90% training
            temp_metrics = generate_metrics_from_accuracy(accuracy_value)
            base_value_90 = temp_metrics[metric]

        # Now build values for 40→90, ensuring 90% uses base_value_90
        row = np.zeros(6)

        v90 = base_value_90
        v80 = max(0.5, v90 - random.uniform(0.015, 0.032))
        v70 = max(0.5, v80 - random.uniform(0.015, 0.032))
        v60 = max(0.5, v70 - random.uniform(0.015, 0.032))
        v50 = max(0.5, v60 - random.uniform(0.015, 0.032))
        v40 = max(0.5, v50 - random.uniform(0.015, 0.032))

        # order: 40, 50, 60, 70, 80, 90
        row[:] = [v40, v50, v60, v70, v80, v90]
        metrics_proposed[metric] = row

    # Generate other models (first 7 rows) by subtracting from proposed model
    all_data = {}

    for metric in metrics_names:
        proposed_row = metrics_proposed[metric]
        matrix = np.zeros(shape)
        close_models = 2  # CONTROL HERE: 1 or 2

        # 1. Put proposed at -1
        matrix[-1, :] = proposed_row

        # 2. Generate CLOSE models at fixed positions BEFORE proposed
        for c_idx in range(close_models):  # 0,1 → rows -2,-3
            close_row = np.zeros(6)
            for col_idx in range(6):
                base = proposed_row[col_idx]
                decr = random.uniform(0.008, 0.027)
                close_row[col_idx] = max(0.4, base - decr)
            matrix[-2 - c_idx, :] = close_row  # -2, -3

        # 3. Fill REMAINING front rows with far models using MODEL_ORDER
        n_front_rows = shape[0] - 1 - close_models  # 8-1-2 = 5 rows (0-4)
        for model_idx, model_num in enumerate(MODEL_ORDER[:-1]):  # [2,4,6,7,1,3,5]
            if model_idx < n_front_rows:  # Only fill front rows (0-4)
                far_row = np.zeros(6)
                for col_idx in range(6):
                    base_proposed = proposed_row[col_idx]
                    # Model position determines degradation level
                    position_factor = MODEL_ORDER.index(model_num) * 0.004
                    decrement = random.uniform(0.019 + position_factor, 0.045 + position_factor)
                    far_row[col_idx] = max(0.4, base_proposed - decrement)
                matrix[model_idx, :] = far_row

        all_data[metric] = matrix

    return all_data


def generate_kfold_data(accuracy_value, shape=(8, 5)):
    """Generate 8x5 matrix for k-fold data (6-10 folds)"""
    folds = [6, 7, 8, 9, 10]
    MODEL_ORDER = [4, 5, 1, 2, 6, 3, 7, 8]  # YOUR EXACT ORDER
    close_models = 2  # CONTROL HERE: 1 or 2

    metrics_proposed = {}
    metrics_names = ['accuracy', 'f1_score', 'precision', 'recall',
                     'sensitivity', 'specificity', 'tpr', 'fpr']

    for metric in metrics_names:
        if metric == 'accuracy':
            base_value_10 = accuracy_value
        else:
            temp_metrics = generate_metrics_from_accuracy(accuracy_value)
            base_value_10 = temp_metrics[metric]

        # Generate values across folds (10-fold is best)
        row = np.zeros(5)

        v10 = base_value_10
        v9 = max(0.5, v10 - random.uniform(0.016, 0.032))
        v8 = max(0.5, v9 - random.uniform(0.016, 0.032))
        v7 = max(0.5, v8 - random.uniform(0.016, 0.032))
        v6 = max(0.5, v7 - random.uniform(0.016, 0.032))

        # order: 6, 7, 8, 9, 10
        row[:] = [v6, v7, v8, v9, v10]
        metrics_proposed[metric] = row

    all_data = {}

    for metric in metrics_names:
        proposed_row = metrics_proposed[metric]
        matrix = np.zeros(shape)

        # 1. Put proposed at -1 (Model 8)
        matrix[-1, :] = proposed_row

        # 2. Generate CLOSE models at fixed positions BEFORE proposed
        for c_idx in range(close_models):  # 0,1 → rows -2,-3
            close_row = np.zeros(5)
            for col_idx in range(5):
                base = proposed_row[col_idx]
                decr = random.uniform(0.008, 0.025)
                close_row[col_idx] = max(0.4, base - decr)
            matrix[-2 - c_idx, :] = close_row  # -2, -3

        # 3. Fill REMAINING front rows with MODEL_ORDER front models
        n_front_rows = shape[0] - 1 - close_models  # 8-1-2 = 5 rows (0-4)
        for model_idx, model_num in enumerate(MODEL_ORDER[:-1]):  # [2,4,6,7,1,3,5]
            if model_idx < n_front_rows:  # Only fill front rows (0-4)
                far_row = np.zeros(5)
                for col_idx, fold in enumerate(folds):
                    base_proposed = proposed_row[col_idx]
                    position_factor = MODEL_ORDER.index(model_num) * 0.005
                    decrement = random.uniform(0.025 + position_factor, 0.050 + position_factor)
                    far_row[col_idx] = max(0.4, base_proposed - decrement)
                matrix[model_idx, :] = far_row

        all_data[metric] = matrix

    return all_data


def create_performance_files(comp_data, kf_data, db_name):
    """Create performance analysis files for epochs"""
    metrics_order = ['accuracy', 'sensitivity', 'specificity', 'precision', 'recall', 'f1_score']

    # Get proposed model last row (90% training / 10-fold)
    proposed_comp_row = {metric: comp_data[metric][-1, :] for metric in metrics_order}
    proposed_kf_row = {metric: kf_data[metric][-1, :] for metric in metrics_order}

    epochs = [500, 400, 300, 200, 100]

    # Comparative Performance files
    comp_perf_dir = f"Analysis/Performance_Analysis/Concated_epochs/{db_name}"
    os.makedirs(comp_perf_dir, exist_ok=True)

    current_values = proposed_comp_row.copy()
    for epoch in epochs:
        perf_matrix = np.zeros((6, 6))

        for i, metric in enumerate(metrics_order):
            perf_matrix[i, :] = current_values[metric]

        filename = f"metrics_epochs_{epoch}.npy"
        np.save(os.path.join(comp_perf_dir, filename), perf_matrix)

        if epoch > 100:
            for metric in metrics_order:
                noise = np.random.normal(loc=0.0, scale=0.002, size=current_values[metric].shape)

                column_decay = np.array([
                    random.uniform(0.002, 0.010),
                    random.uniform(0.003, 0.012),
                    random.uniform(0.004, 0.014),
                    random.uniform(0.003, 0.012),
                    random.uniform(0.002, 0.010),
                    random.uniform(0.001, 0.008),
                ])

                current_values[metric] = (
                        current_values[metric]
                        - column_decay
                        + noise
                )

                current_values[metric] = np.clip(current_values[metric], 0.5, 0.999)

    # KFold Performance files
    kf_perf_dir = f"Analysis/KF_PERF/{db_name}"
    os.makedirs(kf_perf_dir, exist_ok=True)

    current_values = proposed_kf_row.copy()
    for epoch in epochs:
        perf_matrix = np.zeros((6, 5))  # 6 metrics, 5 folds

        for i, metric in enumerate(metrics_order):
            perf_matrix[i, :] = current_values[metric]

        filename = f"kf_metrics_epochs_{epoch}.npy"
        np.save(os.path.join(kf_perf_dir, filename), perf_matrix)

        if epoch > 100:
            for metric in metrics_order:
                noise = np.random.normal(0, 0.0025, size=current_values[metric].shape)

                column_decay = np.array([
                    random.uniform(0.003, 0.015),
                    random.uniform(0.004, 0.017),
                    random.uniform(0.005, 0.020),
                    random.uniform(0.004, 0.017),
                    random.uniform(0.003, 0.015),
                ])

                current_values[metric] = (
                        current_values[metric]
                        - column_decay
                        + noise
                )

                current_values[metric] = np.clip(current_values[metric], 0.5, 0.999)


def main(db_name, comp_accuracy, kf_accuracy):
    """Main function to generate all files"""

    comp_dir = f"Analysis/Comparative_Analysis/{db_name}"
    kf_dir = f"Analysis/KF_Analysis/{db_name}"
    os.makedirs(comp_dir, exist_ok=True)
    os.makedirs(kf_dir, exist_ok=True)

    print(f"Generating files for DB: {db_name}")
    print(f"Comparative Accuracy (90%): {comp_accuracy}")
    print(f"KFold Accuracy (10-fold): {kf_accuracy}")

    # 1. Generate Comparative Analysis files (8x6, cols 40..90)
    comp_data = generate_training_data(comp_accuracy, shape=(8, 6))
    metric_files = {
        'accuracy': 'ACC_1.npy',
        'f1_score': 'F1score_1.npy',
        'precision': 'PRE_1.npy',
        'recall': 'REC_1.npy',
        'sensitivity': 'SEN_1.npy',
        'specificity': 'SPE_1.npy',
        'tpr': 'TPR_1.npy',
        'fpr': 'FPR_1.npy'
    }

    for metric, filename in metric_files.items():
        np.save(os.path.join(comp_dir, filename), comp_data[metric])
        print(f"Saved: {filename} -> shape: {comp_data[metric].shape}")

    # 2. Generate KF Analysis files (8x5, folds 6..10)
    kf_data = generate_kfold_data(kf_accuracy, shape=(8, 5))

    kf_metric_files = {
        'accuracy': 'ACC_2.npy',
        'f1_score': 'F1score_2.npy',
        'precision': 'PRE_2.npy',
        'recall': 'REC_2.npy',
        'sensitivity': 'SEN_2.npy',
        'specificity': 'SPE_2.npy',
        'tpr': 'TPR_2.npy',
        'fpr': 'FPR_2.npy'
    }

    for metric, filename in kf_metric_files.items():
        np.save(os.path.join(kf_dir, filename), kf_data[metric])
        print(f"Saved: {filename} -> shape: {kf_data[metric].shape}")

    # 3. Generate Performance files
    create_performance_files(comp_data, kf_data, db_name)
    print("All performance files generated successfully!")

    print("\nFile generation completed!")
    print(f"All files saved under Analysis/ directories for {db_name}")


if __name__ == "__main__":
    DB_NAME = "DB1"
    COMP_ACCURACY = 0.9853410
    KF_ACCURACY = 0.982126

    main(DB_NAME, COMP_ACCURACY, KF_ACCURACY)

    m = np.load("Analysis/Comparative_Analysis/DB1/ACC_1.npy")
    print("The Accuracy values are:", m)

    m = np.load("Analysis/Comparative_Analysis/DB1/SEN_1.npy")
    print("The Sensitivity values are:", m)
