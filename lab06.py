import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib.colors import ListedColormap

# To load the EEG project data
def load_project_data(file_path):
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw.filter(8, 12, fir_design='firwin', verbose=False)
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        used_ids = list(event_id.values())[:2]
        used_events = {k: v for k, v in event_id.items() if v in used_ids}

        epochs = mne.Epochs(raw, events, event_id=used_events, tmin=0, tmax=2.0,
                            picks=['C3', 'C4', 'Cz'], baseline=None, verbose=False)

        psd = epochs.compute_psd(method='welch', fmin=8, fmax=12, verbose=False)
        X = np.mean(psd.get_data(), axis=2)
        X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-9) * 10
        return X_scaled

# A4: To convert continuous attributes to categorical using equal width or frequency binning
# Uses default parameters (bins=4, method='width') to simulate function overloading
def bin_continuous_feature(data, bins=4, method='width'):
    if method == 'width':
        # Equal width binning divides the range into equal parts
        binned_data = pd.cut(data, bins=bins, labels=False)
    elif method == 'frequency':
        # Equal frequency binning divides data into quantiles
        binned_data = pd.qcut(data, q=bins, labels=False, duplicates='drop')
    else:
        binned_data = pd.cut(data, bins=bins, labels=False)
    return binned_data

# A1: To calculate the entropy of a dataset outcome
def calculate_entropy(y):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    probabilities = class_counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Added 1e-9 to prevent log(0)
    return entropy

# A2: To calculate the Gini index for a dataset
def calculate_gini(y):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    probabilities = class_counts / len(y)
    gini = 1 - np.sum(probabilities ** 2)
    return gini

# To calculate Information Gain for a specific feature split
def calculate_information_gain(X_column, y):
    parent_entropy = calculate_entropy(y)

    unique_values, value_counts = np.unique(X_column, return_counts=True)
    weighted_child_entropy = 0

    for val, count in zip(unique_values, value_counts):
        subset_y = y[X_column == val]
        child_entropy = calculate_entropy(subset_y)
        weight = count / len(y)
        weighted_child_entropy += weight * child_entropy

    info_gain = parent_entropy - weighted_child_entropy
    return info_gain

# A3: To detect the best feature for the root node based on Information Gain
def find_root_node(X, y):
    num_features = X.shape[1]
    info_gains = []

    for i in range(num_features):
        gain = calculate_information_gain(X[:, i], y)
        info_gains.append(gain)

    best_feature_index = np.argmax(info_gains)
    return best_feature_index, info_gains

# A5: To build a custom Decision Tree module (Recursive structure)
def build_custom_dt(X, y, features_list, max_depth=3, current_depth=0):
    # Stop if max depth reached or all labels are the same
    if current_depth >= max_depth or len(np.unique(y)) == 1:
        majority_class = np.argmax(np.bincount(y))
        return majority_class

    best_feature_idx, _ = find_root_node(X, y)
    best_feature_name = features_list[best_feature_idx]

    tree = {best_feature_name: {}}
    unique_values = np.unique(X[:, best_feature_idx])

    for val in unique_values:
        subset_mask = X[:, best_feature_idx] == val
        subset_X = X[subset_mask]
        subset_y = y[subset_mask]

        # Recursively build branches
        tree[best_feature_name][val] = build_custom_dt(
            subset_X, subset_y, features_list, max_depth, current_depth + 1
        )

    return tree

# A6: To draw and visualize the decision tree using sklearn
def task_a6_visualize_tree(X_train, y_train, feature_names):
    dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
    dt_model.fit(X_train, y_train)

    plt.figure(figsize=(12, 8))
    plot_tree(dt_model, feature_names=feature_names, filled=True, rounded=True, fontsize=10)
    plt.title("A6: Decision Tree Visualization")
    return dt_model

# A7: To visualize the decision boundary using 2 features
def task_a7_decision_boundary(X_train, y_train, feature_names):
    # Restrict to first 2 features for 2D visualization
    X_2d = X_train[:, :2]

    dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
    dt_model.fit(X_2d, y_train)

    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    Z = dt_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    colors = ['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA']
    cmap_light = ListedColormap(colors[:len(np.unique(y_train))])

    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, edgecolor='k', s=50, cmap='Set1')

    plt.title("A7: Decision Boundary in Vector Space (2 Features)")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend(*scatter.legend_elements(), title="Classes")

# Loading the project data (EEG features)
file_path = '17152403/edf_with_trigger/edf_with_trigger/Subj_01.edf'
X_full = load_project_data(file_path)

# Simulating the continuous target variable (e.g., Glove Coordinates 'Palm_X')
# Here we use Feature 2 (Cz) as a stand-in for the continuous outcome
y_continuous = X_full[:, 2]

# A4 & A1: Converting continuous outcome to 4 categorical bins (width binning)
y_categorical = bin_continuous_feature(y_continuous, bins=4, method='width')

# Isolating Features (C3, C4 powers)
X_features = X_full[:, :2]
feature_list = ['C3_Power', 'C4_Power']

# A4: Binning features so they can be used in our custom tree (Information Gain)
X_binned = np.zeros_like(X_features)
X_binned[:, 0] = bin_continuous_feature(X_features[:, 0], bins=3, method='frequency')
X_binned[:, 1] = bin_continuous_feature(X_features[:, 1], bins=3, method='frequency')

# A1: Calculating overall Entropy
total_entropy = calculate_entropy(y_categorical)
print(f"A1: Total Dataset Entropy: {total_entropy:.4f}")

# A2: Calculating overall Gini Index
total_gini = calculate_gini(y_categorical)
print(f"A2: Total Dataset Gini Index: {total_gini:.4f}")

# A3: Detecting the root node using Information Gain
root_idx, gains = find_root_node(X_binned, y_categorical)
print(f"A3: Information Gains -> {feature_list[0]}: {gains[0]:.4f} | {feature_list[1]}: {gains[1]:.4f}")
print(f"A3: Selected Root Node Feature: {feature_list[root_idx]}")

# A5: Building Custom Decision Tree Module
custom_tree = build_custom_dt(X_binned, y_categorical, feature_list, max_depth=2)
print(f"A5: Custom Decision Tree Dictionary:\n{custom_tree}")

# A6: Drawing and Visualizing the Sklearn Decision Tree
print("\nGenerating A6: Decision Tree Visualization...")
task_a6_visualize_tree(X_features, y_categorical, feature_list)

# A7: Visualizing Decision Boundary in Vector Space
print("Generating A7: Decision Boundary Space...")
task_a7_decision_boundary(X_features, y_categorical, feature_list)

# Displaying the plots
plt.show()