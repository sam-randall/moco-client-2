import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
import operator
from sklearn.model_selection import train_test_split
from scipy.stats import beta

def find_threshold(X, y, feature_index):
    # Get the unique values of the feature
    unique_values = np.unique(X[:, feature_index])
    
    best_threshold = None
    best_accuracy = 0
    
    for threshold in unique_values:
        # Predict class 0 if feature value <= threshold, else class 1
        predictions = (X[:, feature_index] > threshold).astype(int)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == y)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            
    return best_threshold, best_accuracy

OP_SYMBOLS = {
    operator.gt: ">",
    operator.ge: ">=",
    operator.lt: "<",
    operator.le: "<=",
    operator.eq: "==",
    operator.ne: "!=",
}

class Feature:
    def __init__(self, feature, feature_name):
        self.feature = feature
        self.feature_name = feature_name

def scan_thresholds_for_binary_feature(feature: Feature, mask: np.ndarray, op: operator, y_class: int) -> pd.DataFrame:
    '''Scans thresholds for a binary feature and returns a DataFrame with the results.'''
    vals = np.unique(feature.feature)

    if op in [operator.gt, operator.ge]:
        vals = np.sort(vals)[::-1]  # Sort descending for greater than operators
    else:
        vals = np.sort(vals)  # Sort ascending for less than operators
    rows = []

    for val in vals:
        pred = op(feature.feature, val).astype(int)
        d = pd.Series(mask[pred == 1]).value_counts().to_dict()
        percentage_true = d.get(True, 0) / (d.get(True, 0) + d.get(False, 0)) * 100 if (True in d or False in d) else 0
        percentage_false = 100 - percentage_true if (True in d or False in d) else 0

        rows.append({
            "Feature ID": feature.feature_name,
            "Threshold": val,
            "Percentage True": percentage_true,
            "Percentage False": percentage_false,
            "Group Size": d.get(True, 0) + d.get(False, 0),
            "Operator Symbol": OP_SYMBOLS[op],
            "Operator": op,
            "Rule": f"feature {feature.feature_name} {OP_SYMBOLS[op]} {val} => class {y_class}"
        })
    return pd.DataFrame(rows)

from scipy.stats import beta

def precision_lower_bound(n_true, n_total, confidence=0.95):
    if n_total == 0:
        return 0.0

    if n_true == 0:
        return 0.0

    alpha = 1 - confidence

    return beta.ppf(
        alpha,
        n_true,
        n_total - n_true + 1
    ) * 100

def scan_thresholds_for_feature(X, feature_id, y, y_class, op, confidence = 0.95) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    feature = X_train[:, feature_id]
    feature = Feature(feature, feature_id)
    thresholds = scan_thresholds_for_binary_feature(feature, y_train == y_class, op, y_class)
    # Validate thresholds using unseen data
    validation_feature = X_test[:, feature_id]
    validation_mask = y_test == y_class

    results = []

    for _, row in thresholds.iterrows():
        selected = op(validation_feature, row["Threshold"])

        n_total = selected.sum()
        n_true = validation_mask[selected].sum()

        precision = (
            n_true / n_total * 100
            if n_total > 0 else 0
        )

        lower_bound = precision_lower_bound(
            n_true, n_total, confidence
        )

        result = row.to_dict()

        result["Validation Group Size"] = n_total
        result["Validation Precision"] = precision
        result["Precision Lower Bound"] = lower_bound

        results.append(result)

    return pd.DataFrame(results)

    

def filter_rules(df, threshold_percentage=90, min_group_size=5):
    filtered_df = df[
        ((df["Percentage True"] >= threshold_percentage) | (df["Percentage False"] >= threshold_percentage)) &
        (df["Group Size"] >= min_group_size)
    ]
    return filtered_df

def scan_thresholds(X, y, feature_index):
    feature = X[:, feature_index]
    c = 0  # Class to check against (Setosa)
    op = operator.le  # Operator for comparison

    return scan_thresholds_for_feature(feature, feature_index, y, c, op)


def scan_matrix_for_rules(X, y):
    rules = []
    y_unique = np.unique(y)
    dfs = []
    for y_class in y_unique:
        for feature_index in range(X.shape[1]):
            df_le = scan_thresholds_for_feature(X, feature_index, y, y_class=y_class, op=operator.le)
            df_ge = scan_thresholds_for_feature(X, feature_index, y, y_class=y_class, op=operator.ge)

            dfs.append(df_le)
            dfs.append(df_ge)

    combined_df = pd.concat(dfs, ignore_index=True)

    return combined_df

def main():
    X, y = load_iris(return_X_y=True)
    # t, a = find_threshold(X, y == 2, feature_index=0)

    rules = scan_matrix_for_rules(X, y)
    rules_filtered = filter_rules(rules, threshold_percentage=90, min_group_size=5)
    

    print("All rules:")
    print(rules_filtered)




if __name__ == "__main__":
    main()