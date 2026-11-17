import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
import operator

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

def scan_thresholds_for_feature(feature, y, y_class, op):
    vals = np.unique(feature)
    rows = []

    for val in vals:
        pred = op(feature, val).astype(int)
        d = pd.Series((y == y_class)[pred == 1]).value_counts().to_dict()
        percentage_true = d.get(True, 0) / (d.get(True, 0) + d.get(False, 0)) * 100 if (True in d or False in d) else 0
        percentage_false = 100 - percentage_true if (True in d or False in d) else 0

        rows.append({
            "Threshold": val,
            "Percentage True": percentage_true,
            "Percentage False": percentage_false,
            "Group Size": d.get(True, 0) + d.get(False, 0),
            "Operator": OP_SYMBOLS[op]
        })
    return pd.DataFrame(rows)

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

    return scan_thresholds_for_feature(feature, y, c, op)

def main():
    X, y = load_iris(return_X_y=True)
    # t, a = find_threshold(X, y == 2, feature_index=0)


    # print("Best threshold for feature 0 (Sepal Length):", t)
    # print("Best accuracy for feature 0 (Sepal Length):", a)

    for i in range(X.shape[1]):
        df_le = scan_thresholds_for_feature(X[:, i], y, y_class=0, op=operator.le) 
        df_ge = scan_thresholds_for_feature(X[:, i], y, y_class=0, op=operator.ge) # Scan thresholds for each feature

        rules_le = filter_rules(df_le)
        rules_ge = filter_rules(df_ge)

        print(f"Feature {i}:")
        print("  Operator <= :")
        print(rules_le)
        print("  Operator >= :")
        print(rules_ge)
    percentages_true, percentages_false, _ = scan_thresholds(X, y, feature_index=0)  # Scan thresholds for Sepal Length (feature index 0)

    # Inputs for the function.
    feature = X[:, 0]  # Sepal Length
    c = 0
    op = operator.le

    vals = np.unique(feature)
    # Scan for a rule that achieves a good ratio of true to false or false to true.
    # Return all the thresholds where the percentage of true is above 90% or the percentage of false is above 90%.

    plt.show()



if __name__ == "__main__":
    main()