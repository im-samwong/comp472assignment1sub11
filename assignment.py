import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)

# Set the working directory to the script's location
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)


# Function to evaluate the classifier and append results to a file
def evaluate_classifier(
    clf, X_train, X_test, y_train, y_test, dataset_name, model_name, runs=5
):
    # Open the file just once and write as needed
    with open(f"{dataset_name}-performance.txt", "a") as file:
        file.write(f"**************************************************\n")
        file.write(f"(A) Model: {model_name}\n")
        if hasattr(clf, "best_params_"):
            file.write(f"Best parameters: {clf.best_params_}\n")

        # These lists will store all runs metrics to compute average and variance
        accuracies = []
        macro_f1_scores = []
        weighted_f1_scores = []

        for i in range(runs):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Compute metrics
            cm = confusion_matrix(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average=None, zero_division=0
            )
            accuracy = accuracy_score(y_test, y_pred)
            macro_f1 = metrics.f1_score(y_test, y_pred, average="macro")
            weighted_f1 = metrics.f1_score(y_test, y_pred, average="weighted")

            # Append to lists
            accuracies.append(accuracy)
            macro_f1_scores.append(macro_f1)
            weighted_f1_scores.append(weighted_f1)

            # Write current run metrics
            file.write(f"----Run {i+1}----\n")
            file.write(f"(B) Confusion Matrix:\n{cm}\n")
            file.write(f"(C) Precision: {precision}\n")
            file.write(f"    Recall: {recall}\n")
            file.write(f"    F1: {f1}\n")
            file.write(f"(D) Accuracy: {accuracy}\n")
            file.write(f"    Macro-average F1: {macro_f1}\n")
            file.write(f"    Weighted-average F1: {weighted_f1}\n")

        # Compute averages and variances
        mean_accuracy = np.mean(accuracies)
        var_accuracy = np.var(accuracies, ddof=1)  # ddof=1 provides sample variance
        mean_macro_f1 = np.mean(macro_f1_scores)
        var_macro_f1 = np.var(macro_f1_scores, ddof=1)
        mean_weighted_f1 = np.mean(weighted_f1_scores)
        var_weighted_f1 = np.var(weighted_f1_scores, ddof=1)

        # Write aggregate metrics
        file.write(f"----Aggregate Metrics----\n")
        file.write(f"(A) Average Accuracy: {mean_accuracy}\n")
        file.write(f"    Variance in Accuracy: {var_accuracy}\n")
        file.write(f"(B) Average Macro-average F1: {mean_macro_f1}\n")
        file.write(f"    Variance in Macro-average F1: {var_macro_f1}\n")
        file.write(f"(C) Average Weighted-average F1: {mean_weighted_f1}\n")
        file.write(f"    Variance in Weighted-average F1: {var_weighted_f1}\n")
        file.write(f"**************************************************\n\n")


# Penguins data set
penguins_data_set = pd.read_csv(r".\COMP472-A1-datasets\penguins.csv")

# print(penguins_data_set)

# 1-Hot vectors
penguins_one_hot = pd.get_dummies(penguins_data_set, columns=["island", "sex"])

penguins_one_hot["island_Biscoe"] = penguins_one_hot["island_Biscoe"].astype(int)
penguins_one_hot["island_Dream"] = penguins_one_hot["island_Dream"].astype(int)
penguins_one_hot["island_Torgersen"] = penguins_one_hot["island_Torgersen"].astype(int)
penguins_one_hot["sex_FEMALE"] = penguins_one_hot["sex_FEMALE"].astype(int)
penguins_one_hot["sex_MALE"] = penguins_one_hot["sex_MALE"].astype(int)

# print(penguins_one_hot)

# Manual Categorical
penguins_data_set["island"] = penguins_data_set["island"].map(
    {"Torgersen": 0, "Biscoe": 1, "Dream": 2}
)
penguins_data_set["sex"] = penguins_data_set["sex"].map({"MALE": 0, "FEMALE": 1})

# print(penguins_data_set)

# Assuming 'penguins_data_set' is your DataFrame after loading the penguins data
species_counts = penguins_data_set["species"].value_counts(normalize=True) * 100

# Plot the percentages using a bar chart
plt.figure(figsize=(8, 6))
species_counts.plot(kind="bar")
plt.title("Percentage of Instances in Each Penguin Species")
plt.xlabel("Species")
plt.ylabel("Percentage of Instances")
plt.xticks(
    rotation=0
)  # This ensures that class names are not rotated; modify if necessary.

# Save the plot as a PNG file
plt.savefig("penguin-classes.png")
plt.close()  # Close the plot to free up memory

# Convert the PNG image to GIF and save it in the specified directory
im = Image.open("penguin-classes.png")
im.save("penguin-classes.gif", "GIF")


# Abalone data set(no need to manipulate since all values are numerical already)
abalone_data_set = pd.read_csv(r".\COMP472-A1-datasets\abalone.csv")

# print(abalone_data_set)

# Assuming 'abalone_data_set' is your DataFrame after loading the abalone data
sex_counts = abalone_data_set["Type"].value_counts(normalize=True) * 100

# Plot the percentages using a bar chart
plt.figure(figsize=(8, 6))
sex_counts.plot(kind="bar")
plt.title("Percentage of Instances in Each Abalone Sex Class")
plt.xlabel("Sex")
plt.ylabel("Percentage of Instances")
plt.xticks(
    rotation=0
)  # This ensures that class names are not rotated; modify if necessary.

# Save the plot as a PNG file
plt.savefig("abalone-classes.png")
plt.close()  # Close the plot to free up memory

# Convert the PNG image to GIF and save it in the specified directory
im = Image.open("abalone-classes.png")
im.save("abalone-classes.gif", "GIF")

# Train_Test_Split
# a
# This assumes 'Species' is the target class column and all other columns are features
features = penguins_data_set.drop("species", axis=1)
# This selects just the 'Species' column
target_class = penguins_data_set["species"]
# Convert feature columns into a NumPy array
features_array = features.to_numpy()
# Convert the target class column into a NumPy array
target_class_array = target_class.to_numpy()

Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    features_array, target_class_array, test_size=0.2
)  # , random_state=42)

base_dt = tree.DecisionTreeClassifier(max_depth=4)

evaluate_classifier(base_dt, Xp_train, Xp_test, yp_train, yp_test, "penguin", "Base-DT")

plt.figure(figsize=(20, 10))  # Set the size of the figure
tree.plot_tree(base_dt, filled=True)
plt.title("Decision Tree for Penguin Dataset")
# plt.show()


# This assumes 'Type' is the target class column and all other columns are features
features = abalone_data_set.drop("Type", axis=1)
# This selects just the 'Type' column
target_class = abalone_data_set["Type"]
# Convert feature columns into a NumPy array
features_array = features.to_numpy()
# Convert the target class column into a NumPy array
target_class_array = target_class.to_numpy()

Xa_train, Xa_test, ya_train, ya_test = train_test_split(
    features_array, target_class_array, test_size=0.2
)  # , random_state=42)

base_dt = tree.DecisionTreeClassifier(max_depth=4)

evaluate_classifier(base_dt, Xa_train, Xa_test, ya_train, ya_test, "abalone", "Base-DT")

plt.figure(figsize=(20, 10))  # Set the size of the figure
tree.plot_tree(base_dt, filled=True)
plt.title("Decision Tree for Abalone Dataset")
# plt.show()

# b
# Define the parameter grid to search
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 3, 5],  # Replace 10 and 20 with your choice of depths
    "min_samples_split": [2, 50, 100],  # Replace with your choice of min samples split
}

# Initialize the grid search with a decision tree classifier
grid_search = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=5)

# Penguin
# Fit the grid search to the training data
grid_search.fit(Xp_train, yp_train)

# Get the best model
best_dt = grid_search.best_estimator_

# Now you can use best_dt to make predictions and plot the tree
evaluate_classifier(
    grid_search,
    Xp_train,
    Xp_test,
    yp_train,
    yp_test,
    "penguin",
    "Top-DT with GridSearchCV",
)

# Plot the best decision tree
plt.figure(figsize=(20, 10))
tree.plot_tree(best_dt, filled=True)
plt.title("Best Decision Tree for Penguin data set found by GridSearch")
# plt.show()

# Abalone
# Fit the grid search to the training data
grid_search.fit(Xa_train, ya_train)

# Get the best model
best_dt = grid_search.best_estimator_

evaluate_classifier(
    grid_search,
    Xa_train,
    Xa_test,
    ya_train,
    ya_test,
    "abalone",
    "Top-DT with GridSearchCV",
)

# Plot the best decision tree
plt.figure(figsize=(20, 10))
tree.plot_tree(best_dt, filled=True)
plt.title("Best Decision Tree for Abalone data set found by GridSearch")
# plt.show()

#c
#Penguin
# Initialize the MLP with the specified parameters
base_mlp = MLPClassifier(hidden_layer_sizes=(100, 100),
                         activation='logistic',
                         solver='sgd',
                         max_iter=1000)
                         #random_state=42)  # random_state is optional for reproducibility

evaluate_classifier(base_mlp, Xp_train, Xp_test, yp_train, yp_test, "penguin", "Base-MLP")

#Abalone
evaluate_classifier(base_mlp, Xa_train, Xa_test, ya_train, ya_test, "abalone", "Base-MLP")


#d
# Define the parameter grid to search
param_grid = {
    'activation': ['logistic', 'tanh', 'relu'],
    'hidden_layer_sizes': [(30, 50), (5, 5, 5)],
    'solver': ['adam', 'sgd']
}
grid_search = GridSearchCV(estimator=MLPClassifier(max_iter=1000), param_grid=param_grid)

#Penguin
evaluate_classifier(grid_search, Xp_train, Xp_test, yp_train, yp_test, "penguin", "Top-MLP with GridSearchCV")

#Abalone
evaluate_classifier(grid_search, Xa_train, Xa_test, ya_train, ya_test, "abalone", "Top-MLP with GridSearchCV")