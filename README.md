🌸 Iris Flower Classification using Decision Tree

This project builds an end-to-end Machine Learning model to classify Iris flower species (Setosa, Versicolor, Virginica) based on sepal and petal measurements. The workflow includes data analysis, feature engineering, model training, pruning, visualization, and hyperparameter tuning.

🚀 Project Overview

This project demonstrates:

Loading and exploring the Iris dataset

Feature engineering (adding petal & sepal area)

Training a Decision Tree Classifier

Pruning for better generalization

Visualizing the decision tree and decision boundaries

Hyperparameter tuning using GridSearchCV and RandomizedSearchCV

Evaluating performance using accuracy, F1 score, and confusion matrix

🧠 Key Features

Feature Engineering: Created new columns like petal_area and sepal_area to improve model insights.

Pruned Decision Tree: Used max_depth and min_samples_leaf to reduce overfitting.

Visualizations:

Decision Tree structure

2D decision boundaries

Feature importance

Model Optimization:

GridSearchCV

RandomizedSearchCV

High Model Performance:

93% Test Accuracy

97% Cross-Validation F1 Macro Score

📂 Dataset

The dataset is the well-known Iris dataset, available in sklearn.datasets.
It contains 150 samples and 3 classes:

Setosa

Versicolor

Virginica

Features:

Sepal length

Sepal width

Petal length

Petal width

🛠️ Technologies Used

Python

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

SciPy

📘 Project Workflow
1️⃣ Load and explore data
2️⃣ Create a DataFrame
3️⃣ Feature Engineering
4️⃣ Train–Test Split
5️⃣ Baseline Decision Tree Model
6️⃣ Pruned Decision Tree
7️⃣ Visualizations
8️⃣ Hyperparameter tuning
9️⃣ Model evaluation
📊 Model Performance
Metric	Score
Test Accuracy	93.33%
Test F1 Macro	93.26%
CV F1 Macro	97.47%

The model shows strong and stable performance without overfitting.

📉 Visualizations Included

Decision Tree plot

Decision boundaries (Petal Length vs Petal Width)

Feature importance chart

Pairplots for EDA

🧪 How to Run the Project
pip install numpy pandas matplotlib seaborn scikit-learn scipy


Run the Jupyter Notebook:

jupyter notebook

📝 Conclusion

This project highlights the power of Decision Trees for interpretable machine learning. With pruning, feature engineering, and hyperparameter tuning, the model achieves high accuracy while remaining easy to understand.
