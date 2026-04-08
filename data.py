import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import seaborn as sns
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target

# Basic Info`
# `
print(df.head())
print(df.info())
print(df.describe())

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# ===============================
# VISUALIZATION
# ===============================

# 1. Countplot
sns.countplot(x='species', data=df)
plt.title("Species Count")
plt.show()

# 2. Pairplot
sns.pairplot(df, hue='species', diag_kind='hist')  # safer version
plt.show()

# 3. Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()

# 4. Histogram
df.hist(figsize=(10,8))
plt.show()

# 5. Boxplot
plt.figure(figsize=(10,6))
sns.boxplot(data=df.iloc[:, :-1])  # exclude species
plt.title("Boxplot")
plt.show()