import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv, det, eig, svd
from scipy.linalg import lu
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ----------------------------------------------------------
# 1. LOAD DATASET
# ----------------------------------------------------------

df = pd.read_csv("student_performance_dataset.csv")

print("First 5 rows of dataset:")
print(df.head())

# Select only numeric subject columns
subjects = df.select_dtypes(include=np.number)
X = subjects.values

print("\nShape of Data Matrix (Students x Subjects):", X.shape)

# ----------------------------------------------------------
# PART A: VECTOR OPERATIONS
# ----------------------------------------------------------

print("\n================ VECTOR OPERATIONS ================")

v1 = X[0]
v2 = X[1]

print("Student 1 Vector:", v1)
print("Student 2 Vector:", v2)

# Norms
l1_norm = np.linalg.norm(v1, 1)
l2_norm = np.linalg.norm(v1, 2)

print("\nL1 Norm of Student 1:", l1_norm)
print("L2 Norm of Student 1:", l2_norm)

# Dot product & angle
dot_product = np.dot(v1, v2)
angle = np.arccos(dot_product / 
                  (np.linalg.norm(v1)*np.linalg.norm(v2)))

print("\nDot Product:", dot_product)
print("Angle (radians):", angle)

# Cross product (first 3 subjects)
if len(v1) >= 3:
    cross_product = np.cross(v1[:3], v2[:3])
    print("\nCross Product (first 3 subjects):", cross_product)

# Projection
projection = (np.dot(v1, v2) / np.dot(v2, v2)) * v2
print("\nProjection of v1 onto v2:", projection)

# ----------------------------------------------------------
# PART B: MATRIX OPERATIONS
# ----------------------------------------------------------

print("\n================ MATRIX OPERATIONS ================")

A = X[:5]
B = X[5:10]

# Matrix Addition
addition = A + B
print("\nMatrix Addition (First 5 + Next 5):\n", addition)

# Matrix Multiplication
multiplication = np.dot(A, B.T)
print("\nMatrix Multiplication:\n", multiplication)

# Square Matrix Example
square_matrix = X[:5, :5]

transpose = square_matrix.T
determinant = det(square_matrix)

print("\nTranspose:\n", transpose)
print("\nDeterminant:", determinant)

if determinant != 0:
    inverse = inv(square_matrix)
    print("\nInverse:\n", inverse)
else:
    print("\nMatrix not invertible (Determinant = 0)")

# ----------------------------------------------------------
# PART C: COVARIANCE & EIGEN
# ----------------------------------------------------------

print("\n================ EIGEN ANALYSIS ================")

cov_matrix = np.cov(X.T)
print("\nCovariance Matrix:\n", cov_matrix)

eigenvalues, eigenvectors = eig(cov_matrix)

print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)

# ----------------------------------------------------------
# PART D: LU DECOMPOSITION
# ----------------------------------------------------------

print("\n================ LU DECOMPOSITION ================")

P, L, U = lu(square_matrix)

print("\nL Matrix:\n", L)
print("\nU Matrix:\n", U)

# ----------------------------------------------------------
# PART E: SVD
# ----------------------------------------------------------

print("\n================ SINGULAR VALUE DECOMPOSITION ================")

U_svd, S_svd, Vt_svd = svd(X)

print("\nSingular Values:\n", S_svd)

# ----------------------------------------------------------
# PART F: PCA (Dimensionality Reduction)
# ----------------------------------------------------------

print("\n================ PCA ================")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection (2D)")
plt.show()

# ----------------------------------------------------------
# PART G: LDA CLASSIFICATION
# ----------------------------------------------------------

print("\n================ LDA ================")

# Create Above/Below Average category
if 'Average' in df.columns:
    df['Category'] = np.where(df['Average'] >= df['Average'].mean(),
                              "Above Average",
                              "Below Average")
else:
    df['Average'] = subjects.mean(axis=1)
    df['Category'] = np.where(df['Average'] >= df['Average'].mean(),
                              "Above Average",
                              "Below Average")

y = df['Category']

lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X_scaled, y)

print("\nLDA Projection:\n", X_lda)

plt.figure()
plt.hist(X_lda[y=="Above Average"], alpha=0.7)
plt.hist(X_lda[y=="Below Average"], alpha=0.7)
plt.title("LDA Classification")
plt.show()

print("\n================ PROJECT COMPLETED ================")