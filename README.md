## Project Overview

This project applies **core Linear Algebra concepts** to analyze a real-world **Student Performance Dataset**.

The objective is to demonstrate how mathematical foundations such as:

* Vectors
* Matrix operations
* Eigenvalues & Eigenvectors
* LU Decomposition
* Singular Value Decomposition (SVD)
* Principal Component Analysis (PCA)
* Linear Discriminant Analysis (LDA)

are used in real-world data science and machine learning problems.

## Project Objectives

* Represent students as vectors in multi-dimensional space
* Perform matrix operations on academic data
* Analyze subject relationships using covariance
* Identify dominant performance patterns using eigen decomposition
* Reduce dataset dimensions using PCA
* Classify students using LDA
* Demonstrate the role of linear algebra in ML & AI

## Dataset

**File Used:**
`student_performance_dataset.csv`

The dataset contains numerical subject scores for multiple students.

Each student is represented as:

[
X = [x_1, x_2, x_3, ..., x_n]
]

Where:

* ( x_i ) = marks in subject i
* n = total number of subjects

## Mathematical Concepts Used

### Vector Operations

* L1 Norm
* L2 Norm
* Dot Product
* Angle Between Vectors
* Cross Product
* Vector Projection

### Matrix Operations

* Matrix Addition
* Matrix Multiplication
* Transpose
* Determinant
* Inverse

### Covariance Matrix

Measures relationships between subjects.

### Eigenvalues & Eigenvectors

Used to determine dominant variance directions.

[
Ax = \lambda x
]

### LU Decomposition

Factorization:

[
A = LU
]

### Singular Value Decomposition (SVD)

[
A = U \Sigma V^T
]

Used for dimensionality reduction.

### Principal Component Analysis (PCA)

* Reduces high-dimensional data into 2 principal components
* Preserves maximum variance

### Linear Discriminant Analysis (LDA)

* Classifies students into:

  * Above Average
  * Below Average
* Maximizes class separability

## Key Insights

* Student performance follows structured variance patterns
* Some subjects are highly correlated
* A small number of principal components explain most variance
* Linear combinations of subjects can classify students effectively
* Dataset can be compressed without major information loss

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* SciPy

## Output Includes

* Vector similarity analysis
* Covariance matrix
* Eigenvalues & Eigenvectors
* LU & SVD decomposition
* PCA 2D visualization
* LDA classification results

## Final Conclusion

This project successfully demonstrates the application of fundamental linear algebra techniques to real-world educational data. Through matrix transformations, eigen analysis, and dimensionality reduction, we validate that academic performance data exhibits structured geometric patterns, reinforcing the importance of linear algebra in modern computational intelligence systems.
