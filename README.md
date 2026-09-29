# Matrix-operations-linear-algebra-with-NumPy-
# EXPERIMENT - 1

## AIM

Introduce matrix operations used in ML: transpose, inverse, solve linear systems, eigen 
decomposition. 


---

## ALGORITHM

**Brief Algorithm:**

Direct use of numpy.linalg routines; relate to linear regression normal equations. 

---

## PROGRAM

**Run the following program in a Jupyter Notebook code cell:**

```python
import numpy as np

A = np.array([[2.0, 1.0],
              [1.0, 3.0]])

b = np.array([1.0, 2.0])

print("Matrix A:\n", A)

print("\nTranspose A^T:\n", A.T)

print("\nInverse A^-1:\n", np.linalg.inv(A))

eigvals, eigvecs = np.linalg.eig(A)

print("\nEigenvalues:", np.round(eigvals, 4))

x = np.linalg.solve(A, b)

print("\nSolve A x = b --> x:", np.round(x, 4))
```

---

## DETAILED EXPLANATION

### 1. Import NumPy

```python
import numpy as np
```

Loads the NumPy library for working with arrays and performing linear algebra operations.

---

### 2. Define Matrix A

```python
A = np.array([[2.0, 1.0],
              [1.0, 3.0]])
```

Defines a **2 × 2 matrix A**.

\[
A =
\begin{bmatrix}
2 & 1 \\
1 & 3
\end{bmatrix}
\]

---

### 3. Define Vector b

```python
b = np.array([1.0, 2.0])
```

Defines vector `b`, which is used for solving the linear system:

\[
Ax = b
\]

---

### 4. Display Matrix A

```python
print("Matrix A:\n", A)
```

Displays matrix `A` for reference.

---

### 5. Calculate Transpose

```python
A.T
```

Computes the **transpose** of matrix `A`.

The transpose changes rows into columns and columns into rows.

\[
A^T =
\begin{bmatrix}
2 & 1 \\
1 & 3
\end{bmatrix}
\]

---

### 6. Calculate Inverse

```python
np.linalg.inv(A)
```

Computes the **inverse of matrix A**.

The inverse exists only if matrix `A` is **invertible**, meaning its determinant is non-zero.

For the given matrix:

\[
A^{-1} =
\begin{bmatrix}
0.6 & -0.2 \\
-0.2 & 0.4
\end{bmatrix}
\]

---

### 7. Calculate Eigenvalues and Eigenvectors

```python
np.linalg.eig(A)
```

Computes the **eigenvalues and eigenvectors** of matrix `A`.

Eigenvalues and eigenvectors are important in Machine Learning because they help identify important directions and patterns in data.

They are particularly useful in:

- Principal Component Analysis (PCA)
- Dimensionality reduction
- Feature extraction
- Covariance matrix analysis

---

### 8. Solve Linear System

```python
np.linalg.solve(A, b)
```

Numerically solves the linear system:

\[
Ax = b
\]

For the given values:

\[
\begin{bmatrix}
2 & 1 \\
1 & 3
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
=
\begin{bmatrix}
1 \\
2
\end{bmatrix}
\]

The solution is:

\[
x =
\begin{bmatrix}
0.2 \\
0.6
\end{bmatrix}
\]

`np.linalg.solve(A, b)` is generally more numerically stable than calculating:

```python
np.linalg.inv(A).dot(b)
```

---

## SAMPLE OUTPUT

```text
Matrix A:
 [[2. 1.]
  [1. 3.]]

Transpose A^T:
 [[2. 1.]
  [1. 3.]]

Inverse A^-1:
 [[ 0.6 -0.2]
  [-0.2  0.4]]

Eigenvalues: [3.3028 1.6972]

Solve A x = b --> x: [0.2 0.6]
```

---

## RESULT

Students successfully performed concrete matrix computations using NumPy and learned how to solve linear systems.

The experiment also demonstrates the relationship between matrix operations and Machine Learning concepts such as **Linear Regression** and **Principal Component Analysis (PCA)**.

---

## THEORY

Linear algebra is an important mathematical foundation of Machine Learning.
