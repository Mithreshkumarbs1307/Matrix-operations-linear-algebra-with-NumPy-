# Matrix-operations-linear-algebra-with-NumPy-
EXPERIMENT - 1
#AIM: 
Introduce matrix operations used in ML: transpose, inverse, solve linear systems, eigen 
decomposition. 
#ALGORITHM (brief): 
Direct use of numpy.linalg routines; relate to linear regression normal equations.
#PROGRAM (run in a Jupyter code cell): 
import numpy as np 
A = np.array([[2.0, 1.0],[1.0, 3.0]]) 
b = np.array([1.0, 2.0]) 
print("Matrix A:\n", A) 
print("\nTranspose A^T:\n", A.T) 
print("\nInverse A^-1:\n", np.linalg.inv(A)) 
eigvals, eigvecs = np.linalg.eig(A) 
print("\nEigenvalues:", np.round(eigvals,4)) 
x = np.linalg.solve(A,b) 
print("\nSolve A x = b --> x:", np.round(x,4)) 
#DETAILED EXPLANATION: 
1. import numpy as np — load NumPy for arrays and linear algebra. 
2. A = np.array(...) — define a 2×2 matrix A. 
3. b = np.array(...) — define vector b for solving A x = b. 
4. print("Matrix A:", A) — show matrix for reference. 
5. A.T — compute transpose (rows ↔ columns). 
6. np.linalg.inv(A) — compute inverse of A (exists only if A is invertible). 
7. np.linalg.eig(A) — compute eigenvalues and eigenvectors; eigenvalues help understand variance directions (PCA). 
8. np.linalg.solve(A,b) — numerically solve linear system more stably than inv(A).dot(b).
#SAMPLE OUTPUT:
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
#RESULT:
Students see concrete matrix computations and how to solve linear systems; relate solve() to normal equations in linear regression.
#THEORY:
Linear algebra (matrix inversion, eigen-decomposition) underpins PCA, linear models, and many optimization problems.Conditioning/invertibility matters. 
