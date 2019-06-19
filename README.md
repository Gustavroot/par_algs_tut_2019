# Examples for the Parallel Algorithms tutorial, Summer 2019, University of Wuppertal

To use the examples in examples/Python/lu_decomposition/, make sure to install at least NumPy, and in case of interest of using cuBLAS, install PyCUDA and Scikit-CUDA.

To install Scikit-CUDA:

pip install scikit-cuda

and to use anything related to CUDA, install it: https://developer.nvidia.com/how-to-cuda-c-cpp.

To run any of the examples, go to the corresponding directory and execute:

python3 main.py

and to play around with the different implementations for the specific example, change the input parameters in input_params.txt of the corresponding directory.

-----------------------------------

# RECENT UPDATES:

The code has been abstracted through the use of two classes: Solver and Matrix.

Objects of the type Solver have the ability of solving systems of equations of the form Ax=b.

Objects of the type Matrix are an encapsulation of NumPy matrices, with a better control over some attributes and actions necessary for the methods covered in the course.

To get a taste on the way in which these two new classes work, see the file examples/Python/test.py. The main important point there, for now, is the for loop, where multiple algorithms for computing LU decomposition are used, and their execution times are displayed for comparison. Note that, underneath, the class Matrix and especifically in its method computeLU(...), the implementations (already) done in examples/Python/lu_decomposition/lu_decomposers.py are used.

To run the tests associated to this new abstraction of the code, go to examples/Python/ and execute on the terminal:

python3 tests.py

-----------------------------------

# IMMEDIATE TODOs:

 ** switch n and m, confusing notation (and to comply with NumPy and SciPy)

 ** implement two (out of 6) block variants of GE for LU decomposition. Compare both implementations, exec time - wise. Make sure to use BLAS operations (through NumPy). No GPU

 ** add count of number of fundamental arithmetic operations, and add display of flops measurement

 ** extend the (two) implemented blocks forms to be boosted with GPU through PyCUDA and Scikit-CUDA
