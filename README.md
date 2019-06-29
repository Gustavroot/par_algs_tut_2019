# Examples for the Parallel Algorithms tutorial, Summer 2019, University of Wuppertal

The code has been abstracted through the use of two classes: Solver and Matrix.

Objects of the type Solver have the ability of solving systems of equations of the form Ax=b.

Objects of the type Matrix are an encapsulation of NumPy matrices, with a (maybe) better control over some attributes and actions necessary for the methods covered in the course.

To get a taste on the way in which these two new classes work, run the script examples/Python/test.py:

cd examples/Python/
python3 test.py

and you can play around with the parameters of the run by going to to the file examples/Python/lu_decomposition/inp_params.txt and changing values there. Note that, underneath, the class Matrix and especifically in its method computeLU(...), the implementations done in examples/Python/lu_decomposition/lu_decomposers.py are used.

Get a better feeling on how well LU decomposers work (compared to each other) by trying the following three values for the parameter 'ge_alg' in inp_params.txt: kij_opt, jik_opt, ijk_blocked_opt. Try to make sense out of the timing results!

-----------------------------------

Deprecated(-ish):

To use the examples in examples/Python/lu_decomposition/, make sure to install at least NumPy, and in case of interest of using cuBLAS, install PyCUDA and Scikit-CUDA.

To install Scikit-CUDA:

pip install scikit-cuda

and to use anything related to CUDA, install it: https://developer.nvidia.com/how-to-cuda-c-cpp.

To run any of the examples, go to the corresponding directory and execute:

python3 main.py

and to play around with the different implementations for the specific example, change the input parameters in input_params.txt of the corresponding directory.

-----------------------------------

# IMMEDIATE TODOs:

 ** switch n and m, confusing notation (and to comply with NumPy and SciPy)

 ** add count of number of fundamental arithmetic operations, and add display of flops measurement

 ** extend the (two) implemented blocks forms to be boosted with GPU through PyCUDA and Scikit-CUDA (is there a way? Workaround for the small-blocks LU decomp ... ?)
