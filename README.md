# Examples for the Parallel Algorithms tutorial, Summer 2019, University of Wuppertal

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

 ** implement two (out of 6) block variants of GE for LU decomposition. Compare both implementations, exec time - wise. Make sure to use BLAS operations (through NumPy). No GPU

 ** add count of number of fundamental arithmetic operations, and add display of flops measurement

 ** extend the (two) implemented blocks forms to be boosted with GPU through PyCUDA and Scikit-CUDA
