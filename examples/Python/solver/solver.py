class Solver:

    def __init__(self, params):

        # Default values
        self.usingMPI = False
        self.useSendAhead = False
        # Possible values for self.parallelWrappedStorage: {'column', 'row', '2D'}
        self.parallelWrappedStorage = "column"
        # Possible values for self.verbosity: {'FULL', 'SILENT'}
        self.verbosity = "FULL"

        # Unpacking build params -- replace default values
        if 'usingMPI' in params:
            self.usingMPI = params['usingMPI']
        if 'useSendAhead' in params:
            self.useSendAhead = params['useSendAhead']
        if 'self.parallelWrappedStorage' in params:
            self.parallelWrappedStorage = params['self.parallelWrappedStorage']
        if 'self.verbosity' in params:
            self.verbosity = params['self.verbosity']


    # Based on the attributes of A, the solver will chose the best solution scheme
    def solve(self, A, b, x0):

        if self.verbosity == "FULL":
            print("\nSolving started...")

        if (not A.isSquare):
            if self.verbosity == "FULL":
                print("\nThe matrix is not square. Assuming a Least Squares problem wants to be solved.")
            if A.wasQRApplied:
                return self.solveLeastSquares(A)
            else:
                # QR hasn't ben applied
                A.computeQR()
                return self.solveLeastSquares(A)

        # For solves, support for real and dense matrices
        if A.isSparse or (not A.isReal):
            raise Exception("No methods availble to solve systems with sparse or complex matrices, for now.")
        else:
            if A.isSymmetric:
                # Use Cholesky for the solve
                if A.wasCholeskyApplied:
                    return self.solveUsingCholesky(A, b)
                else:
                    # If Cholesky hasn't been applied, apply it and then call the Cholesky solver
                    A.computeCholesky()
                    return self.solveUsingCholesky(Achol, b)
            else:
                # Use LU decomposition instead
                if A.wasLUApplied:
                    return self.solveUsingLU(A.getLU(), b)
                else:
                    # If LU hasn't been applied, apply it and then call the LU solver
                    A.computeLU()
                    return self.solveUsingLU(A, b)

        if self.verbosity == "FULL":
            print("\n...done solving.")


    def solveLeastSquares(self, A):
        # TODO: add QR check here, and solve using QR decomposition
        raise Exception("The method solveLeastSquares is unavailable, for now.")


    def solveUsingLU(self, A, b):
        raise Exception("Solve by using LU decomposition is unavailble, for now.")


    def solveUsingCholesky(self, A, b):
        raise Exception("Solve by using Cholesky decomposition is unavailble, for now.")
