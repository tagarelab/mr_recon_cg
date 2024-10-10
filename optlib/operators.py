"""
 Operators defined for use with optimization for reconstruction
 -- Hemant Tagare 2/19/2024
 Current operators:
    zero_op
    matrix_op
    scalar_prod_op
    hadamard_op
    real_fftn_op
    composite_op
    add_op
"""


#########################################################################
#  Operators
#  Each operator must be a class with forward and transpose methods
###########################################################################
import numpy as np

class zero_op:
    """
        Returns 0
    """
    def forward(self,x):
        return 0.0
    def transpose(self,x):
        return 0.0

class matrix_op:
    """
        matrix_op(A) converts the matrix A to a matrix 
        operator and its transpose
    """

    def __init__(self, A):
        self.A = A

    def forward(self, x):
        return np.matmul(self.A, x)

    def transpose(self, x):
        return np.matmul(np.transpose(self.A), x)


class hadamard_matrix_op:
    """
        hadarmard_matrix_op(A) performs matrix multiplication of A[:,:,i] and x[:,i]
    """

    def __init__(self, A):
        self.A = A

    def forward(self, x):
        z = np.zeros((self.A.shape[0], self.A.shape[2]))
        for i in range(self.A.shape[2]):
            z[:, i] = np.matmul(self.A[:, :, i], x[:, i]).flatten()
        return z

    def transpose(self, x):
        z = np.zeros((self.A.shape[0], self.A.shape[1]))
        for i in range(self.A.shape[2]):
            z[:, i] = np.matmul(self.A[:, :, i].T, x[:, i]).flatten()
        return z


class scalar_prod_op:
    """
       scalar_prod(a) is the scalar product with a
    """

    def __init__(self, a):
        self.a = a

    def forward(self, x):
        return self.a*x
    
    def transpose(self,x):
        return self.forward(x)
    
class hadamard_op:
    """
       Hadamard product. Multiply every entry of 
       x with the corresponding entry of A
    """
    def __init__(self,A):
        self.A=A
    def forward(self,x):
        return self.A*x
    def transpose(self,x):
        # return self.forward(x)  #TODO: check with Hemant about this: is this conjugate transpose?
        return np.conj(self.A) * x

class hadamard_op_expand:
    """
       Hadamard product. Multiply every entry of
       x with the corresponding entry of A
    """

    def __init__(self, A):
        self.A = A

    def forward(self, x):
        x = np.expand_dims(x, axis=-1)
        return self.A * x

    def transpose(self, x):
        return self.forward(x)
    
class real_fftn_op:
    """
        FFT of real valued vectors and matrices
    """
    def forward(self,x):
        return np.fft.fftshift(np.fft.fftn(np.real(x)))
    def transpose(self,x):
        return np.real(np.fft.ifftn(np.fft.ifftshift(x)))

    
class composite_op:
    """
       Creates a composite operator 
       Uses mathematical notation
       That is, composite_op(A,B,C) is the operator ABC, i.e. C operates first
       followed by B followed by A
       composite_op accepts any number of operators as arguments
    """
    def __init__(self,*ops):
        self.ops=ops
        
    def forward(self,x):
        for op in reversed(self.ops):
            x=op.forward(x)
        return x
    
    def transpose(self,x):
        for op in self.ops:
            x=op.transpose(x)
        return x

    
class add_op:
    """
        Creates a sum operator
        e.g. add_op(A,B,C) gives the operator A+B+C
    """
    def __init__(self,*ops):
        self.ops=ops

    def forward(self,x):
        y=0
        for op in self.ops:
            y += op.forward(x)
        return y
    
    def transpose(self,x):
        y=0
        for op in self.ops:
            y += op.transpose(x)
        return y
