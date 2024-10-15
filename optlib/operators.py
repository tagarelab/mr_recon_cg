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

from abc import ABC, abstractmethod


#########################################################################
#  Operators
#  Each operator must be a class with forward and transpose methods
###########################################################################
import numpy as np


class operator(ABC):
    """
    Abstract class for operators
    """
    x_dtype = np.float64
    y_dtype = np.float64
    x_shape = None  # default shape of input for forward operator
    y_shape = None  # default shape of input for transpose operator

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def transpose(self, y):
        pass

    def get_x_shape(self):
        return self.x_shape

    def get_y_shape(self):
        return self.y_shape

    def get_x_dtype(self):
        return self.x_dtype

    def get_y_dtype(self):
        return self.y_dtype


class zero_op(operator):
    """
        Returns 0
    """
    def forward(self,x):
        return 0.0
    def transpose(self,x):
        return 0.0


class identity_op(operator):
    """
        Returns x
    """

    def forward(self, x):
        return x

    def transpose(self, x):
        return x


class transposed_op(operator):
    """
        Returns the transpose of the original operator
    """

    def __init__(self, op):
        self.op = op
        self.x_shape = op.get_y_shape()
        self.y_shape = op.get_x_shape()
        self.x_dtype = op.get_y_dtype()
        self.y_dtype = op.get_x_dtype()

    def forward(self, x):
        return self.op.transpose(x)

    def transpose(self, x):
        return self.op.forward(x)


class matrix_op(operator):
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


class hadamard_matrix_op(operator):
    """
        hadarmard_matrix_op(A) performs matrix multiplication of A[:,:,i] and x[:,i]
    """

    def __init__(self, A):
        self.A = A
        self.x_shape = (A.shape[1], A.shape[2])
        self.y_shape = (A.shape[0], A.shape[2])

    def forward(self, x):
        y = np.zeros(self.y_shape)
        for i in range(self.y_shape[1]):
            y[:, i] = np.matmul(self.A[:, :, i], x[:, i]).flatten()
        return y

    def transpose(self, y):
        x = np.zeros(self.x_shape)
        for i in range(self.x_shape[1]):
            x[:, i] = np.matmul(self.A[:, :, i].T, y[:, i]).flatten()
        return x


class scalar_prod_op(operator):
    """
       scalar_prod(a) is the scalar product with a
    """

    def __init__(self, a):
        self.a = a

    def forward(self, x):
        return self.a*x
    
    def transpose(self,x):
        return self.forward(x)


class hadamard_op(operator):
    """
       Hadamard product. Multiply every entry of 
       x with the corresponding entry of A
    """
    def __init__(self,A):
        self.A=A
        self.x_shape = A.shape
        self.y_shape = A.shape

    def forward(self,x):
        return self.A*x

    def transpose(self, y):
        return self.A.conj() * y


class hadamard_op_expand(operator):
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


class real_fftn_op(operator):
    """
        FFT of real valued vectors and matrices
    """
    def forward(self,x):
        return np.fft.fftshift(np.fft.fftn(np.real(x)))
    def transpose(self,x):
        return np.real(np.fft.ifftn(np.fft.ifftshift(x)))


class composite_op(operator):
    """
       Creates a composite operator 
       Uses mathematical notation
       That is, composite_op(A,B,C) is the operator ABC, i.e. C operates first
       followed by B followed by A
       composite_op accepts any number of operators as arguments
    """
    def __init__(self,*ops):
        self.ops=ops
        self.x_shape = ops[-1].get_x_shape()  # note here the order is reversed
        self.y_shape = ops[0].get_y_shape()
        self.x_dtype = ops[-1].get_x_dtype()
        self.y_dtype = ops[0].get_y_dtype()
        
    def forward(self,x):
        for op in reversed(self.ops):
            x=op.forward(x)
        return x
    
    def transpose(self,x):
        for op in self.ops:
            x=op.transpose(x)
        return x


class add_op(operator):
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
