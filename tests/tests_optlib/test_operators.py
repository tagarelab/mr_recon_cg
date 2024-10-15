"""
   Name: test_operators.py
   Purpose:
   Created on: 10/14/2024
   Created by: Heng Sun
   Additional Notes: 
"""

from unittest import TestCase
import numpy as np
from optlib import operators as ops


def test_adjoint_property(op_instance, x=None, y=None):
    if x is None:
        x = gen_test_x(op_instance)
    if y is None:
        y = gen_test_y(op_instance)

    Ax = op_instance.forward(x)
    Aty = op_instance.transpose(y)

    lhs = np.sum(y * Ax.conj())
    rhs = np.sum(x * Aty.conj())

    return np.allclose(lhs, rhs, atol=1e-6)


def gen_test_x(op_instance):
    x_shape = op_instance.x_shape
    dtype = op_instance.get_x_dtype()
    x = np.random.rand(*x_shape).astype(dtype)
    if dtype is complex:
        x = x + 1j * np.random.rand(*x_shape).astype(dtype)

    return x


def gen_test_y(op_instance):
    y_shape = op_instance.y_shape
    dtype = op_instance.y_dtype
    y = np.random.rand(*y_shape).astype(dtype)
    if dtype is complex:
        y = y + 1j * np.random.rand(*y_shape).astype(dtype)

    return y
