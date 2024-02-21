"""
 Conjugate gradient optimization + utility functions
 To be used with operators from operators.py
 CG code follows the notation in the wikipedia page:
    https://en.wikipedia.org/wiki/Conjugate_gradient_method
 -- Hemant Tagare 2/19/2024
"""

import numpy as np
import optlib.operators as op


##########################################################
#  Utility functions for conjugate gradient
##########################################################

def inner_prod(x, y):
    """
        real and complex inner product
        Works for vectors and matrices
    """
    return np.real(np.sum(np.conj(x) * y))


def norm_sq(x):
    """
        Euclidean norm sq for vectors
        Frobenius norm sq for matrices
        Works for real and complex
    """
    return inner_prod(x, x)


def norm(x):
    """
        Euclidean norm for vectors
        Frobenius norm for matrices
        Works for real and complex
    """
    return np.sqrt(norm_sq(x))


def lhs_op(A, B):
    """
        Creates the operator on the LHS of the CG linear equation
        i.e. creates (A^TA+B^TB)
    """

    def lhs(x):
        return A.transpose(A.forward(x)) + B.transpose(B.forward(x))

    return lhs


##################################################################
#    Conjugate gradient minimization with L2 regularization
##################################################################

# def c_grad_inner(y,A,x,B=op.zero_op(),max_iter=30,f_tol=1e-5):
def c_grad_inner(y, A_star, A_T, x, B=op.zero_op(), max_iter=6, f_tol=1e-5):
    """
     Conjugate gradient inner loo for minimizing ||y-Ax||^2 + ||Bx||^2
     The second term is a regulizer, and is optional
     The termination criteria are
         either k=num_steps >= max_iter 
         or ||y-Ax||<= f_tol*||y|| (Scipy criterion)
         
    The minimizer is found by solving the linear equation
            (A^TA + B^TB)x=A^Ty
    
    Inputs: y,A,B as defined by the objective function above. A and B
                 have to be operators with forward and transpose defined
            max_iter: maximum number of iterations
            f_tol: as defined above
    
    Output: tuple (x,flag)
        where x is the solution
             flag=1 if max_iter are reached, else 0
    """
    # Mathematical comments below correspond to Wikipedia CG formulae
    # written in Latex
    # See https://en.wikipedia.org/wiki/Conjugate_gradient_method

    # A_star=lhs_op(A,B) #A_star is the operator A^*=(A^TA + B^TB)
    b = A_T(y)  # A^Ty
    r_k = b - A_star(x)
    p_k = r_k
    x_k = x
    # initialize iteration
    k = 0
    t = f_tol * norm(b)
    res_norm_sq = norm_sq(r_k)
    while ((k < max_iter) & (np.sqrt(res_norm_sq) >= t)):
        AtAp = A_star(p_k)  # Precalculate to save flops
        alpha_k = res_norm_sq / inner_prod(p_k, AtAp)  # alpha_k = r^T_kr_k/p^T_k A^*p_k
        x_k1 = x_k + alpha_k * p_k  # x_{k+1}=x_k+\alpha_k p_k
        r_k1 = r_k - alpha_k * AtAp  # r_{k+1}= r_{k}-\alpha_k A^*p_k
        beta_k = norm_sq(r_k1) / res_norm_sq  # \beta_k = r^T_{k+1}r_{k+1}/r^T_kr_k
        p_k1 = r_k1 + beta_k * p_k  # p_{k+1}= r_{k+1}+\beta_k p_k
        # update
        k = k + 1
        x_k = x_k1
        p_k = p_k1
        r_k = r_k1
        res_norm_sq = norm_sq(r_k)
        print(f"CG: step={k} res_norm={np.sqrt(res_norm_sq)} ")
    return x_k, k >= max_iter


def c_grad(y, A, x, B=op.zero_op(), max_iter=30, max_inner_iter=6, f_tol=1e-5):
    """
    Conjugate gradient inner loo for minimizing ||y-Ax||^2 + ||Bx||^2
   The second term is a regulizer, and is optional
   The termination criteria are
       either k=num_steps >= max_iter
       or ||y-Ax||<= f_tol*||y|| (Scipy criterion)

  The minimizer is found by solving the linear equation
          (A^TA + B^TB)x=A^Ty

  Inputs: y,A,B as defined by the objective function above. A and B
               have to be operators with forward and transpose defined
          max_iter: maximum number of outer iterations
          max_inner_iterations: maximum number of inner iterations (defined below)
          f_tol: as defined above

  Output: tuple (x,flag)
      where x is the solution
           flag=1 if max_iter are reached, else 0

  The algorithm calls c_grad_inner which is the actual congujate gradient
  algorithm. c_grad_inner is run a maximum of max_inner_iter. This is done so that
  c_grad_inner is restarted to prevent round-off trouble.
    """
    A_star = lhs_op(A, B)  # A_star is the operator A^*=(A^TA + B^TB)
    A_T = A.transpose  # A transpose operator
    k = 0
    x, iter_flag = c_grad_inner(y, A_star, A_T, x, B, max_iter=max_inner_iter, f_tol=f_tol)
    while ((k < max_iter) & (iter_flag == True)):  # Restart the cg
        x, iter_flag = c_grad_inner(y, A_star, A_T, x, B, max_iter=max_inner_iter, f_tol=f_tol)
    iter_flag = iter_flag | (k >= max_iter)  # Check if either criteria is met
    return x, iter_flag
