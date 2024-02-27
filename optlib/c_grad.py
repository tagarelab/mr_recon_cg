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

def inner_prod(x,y):
    """
        real and complex inner product
        Works for vectors and matrices
    """
    return np.real(np.sum(np.conj(x)*y))

def norm_sq(x):
    """
        Euclidean norm sq for vectors
        Frobenius norm sq for matrices
        Works for real and complex
    """
    return inner_prod(x,x)

def norm(x):
    """
        Euclidean norm for vectors
        Frobenius norm for matrices
        Works for real and complex
    """
    return np.sqrt(norm_sq(x))
    
            
def lhs_op(A,B):
    """
        Creates the operator on the LHS of the CG linear equation
        i.e. creates (A^TA+B^TB)
    """
    def lhs(x):
        return A.transpose(A.forward(x))+B.transpose(B.forward(x))
    return lhs

    
##################################################################
#    Conjugate gradient minimization with L2 regularization
##################################################################
   
def c_grad_lin_inner(y,A,x0,max_iter=6,f_tol=1e-5):
    """
     Conjugate gradient inner loop for solving Ax=y, assuming
     A is pos-def.
     To solve the linear equation Cx=d (arbitrary C), set A=C^TC
     and y=A^Td
    
    Inputs: y,A as defined above. A and B
                 have to be operators with forward and transpose defined
                 x0 is initial value of x
            max_iter: maximum number of iterations
            f_tol: as defined above
    
    Output: tuple (x,flag)
        where x is the solution
             flag=1 if max_iter are reached, else 0
    """
    # Mathematical comments below correspond to Wikipedia CG formulae
    # written in Latex
    # See https://en.wikipedia.org/wiki/Conjugate_gradient_method
    
    b=y
    r_k=b-A(x0)
    p_k=r_k
    x_k=x0
    #initialize iteration
    k=0
    t=f_tol*norm(b)
    res_norm_sq=norm_sq(r_k)
    while ((k<max_iter) & (np.sqrt(res_norm_sq)>=t)):
        Ap=A(p_k) #Precalculate to save flops
        alpha_k= res_norm_sq/inner_prod(p_k,Ap) # alpha_k = r^T_kr_k/p^T_k A^*p_k
        x_k1=x_k+alpha_k*p_k                      # x_{k+1}=x_k+\alpha_k p_k
        r_k1=r_k-alpha_k*Ap                     # r_{k+1}= r_{k}-\alpha_k A^*p_k
        beta_k=norm_sq(r_k1)/res_norm_sq          # \beta_k = r^T_{k+1}r_{k+1}/r^T_kr_k
        p_k1=r_k1+beta_k*p_k                      # p_{k+1}= r_{k+1}+\beta_k p_k
        #update
        k=k+1
        x_k=x_k1
        p_k=p_k1
        r_k=r_k1
        res_norm_sq=norm_sq(r_k)
   # print(f"CG: step={k} res_norm={np.sqrt(res_norm_sq)} ")
    return x_k,k>=max_iter

def c_grad_lin(b,ATA,x0,max_iter=30,inner_max_iter=6,f_tol=1e-5):
    """
     Outer loop to solve Ax=b. Calles the c_grad_lin_inner loop
        Restarts inner loop every inner_max_iter times to avoid
        numerical round off problems in the inner loop
    """
    x,iter_flag=c_grad_lin_inner(b,ATA,x0,max_iter=inner_max_iter,f_tol=f_tol)
    k=1
    while ((k<max_iter) & (iter_flag==True)): #Restart the cg
        x,iter_flag=c_grad_lin_inner(b,ATA,x,max_iter=inner_max_iter,f_tol=f_tol)
        iter_flag=iter_flag|(k>=max_iter) #Check if either criteria is met
        k=k+1
    return x,iter_flag

def solve_lin_cg(y,A,x0,B=op.zero_op(),c=0,max_iter=6,inner_max_iter=6,f_tol=1e-5):
    """
        Solves linear equation ATAx + B^TBx = A^Ty+c
        To use this to solve linear equation Ax=y for any A (need not be pos. def.)
            use default values in the function call
        B is used as for ridge regression. Typically set B=op.scalar_prod_op(rho) 
    """
    ATA=lhs_op(A,B) #Get ATA
    b=A.transpose(y)+c #Calculate b
    return c_grad_lin(b,ATA,x0,max_iter=max_iter,inner_max_iter=inner_max_iter,f_tol=f_tol)

def solve_L2_min(y,A,x0,B=op.zero_op(),max_iter=6,f_tol=1e-5):
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
    """
    #Create the linear operators
    return solve_lin_cg(y,A,x0,B=op.zero_op(),max_iter=6,f_tol=1e-5)
        
