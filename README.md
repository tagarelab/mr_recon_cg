# MR Reconstruction Code in Python

**2/7/2024:**  Pushed the first notebook. Contains working CG code. Composite and add opearators. Can do L2 regularized CG with real variables. Needs extention to complex variables.

**2/19/2024:** Operators and CG code is now in the module optlib. CG has an inner loop (which is the real CG). The outer loop restarts the inner loop so that takes one gradient step initially. This is to prevent numerical round off in the CG.

**2/27/2024:** Major refactoring of opt_lib and conjugate gradient in ADMM_l1_l2 branch. This will eventually be merged into the master branch. Also added ADMM for L2-L1 optimization and an example of EMI estimation using ADMM in admm_l2_l1.ipnyb
