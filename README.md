# MR Reconstruction Code in Python

**2/7/2024:**  Pushed the first notebook. Contains working CG code. Composite and add opearators. Can do L2 regularized CG with real variables. Needs extention to complex variables.

**2/19/2024:** Operators and CG code is now in the module optlib. CG has an inner loop (which is the real CG). The outer loop restarts the inner loop so that takes one gradient step initially. This is to prevent numerical round off in the CG.
