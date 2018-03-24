# ba_demo_ceres

Bundle adjustment demo using Ceres Solver

Also read a [blog post](http://fzheng.me/2018/01/23/ba-demo-ceres/).

### Dependencies

- Ceres Solver

### Introduction 

This project implements a simple bundle adjustment problem using Cere Solver to solve it. 

The constraints are defined with customized cost functions and self-defined Jacobians.

Two versions of on-manifold local parameterization of SE3 poses (both of which using 6-dof Lie algebra for updating) are used, including

  - quaternion + translation vector (7d), and
  - rotation vector + translation vector (6d)

Please check `parametersse3.hpp` `parametersse3.cpp` for the local parameterization implementation.

### Jacobians

For convenience in expression, we denote the Jacobian matrix of the error function w.r.t. the Lie group by `J_err_grp`, the Jacobian matrix of the Lie group w.r.t. the local Lie algebra increment by `J_grp_alg`, and the Jacobian matrix of the error function w.r.t. the local Lie algebra increment by `J_err_alg`.

In many on-manifold graph optimization problems, only `J_err_alg` are required.  However, in Ceres Solver, one can only seperately assign `J_err_grp` and `J_grp_alg`, but cannot directly define `J_err_alg`. This may be redundant and cost extra computational resources:

  - One has to derive the explicit Jacobians equations of both `J_err_grp` and `J_grp_alg`.
  - The solver may spend extra time in computing `J_err_grp` and `J_grp_alg`, and multiplying `J_err_grp` and `J_grp_alg` to get `J_err_alg`.

Therefore, for convenience, we use a not-that-elegant but effective trick. Let's say that the error function term is of dimension `m`, the Lie group `N`, and the Lie algebra `n`. We define the leading `m*n` block in  `J_err_grp` to be the actual `J_err_alg`, with other elements to be zero; and define the leading `n*n` block in `J_grp_alg` to be identity matrix, with other elements to be zero. Thus, we are free of deriving the two extra Jacobians, and the computational burden of the solver is reduced -- although Ceres still has to multiply the two Jacobians, the overall computational process gets simpler.
 
### Other tips

One advice for self-defined Jacobians: do not access jacobians[0] and jacobians[1] in the meantime, which may cause seg-fault.


### License 

[BSD New](LICENSE)
