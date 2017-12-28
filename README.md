# ba_demo_ceres

Bundle adjustment demo using Ceres Solver

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

Please note that in many on-manifold graph optimization problems, only the Jacobians of error functions w.r.t. local Lie algebras are required.  However, in Ceres Solver, one can only seperately define the Jacobians of error functions w.r.t. the Lie group paramerization, and the Jacobians of Lie groups w.r.t. to the local Lie algebras, rather than directly define the Jacobians of the error functions w.r.t. the local Lie algebra. This may be redundant and cost extra computational resources:

  - One have to derive the explicit Jacobians equations of both `error function w.r.t. Lie group` and `Lie group w.r.t. Lie algebra`.
  - The solver would spend extra time to compute `error functino w.r.t. Lie algebra` Jacobian by multiplying the two Jacobians defined above.

Therefore, for convenience, we use a not-that-elegant but effective trick. Let's say that the error function term is of dimension `m`, the Lie group `N`, and the Lie algebra `n`. We define the first `m*n` block in  `error function w.r.t Lie group` Jacobian to be the actual `error function w.r.t. Lie algebra` Jacobian, with other elements to be zero; and define the first `n*n` block in `Lie group w.r.t. Lie algebra` Jacobian to be identity matrix, with other elements to be zero. Thus, we are free of deriving the two extra Jacobians, and reduce computational burden of the solver -- although Ceres still has to multiply the two Jacobians, the overall computational process gets simpler.
 
### Other tips

One advice for self-defined Jacobians: do not access jacobians[0] and jacobians[1] in the meantime, which may cause seg-fault.
