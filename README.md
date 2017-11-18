# ba_demo_ceres


Bundle adjustment demo using Ceres Solver, with 

- self-defined Jacobians， and 
- two versions of on-manifold local parameterization of SE3 poses (both of which using 6-dof Lie algebra for updating) including
  - quaternion + translation vector (7d), and
  - rotation vector + translation vector (6d)
  
  
One advice for self-defined Jacobians: do not access jacobians[0] and jacobians[1] in the meantime.
