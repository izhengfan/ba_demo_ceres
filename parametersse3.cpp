#include "parametersse3.hpp"

template<>
bool ReprojectionErrorSE3XYZ<7>::Evaluate(const double * const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Map<const Quaterniond> quaterd(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> trans(parameters[0] + 4);
    Eigen::Map<const Eigen::Vector3d> point(parameters[1]);

    Eigen::Vector3d p = quaterd * point + trans;

    double f_by_z = f / p[2];
    residuals[0] = f_by_z * p[0] + cx - _observation_x;
    residuals[1] = f_by_z * p[1] + cy - _observation_y;

    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_cam;
    double f_by_zz = f_by_z / p[2];
    J_cam << f_by_z, 0, - f_by_zz * p[0],
            0, f_by_z, - f_by_zz * p[1];


    if(jacobians != NULL)
    {
        if(jacobians[0] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3(jacobians[0]);
            J_se3.setZero();
            J_se3.block<2,3>(0,0) = - J_cam * skew(p);
            J_se3.block<2,3>(0,3) = J_cam;
        }
        if(jacobians[1] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J_point(jacobians[1]);
            J_point = J_cam * quaterd.toRotationMatrix();
        }
    }

    return true;
}

template<>
bool ReprojectionErrorSE3XYZ<6>::Evaluate(const double * const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Quaterniond quaterd = toQuaterniond(Eigen::Map<const Vector3d>(parameters[0]));
    Eigen::Map<const Eigen::Vector3d> trans(parameters[0] + 3);
    Eigen::Map<const Eigen::Vector3d> point(parameters[1]);

    Eigen::Vector3d p = quaterd * point + trans;

    double f_by_z = f / p[2];
    residuals[0] = f_by_z * p[0] + cx - _observation_x;
    residuals[1] = f_by_z * p[1] + cy - _observation_y;

    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_cam;
    double f_by_zz = f_by_z / p[2];
    J_cam << f_by_z, 0, - f_by_zz * p[0],
            0, f_by_z, - f_by_zz * p[1];

    if(jacobians != NULL)
    {
        if(jacobians[0] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor> > J_se3(jacobians[0]);
            J_se3.block<2,3>(0,0) = - J_cam * skew(p);
            J_se3.block<2,3>(0,3) = J_cam;
        }
        if(jacobians[1] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J_point(jacobians[1]);
            J_point = J_cam * quaterd.toRotationMatrix();
        }
    }

    return true;
}


template<>
bool PoseSE3Parameterization<7>::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> trans(x + 4);
    SE3 se3_delta = SE3::exp(Eigen::Map<const Vector6d>(delta));

    Eigen::Map<const Eigen::Quaterniond> quaterd(x);
    Eigen::Map<Eigen::Quaterniond> quaterd_plus(x_plus_delta);
    Eigen::Map<Eigen::Vector3d> trans_plus(x_plus_delta + 4);

    quaterd_plus = se3_delta.rotation() * quaterd;
    trans_plus = se3_delta.rotation() * trans + se3_delta.translation();

    return true;
}

template<>
bool PoseSE3Parameterization<7>::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > J(jacobian);
    J.setZero();
    J.block<6,6>(0, 0).setIdentity();
    return true;
}



template<>
bool PoseSE3Parameterization<6>::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> trans(x + 3);
    SE3 se3_delta = SE3::exp(Eigen::Map<const Vector6d>(delta));

    Quaterniond quaterd_plus = se3_delta.rotation() * toQuaterniond(Eigen::Map<const Vector3d>(x));
    Eigen::Map<Vector3d> angles_plus(x_plus_delta);
    angles_plus = toAngleAxis(quaterd_plus);

    Eigen::Map<Eigen::Vector3d> trans_plus(x_plus_delta + 3);
    trans_plus = se3_delta.rotation() * trans + se3_delta.translation();
    return true;
}

template<>
bool PoseSE3Parameterization<6>::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor> > J(jacobian);
    J.setIdentity();
    return true;
}



template<>
void PosePointParametersBlock<7>::getPose(int idx, Quaterniond &q, Vector3d &trans)
{
    double* pose_ptr = values + idx * 7;
    q = Map<const Quaterniond>(pose_ptr);
    trans = Map<const Vector3d>(pose_ptr + 4);
}

template<>
void PosePointParametersBlock<7>::setPose(int idx, const Quaterniond &q, const Vector3d &trans)
{
    double* pose_ptr = values + idx * 7;
    Eigen::Map<Vector7d> pose(pose_ptr);
    pose.head<4>() = Eigen::Vector4d(q.coeffs());
    pose.tail<3>() = trans;
}

template<>
void PosePointParametersBlock<6>::getPose(int idx, Quaterniond &q, Vector3d &trans)
{
    double* pose_ptr = values + idx * 6;
    q = toQuaterniond(Vector3d(pose_ptr));
    trans = Map<const Vector3d>(pose_ptr + 3);
}

template<>
void PosePointParametersBlock<6>::setPose(int idx, const Quaterniond &q, const Vector3d &trans)
{
    double* pose_ptr = values + idx * 6;
    Eigen::Map<Vector6d> pose(pose_ptr);
    pose.head<3>() = toAngleAxis(q);
    pose.tail<3>() = trans;
}
