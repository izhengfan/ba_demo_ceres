#ifndef PARAMETERSSE3_HPP
#define PARAMETERSSE3_HPP

#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <iostream>

#include <unordered_set>
#include <unordered_map>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "sophus/se3.hpp"


#include "se3.hpp"

using namespace std;
using namespace Eigen;

Vector3d toAngleAxis(const Quaterniond& quaterd)
{
    double q[4] = {quaterd.w(), quaterd.x(), quaterd.y(), quaterd.z() };
    double a[3];
    ceres::QuaternionToAngleAxis(q, a);
    return Vector3d(a);
}

Quaterniond toQuaterniond(const Vector3d& v3d)
{
    double q[4];
    ceres::AngleAxisToQuaternion(v3d.data(), q);
    return Quaterniond(q[0], q[1], q[2], q[3]);
}

class CameraParameters
{
protected:
    double f;
    double cx;
    double cy;
public:
    CameraParameters(double f_, double cx_, double cy_)
        : f(f_), cx(cx_), cy(cy_) {}

    Vector2d cam_map(const Vector3d& p)
    {
        Vector2d z;
        z[0] = f * p[0] / p[2] + cx;
        z[1] = f * p[1] / p[2] + cy;
        return z;
    }
};

class ReprojectionErrorSE3XYZ: public ceres::SizedCostFunction<2, 6, 3>
{
public:
    ReprojectionErrorSE3XYZ(double f_,
                            double cx_,
                            double cy_,
                            double observation_x,
                            double observation_y)
        : f(f_), cx(cx_), cy(cy_),
          _observation_x(observation_x),
          _observation_y(observation_y){}

    virtual ~ReprojectionErrorSE3XYZ() {}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    double f;
    double cx;
    double cy;

private:
    double _observation_x;
    double _observation_y;
};

bool ReprojectionErrorSE3XYZ::Evaluate(double const* const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Map<const Vector3d> angles(parameters[0]);
    Eigen::Map<const Vector3d> trans(parameters[0] + 3);
    Eigen::Map<const Vector3d> pt(parameters[1]);
    Quaterniond quaterd = toQuaterniond(angles);

    Eigen::Vector3d p = quaterd * pt + trans;

    residuals[0] = f * p[0] / p[2] + cx - _observation_x;
    residuals[1] = f * p[1] / p[2] + cy - _observation_y;

    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_cam;
    double f_by_z = f / p[2];
    J_cam << f_by_z, 0, - f_by_z * p[0] / p[2],
            0, f_by_z, - f_by_z * p[1] / p[2];

    if(jacobians != NULL)
    {
        if(jacobians[0] != NULL)
        {
            Eigen::Map<Matrix<double, 2, 6, RowMajor> > J_se3(jacobians[0]);
            J_se3.block<2,3>(0,0) = - J_cam * SE3::skew(Vector3d(p));
            J_se3.block<2,3>(0,3) = J_cam;
        }
        if(jacobians[1] != NULL)
        {
            Eigen::Map<Matrix<double, 2, 3, RowMajor> > J_point(jacobians[1]);
            J_point = J_cam * quaterd.matrix();
        }
    }


    return true;
}

class CERES_EXPORT PoseSE3Parameterization : public ceres::LocalParameterization {
public:
    PoseSE3Parameterization() {}
    virtual ~PoseSE3Parameterization() {}
    virtual bool Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const;
    virtual bool ComputeJacobian(const double* x,
                                 double* jacobian) const;
    virtual int GlobalSize() const { return 6; }
    virtual int LocalSize() const { return 6; }
};

bool PoseSE3Parameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    double q[4];
    ceres::AngleAxisToQuaternion(x, q);
    Eigen::Quaterniond quaterd(q[0], q[1], q[2], q[3]);
    Eigen::Map<const Eigen::Vector3d> trans(x + 3);

    SE3 se3_delta = SE3::exp(Vector6d(delta));

    //Eigen::Map<Eigen::Quaterniond> quaterd_plus(x_plus_delta);
    Eigen::Quaterniond quaterd_plus = se3_delta.rotation() * quaterd;
    Eigen::Map<Vector3d> angles_plus(x_plus_delta);
    angles_plus = toAngleAxis(quaterd_plus);

    Eigen::Map<Eigen::Vector3d> trans_plus(x_plus_delta + 3);
    trans_plus = se3_delta.rotation() * trans + se3_delta.translation();

    return true;
}

bool PoseSE3Parameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Matrix<double, 6, 6, RowMajor> > J(jacobian);
    J.setIdentity();
    return true;
}

class PointParameterization: public ceres::LocalParameterization
{
public:
    PointParameterization() {}
    virtual ~PointParameterization() {}
    virtual bool Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const
    {
        for(int i = 0; i < 3; i++)
            x_plus_delta[i] = x[i] + delta[i];
        return true;
    }
    virtual bool ComputeJacobian(const double* x,
                                 double* jacobian) const
    {
        ceres::MatrixRef(jacobian, 3, 3) = ceres::Matrix::Identity(3, 3);
        return true;
    }
    virtual int GlobalSize() const { return 3; }
    virtual int LocalSize() const { return 3; }
};

class PosePointParametersBlock
{
public:
    PosePointParametersBlock(int pose_num, int point_num): poseNum(pose_num), pointNum(point_num)
    {
        values = new double[pose_num * 6 + point_num * 3];
    }
    ~PosePointParametersBlock() { delete[] values; }

    double* pose(int idx) {  return values + idx * 6; }

    double* point(int idx) { return values + poseNum * 6 + idx * 3; }

    int poseNum;
    int pointNum;
    double *values;

};

#endif // PARAMETERSSE3_HPP
