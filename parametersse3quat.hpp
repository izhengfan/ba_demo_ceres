#ifndef PARAMETERSSE3QUAT_HPP
#define PARAMETERSSE3QUAT_HPP

#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <iostream>

#include <unordered_set>
#include <unordered_map>

#include <ceres/ceres.h>
#include <ceres/rotation.h>


#include "se3.hpp"

using namespace std;
using namespace Eigen;

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

class ReprojectionErrorSE3XYZ: public ceres::SizedCostFunction<2, 7, 3>
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

bool ReprojectionErrorSE3XYZ::Evaluate(const double * const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Map<const Eigen::Quaterniond> quaterd(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> trans(parameters[0] + 4);
    Eigen::Map<const Eigen::Vector3d> point(parameters[1]);

    Eigen::Vector3d p = quaterd * point + trans;

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
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3(jacobians[0]);
            J_se3.setZero();
            J_se3.block<2,3>(0,0) = - J_cam * SE3::skew(p);
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

class PoseSE3Parameterization : public ceres::LocalParameterization {
public:
    PoseSE3Parameterization() {}
    virtual ~PoseSE3Parameterization() {}
    virtual bool Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const;
    virtual bool ComputeJacobian(const double* x,
                                 double* jacobian) const;
    virtual int GlobalSize() const { return 7; }
    virtual int LocalSize() const { return 6; }
};

bool PoseSE3Parameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Quaterniond> quaterd(x);
    Eigen::Map<const Eigen::Vector3d> trans(x + 4);

    SE3 se3_delta = SE3::exp(Vector6d(delta));

    Eigen::Map<Eigen::Quaterniond> quaterd_plus(x_plus_delta);
    Eigen::Map<Eigen::Vector3d> trans_plus(x_plus_delta + 4);

    quaterd_plus = se3_delta.rotation() * quaterd;
    trans_plus = se3_delta.rotation() * trans + se3_delta.translation();

    return true;
}

bool PoseSE3Parameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > J(jacobian);
    J.setZero();
    J.block<6,6>(0,0).setIdentity();
    //cout << "Compute J!";
    return true;
}

class PosePointParametersBlock
{
public:
    PosePointParametersBlock(int pose_num, int point_num): poseNum(pose_num), pointNum(point_num)
    {
        values = new double[pose_num * 7 + point_num * 3];
    }
    ~PosePointParametersBlock() { delete[] values; }

    double* pose(int idx) {  return values + idx * 7; }

    double* point(int idx) { return values + poseNum * 7 + idx * 3; }

    int poseNum;
    int pointNum;
    double *values;

};


#endif // PARAMETERSSE3QUAT_HPP
