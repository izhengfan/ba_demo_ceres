#ifndef PARAMETERSSE3_HPP
#define PARAMETERSSE3_HPP

#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <iostream>

#include <ceres/ceres.h>

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

/// PoseBlockSize can only be
/// 7 (quaternion + translation vector) or
/// 6 (rotation vector + translation vector)
template<int PoseBlockSize>
class ReprojectionErrorSE3XYZ: public ceres::SizedCostFunction<2, PoseBlockSize, 3>
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

/// PoseBlockSize can only be
/// 7 (quaternion + translation vector) or
/// 6 (rotation vector + translation vector)
template<int PoseBlockSize>
class PoseSE3Parameterization : public ceres::LocalParameterization {
public:
    PoseSE3Parameterization() {}
    virtual ~PoseSE3Parameterization() {}
    virtual bool Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const;
    virtual bool ComputeJacobian(const double* x,
                                 double* jacobian) const;
    virtual int GlobalSize() const { return PoseBlockSize; }
    virtual int LocalSize() const { return 6; }
};

/// PoseBlockSize can only be
/// 7 (quaternion + translation vector) or
/// 6 (rotation vector + translation vector)
template<int PoseBlockSize>
class PosePointParametersBlock
{
public:
    PosePointParametersBlock(){}
    void create(int pose_num, int point_num)
    {
        poseNum = pose_num;
        pointNum = point_num;
        values = new double[pose_num * PoseBlockSize + point_num * 3];
    }
    PosePointParametersBlock(int pose_num, int point_num): poseNum(pose_num), pointNum(point_num)
    {
        values = new double[pose_num * PoseBlockSize + point_num * 3];
    }
    ~PosePointParametersBlock() { delete[] values; }

    void setPose(int idx, const Quaterniond &q, const Vector3d &trans);

    void getPose(int idx, Quaterniond &q, Vector3d &trans);

    double* pose(int idx) {  return values + idx * PoseBlockSize; }

    double* point(int idx) { return values + poseNum * PoseBlockSize + idx * 3; }

    int poseNum;
    int pointNum;
    double *values;

};


#endif // PARAMETERSSE3_HPP
