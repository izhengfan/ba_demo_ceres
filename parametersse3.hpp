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
    //Eigen::Map<const Eigen::Quaterniond> quaterd(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> trans(parameters[0] + 3);
    //Eigen::Map<const Eigen::Vector3d> point(parameters[1]);

    double p[3];

    ceres::AngleAxisRotatePoint(parameters[0], parameters[1], p);

    p[0] += trans[0];
    p[1] += trans[1];
    p[2] += trans[2];

    if(p[2] < 1e-4)
    {
        cout << endl;
    }

    residuals[0] = f * p[0] / p[2] + cx - _observation_x;
    residuals[1] = f * p[1] / p[2] + cy - _observation_y;

    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_cam;
    double f_by_z = f / p[2];
    J_cam << f_by_z, 0, - f_by_z * p[0] / p[2],
            0, f_by_z, - f_by_z * p[1] / p[2];
    cout << "Evaluate!" << endl;

    if(jacobians == NULL)
    {
        cout << endl;
    }

    if(jacobians[0] == NULL)
    {
        cout << endl;
    }
    if(jacobians[1] == NULL)
    {
        cout << endl;
    }

    if(jacobians != NULL && jacobians[0] != NULL && jacobians[1] != NULL)
    {
        cout << "Calculate J!" << endl;
        Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor> > J_se3(jacobians[0]);
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J_point(jacobians[1]);
        J_se3.setZero();
        J_se3.block<2,3>(0,0) = - J_cam * SE3::skew(Vector3d(p));
        J_se3.block<2,3>(0,3) = J_cam;
        Quaterniond quaterd = toQuaterniond(Vector3d(parameters[0]));
        J_point = J_cam * quaterd.toRotationMatrix();
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
    ceres::MatrixRef(jacobian, 6, 6) = ceres::Matrix::Identity(6, 6);
    //cout << "Compute J!";
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

struct Camera {
    Camera(double fx, double fy, double cx, double cy) {
        fx_ = fx;
        fy_ = fy;
        cx_ = cx;
        cy_ = cy;
    }

    Eigen::Vector2d project(double x, double y, double z);
    Eigen::Vector2d project(Eigen::Vector3d point);
    Eigen::Vector3d bacProject(Eigen::Vector2d uv, double d);

    double fx_;
    double fy_;
    double cx_;
    double cy_;
};


Eigen::Vector2d Camera::project(double x, double y, double z) {
    Eigen::Vector2d uv;
    uv(0) = fx_ * x / z + cx_;
    uv(1) = fy_ * y / z + cy_;

    return uv;
}

Eigen::Vector2d Camera::project(Eigen::Vector3d point) {
    return project(point(0), point(1), point(2));
}

Eigen::Vector3d Camera::bacProject(Eigen::Vector2d uv, double d) {
    Eigen::Vector3d point;
    point(2) = d;
    point(0) = (uv(0) - cx_) * d / fx_;
    point(1) = (uv(1) - cy_) * d / fy_;

    return point;
}

/* ############################################################################################
 * ############################################################################################
 */

class reprojectErr : public ceres::SizedCostFunction<2, 6, 3> {
public:
    reprojectErr(Eigen::Vector3d pt, Eigen::Vector2d uv,
                 Eigen::Matrix<double, 2, 2> information,
                 std::shared_ptr<Camera> cam);
    virtual ~reprojectErr() {}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

public:
    Eigen::Vector3d pt_;
    Eigen::Vector2d uv_;
    std::shared_ptr<Camera> cam_;
    Eigen::Matrix<double, 2, 2> sqrt_information_;
    static int index;
};

int reprojectErr::index = 0;

bool reprojectErr::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> lie(parameters[0]);
    Sophus::SE3d T = Sophus::SE3d::exp(lie);

    Eigen::Map<const Eigen::Vector3d> pt(parameters[1]);

    //std::cout << T.matrix3x4() << std::endl;

    Eigen::Vector3d P = T * pt;
    Eigen::Vector2d uv = cam_->project(P);
    Eigen::Vector2d err = uv - uv_;
    err = sqrt_information_ * err;

    residuals[0] = err(0);
    residuals[1] = err(1);

    Eigen::Matrix<double, 2, 6> Jac = Eigen::Matrix<double, 2, 6>::Zero();
    Jac(0, 0) = cam_->fx_ / P(2); Jac(0, 2) = -P(0) / P(2) /P(2) * cam_->fx_; Jac(0, 3) = Jac(0, 2) * P(1);
    Jac(0, 4) = cam_->fx_ - Jac(0, 2) * P(0); Jac(0, 5) = -Jac(0, 0) * P(1);

    Jac(1, 1) = cam_->fy_ / P(2); Jac(1, 2) = -P(1) / P(2) /P(2) * cam_->fy_; Jac(1, 3) = -cam_->fy_ + Jac(1, 2) * P(1);
    Jac(1, 4) = -Jac(1, 2) * P(0); Jac(1, 5) = Jac(1, 1) * P(0);
    Jac = sqrt_information_ * Jac;

    int k = 0;
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 6; ++j) {
            if(k >= 12)
                return false;
            if(jacobians) {
                if(jacobians[0])
                    jacobians[0][k] = Jac(i, j);
            }
            k++;
        }
    }

    if(jacobians != NULL && jacobians[1] != NULL)
    {
        Eigen::Matrix<double, 2, 3> J_pt;
        Eigen::Matrix<double, 2, 3> J_cam;
        double f_by_z = cam_->fx_ / P(2);
        J_cam << f_by_z, 0, - f_by_z * P[0] / P[2],
                0, f_by_z, - f_by_z * P[1] / P[2];
        J_pt = J_cam * T.rotationMatrix();
        for(int i = 0; i < 2; i++)
            for(int j = 0; j < 3; j++)
            {
                jacobians[1][i*3+j] = J_pt(i, j);
            }
    }

    //printf("jacobian ok!\n");

    return true;

}

reprojectErr::reprojectErr(Eigen::Vector3d pt, Eigen::Vector2d uv,
                           Eigen::Matrix<double, 2, 2> information,
                           std::shared_ptr<Camera> cam) :   pt_(pt), uv_(uv), cam_(cam) {

    //printf("index = %d\n", index++);
    Eigen::LLT<Eigen::Matrix<double, 2, 2>> llt(information);
    sqrt_information_ = llt.matrixL();
}


class CERES_EXPORT SE3Parameterization : public ceres::LocalParameterization {
public:
    SE3Parameterization() {}
    virtual ~SE3Parameterization() {}
    virtual bool Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const;
    virtual bool ComputeJacobian(const double* x,
                                 double* jacobian) const;
    virtual int GlobalSize() const { return 6; }
    virtual int LocalSize() const { return 6; }
};

bool SE3Parameterization::ComputeJacobian(const double *x, double *jacobian) const {
    ceres::MatrixRef(jacobian, 6, 6) = ceres::Matrix::Identity(6, 6);
    return true;
}

bool SE3Parameterization::Plus(const double* x,
                  const double* delta,
                  double* x_plus_delta) const {
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> lie(x);
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta_lie(delta);

    Sophus::SE3d T = Sophus::SE3d::exp(lie);
    Sophus::SE3d delta_T = Sophus::SE3d::exp(delta_lie);
    Eigen::Matrix<double, 6, 1> x_plus_delta_lie = (delta_T * T).log();

    for(int i = 0; i < 6; ++i) x_plus_delta[i] = x_plus_delta_lie(i, 0);

    return true;

}

#endif // PARAMETERSSE3_HPP
