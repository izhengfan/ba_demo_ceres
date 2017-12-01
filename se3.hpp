#ifndef SE3_HPP_
#define SE3_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <math.h>

const double SMALL_EPS = 1e-10;

typedef Eigen::Matrix<double, 6, 1, Eigen::ColMajor> Vector6d;
typedef Eigen::Matrix<double, 7, 1, Eigen::ColMajor> Vector7d;

inline Eigen::Matrix3d skew(const Eigen::Vector3d&v)
{
    Eigen::Matrix3d m;
    m.fill(0.);
    m(0,1)  = -v(2);
    m(0,2)  =  v(1);
    m(1,2)  = -v(0);
    m(1,0)  =  v(2);
    m(2,0) = -v(1);
    m(2,1) = v(0);
    return m;
}

inline Eigen::Vector3d deltaR(const Eigen::Matrix3d& R)
{
    Eigen::Vector3d v;
    v(0)=R(2,1)-R(1,2);
    v(1)=R(0,2)-R(2,0);
    v(2)=R(1,0)-R(0,1);
    return v;
}


inline Eigen::Vector3d toAngleAxis(const Eigen::Quaterniond& quaterd, double* angle=NULL)
{
    Eigen::Quaterniond unit_quaternion = quaterd.normalized();
    double n = unit_quaternion.vec().norm();
    double w = unit_quaternion.w();
    double squared_w = w*w;

    double two_atan_nbyw_by_n;
    // Atan-based log thanks to
    //
    // C. Hertzberg et al.:
    // "Integrating Generic Sensor Fusion Algorithms with Sound State
    // Representation through Encapsulation of Manifolds"
    // Information Fusion, 2011

    if (n < SMALL_EPS)
    {
        // If quaternion is normalized and n=1, then w should be 1;
        // w=0 should never happen here!
        assert(fabs(w)>SMALL_EPS);

        two_atan_nbyw_by_n = 2./w - 2.*(n*n)/(w*squared_w);
    }
    else
    {
        if (fabs(w)<SMALL_EPS)
        {
            if (w>0)
            {
                two_atan_nbyw_by_n = M_PI/n;
            }
            else
            {
                two_atan_nbyw_by_n = -M_PI/n;
            }
        }
        two_atan_nbyw_by_n = 2*atan(n/w)/n;
    }
    if(angle!=NULL) *angle = two_atan_nbyw_by_n*n;
    return two_atan_nbyw_by_n * unit_quaternion.vec();
}

inline Eigen::Quaterniond toQuaterniond(const Eigen::Vector3d& v3d, double* angle = NULL)
{
    double theta = v3d.norm();
    if(angle != NULL)
        *angle = theta;
    double half_theta = 0.5*theta;

    double imag_factor;
    double real_factor = cos(half_theta);
    if(theta<SMALL_EPS)
    {
        double theta_sq = theta*theta;
        double theta_po4 = theta_sq*theta_sq;
        imag_factor = 0.5-0.0208333*theta_sq+0.000260417*theta_po4;
    }
    else
    {
        double sin_half_theta = sin(half_theta);
        imag_factor = sin_half_theta/theta;
    }

    return Eigen::Quaterniond(real_factor,
                              imag_factor*v3d.x(),
                              imag_factor*v3d.y(),
                              imag_factor*v3d.z());
}


class SE3 {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

protected:

    Eigen::Quaterniond _r;
    Eigen::Vector3d _t;

public:
    SE3(){
        _r.setIdentity();
        _t.setZero();
    }

    SE3(const Eigen::Matrix3d& R, const Eigen::Vector3d& t):_r(Eigen::Quaterniond(R)),_t(t){
        normalizeRotation();
    }

    SE3(const Eigen::Quaterniond& q, const Eigen::Vector3d& t):_r(q),_t(t){
        normalizeRotation();
    }

    inline const Eigen::Vector3d& translation() const {return _t;}

    inline Eigen::Vector3d& translation() {return _t;}

    inline void setTranslation(const Eigen::Vector3d& t_) {_t = t_;}

    inline const Eigen::Quaterniond& rotation() const {return _r;}

    inline Eigen::Quaterniond& rotation() {return _r;}

    void setRotation(const Eigen::Quaterniond& r_) {_r=r_;}

    inline SE3 operator* (const SE3& tr2) const{
        SE3 result(*this);
        result._t += _r*tr2._t;
        result._r*=tr2._r;
        result.normalizeRotation();
        return result;
    }

    inline SE3& operator*= (const SE3& tr2){
        _t+=_r*tr2._t;
        _r*=tr2._r;
        normalizeRotation();
        return *this;
    }

    inline Eigen::Vector3d operator* (const Eigen::Vector3d& v) const {
        return _t+_r*v;
    }

    inline SE3 inverse() const{
        SE3 ret;
        ret._r=_r.conjugate();
        ret._t=ret._r*(_t*-1.);
        return ret;
    }

    inline double operator [](int i) const {
        assert(i<7);
        if (i<4)
            return _r.coeffs()[i];
        return _t[i-4];
    }


    inline Vector7d toVector() const{
        Vector7d v;
        v.head<4>() = Eigen::Vector4d(_r.coeffs());
        v.tail<3>() = _t;
        return v;
    }

    inline void fromVector(const Vector7d& v){
        _r=Eigen::Quaterniond(v[3], v[0], v[1], v[2]);
        _t=Eigen::Vector3d(v[4], v[5], v[6]);
    }


    Vector6d log() const {
        Vector6d res;

        double theta;
        res.head<3>() = toAngleAxis(_r, &theta);

        Eigen::Matrix3d Omega = skew(res.head<3>());
        Eigen::Matrix3d V_inv;
        if (theta<SMALL_EPS)
        {
            V_inv = Eigen::Matrix3d::Identity()- 0.5*Omega + (1./12.)*(Omega*Omega);
        }
        else
        {
            V_inv = ( Eigen::Matrix3d::Identity() - 0.5*Omega
                      + ( 1-theta/(2*tan(theta/2)))/(theta*theta)*(Omega*Omega) );
        }

        res.tail<3>() = V_inv*_t;

        return res;
    }

    Eigen::Vector3d map(const Eigen::Vector3d & xyz) const
    {
        return _r*xyz + _t;
    }


    static SE3 exp(const Vector6d & update)
    {
        Eigen::Vector3d omega(update.data());
        Eigen::Vector3d upsilon(update.data()+3);

        double theta;
        Eigen::Matrix3d Omega = skew(omega);

        Eigen::Quaterniond R = toQuaterniond(omega, &theta);
        Eigen::Matrix3d V;
        if (theta<SMALL_EPS)
        {
            V = R.matrix();
        }
        else
        {
            Eigen::Matrix3d Omega2 = Omega*Omega;

            V = (Eigen::Matrix3d::Identity()
                 + (1-cos(theta))/(theta*theta)*Omega
                 + (theta-sin(theta))/(pow(theta,3))*Omega2);
        }
        return SE3(R, V*upsilon);
    }

    Eigen::Matrix<double, 6, 6, Eigen::ColMajor> adj() const
    {
        Eigen::Matrix3d R = _r.toRotationMatrix();
        Eigen::Matrix<double, 6, 6, Eigen::ColMajor> res;
        res.block(0,0,3,3) = R;
        res.block(3,3,3,3) = R;
        res.block(3,0,3,3) = skew(_t)*R;
        res.block(0,3,3,3) = Eigen::Matrix3d::Zero(3,3);
        return res;
    }

    Eigen::Matrix<double,4,4,Eigen::ColMajor> to_homogeneous_matrix() const
    {
        Eigen::Matrix<double,4,4,Eigen::ColMajor> homogeneous_matrix;
        homogeneous_matrix.setIdentity();
        homogeneous_matrix.block(0,0,3,3) = _r.toRotationMatrix();
        homogeneous_matrix.col(3).head(3) = translation();

        return homogeneous_matrix;
    }

    void normalizeRotation(){
        if (_r.w()<0){
            _r.coeffs() *= -1;
        }
        _r.normalize();
    }
};

#endif // SE3_HPP_
