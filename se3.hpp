#ifndef SE3_HPP_
#define SE3_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>


typedef Eigen::Matrix<double, 6, 1, Eigen::ColMajor> Vector6d;
typedef Eigen::Matrix<double, 7, 1, Eigen::ColMajor> Vector7d;


class SE3 {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    static Eigen::Matrix3d skew(const Eigen::Vector3d&v)
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

    static Eigen::Vector3d deltaR(const Eigen::Matrix3d& R)
    {
        Eigen::Vector3d v;
        v(0)=R(2,1)-R(1,2);
        v(1)=R(0,2)-R(2,0);
        v(2)=R(1,0)-R(0,1);
        return v;
    }

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

    inline void setTranslation(const Eigen::Vector3d& t_) {_t = t_;}

    inline const Eigen::Quaterniond& rotation() const {return _r;}

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
        Eigen::Matrix3d _R = _r.toRotationMatrix();
        double d =  0.5*(_R(0,0)+_R(1,1)+_R(2,2)-1);
        Eigen::Vector3d omega;
        Eigen::Vector3d upsilon;


        Eigen::Vector3d dR = deltaR(_R);
        Eigen::Matrix3d V_inv;

        if (d>0.99999)
        {

            omega=0.5*dR;
            Eigen::Matrix3d Omega = skew(omega);
            V_inv = Eigen::Matrix3d::Identity()- 0.5*Omega + (1./12.)*(Omega*Omega);
        }
        else
        {
            double theta = acos(d);
            omega = theta/(2*sqrt(1-d*d))*dR;
            Eigen::Matrix3d Omega = skew(omega);
            V_inv = ( Eigen::Matrix3d::Identity() - 0.5*Omega
                      + ( 1-theta/(2*tan(theta/2)))/(theta*theta)*(Omega*Omega) );
        }

        upsilon = V_inv*_t;
        for (int i=0; i<3;i++){
            res[i]=omega[i];
        }
        for (int i=0; i<3;i++){
            res[i+3]=upsilon[i];
        }

        return res;

    }

    Eigen::Vector3d map(const Eigen::Vector3d & xyz) const
    {
        return _r*xyz + _t;
    }


    static SE3 exp(const Vector6d & update)
    {
        Eigen::Vector3d omega;
        for (int i=0; i<3; i++)
            omega[i]=update[i];
        Eigen::Vector3d upsilon;
        for (int i=0; i<3; i++)
            upsilon[i]=update[i+3];

        double theta = omega.norm();
        Eigen::Matrix3d Omega = skew(omega);

        Eigen::Matrix3d R;
        Eigen::Matrix3d V;
        if (theta<0.00001)
        {
            //TODO: CHECK WHETHER THIS IS CORRECT!!!
            R = (Eigen::Matrix3d::Identity() + Omega + Omega*Omega);

            V = R;
        }
        else
        {
            Eigen::Matrix3d Omega2 = Omega*Omega;

            R = (Eigen::Matrix3d::Identity()
                 + sin(theta)/theta *Omega
                 + (1-cos(theta))/(theta*theta)*Omega2);

            V = (Eigen::Matrix3d::Identity()
                 + (1-cos(theta))/(theta*theta)*Omega
                 + (theta-sin(theta))/(pow(theta,3))*Omega2);
        }
        return SE3(Eigen::Quaterniond(R),V*upsilon);
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

    /**
       * cast SE3 into an Isometry3D
       */
    operator Isometry3D() const
    {
        Isometry3D result = (Isometry3D) rotation();
        result.translation() = translation();
        return result;
    }
};

#endif // SE3_HPP_
