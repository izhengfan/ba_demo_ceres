#include <Eigen/StdVector>
#include <iostream>
#include <stdint.h>

#include <unordered_set>
#include <unordered_map>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace Eigen;
using namespace std;

typedef Eigen::Matrix<double, 6, 1> Vector6d;

class ReprojectionErrorSE3XYZ: public ceres::SizedCostFunction<2, 7, 3>
{
public:
    ReprojectionErrorSE3XYZ(double observation_x, double observation_y)
        : _observation_x(observation_x), _observation_y(observation_y) {}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;


    /// Unnecessary operator
    /*
    template <typename T>
    bool operator()(const T *const camera, const T *const point, T *residuals) const
    {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);

        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        residuals[0] = f * p[0] / p[2] + cx - T(_observation_x);
        residuals[1] = f * p[1] / p[2] + cy - T(_observation_y);
    }
    */

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
    J_cam << f_by_z, 0, - f_by_z * p[0] / z,
            0, f_by_z, - f_by_z * p[1] / z;
    if(jacobians != NULL && jacobians[0] != NULL && jacobians[1] != NULL)
    {
        Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3(jacobians[0]);
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J_point(jacobians[1]);
        J_se3.Zero();
        J_se3.block<2,3>(0,0) = - J_cam * skew(p);
        J_se3.block<2,3>(0,3) = J_cam;
        J_point = J_cam * q.toRotationMatrix();
    }

    return true;
}


class SE3Parameterization : public ceres::LocalParameterization {
public:
    SE3Parameterization() {}
    virtual ~SE3Parameterization() {}
    virtual bool Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const;
    virtual bool ComputeJacobian(const double* x,
                                 double* jacobian) const;
    virtual int GlobalSize() const { return 7; }
    virtual int LocalSize() const { return 6; }
};

class Sample
{
public:
    static int uniform(int from, int to);
    static double uniform();
    static double gaussian(double sigma);
};

static double uniform_rand(double lowerBndr, double upperBndr)
{
    return lowerBndr + ((double)std::rand() / (RAND_MAX + 1.0)) * (upperBndr - lowerBndr);
}

static double gauss_rand(double mean, double sigma)
{
    double x, y, r2;
    do
    {
        x = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
        y = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
        r2 = x * x + y * y;
    } while (r2 > 1.0 || r2 == 0.0);
    return mean + sigma * y * std::sqrt(-2.0 * log(r2) / r2);
}

int Sample::uniform(int from, int to)
{
    return static_cast<int>(uniform_rand(from, to));
}

double Sample::uniform()
{
    return uniform_rand(0., 1.);
}

double Sample::gaussian(double sigma)
{
    return gauss_rand(0., sigma);
}

int main(int argc, const char *argv[])
{
    if (argc < 2)
    {
        cout << endl;
        cout << "Please type: " << endl;
        cout << "ba_demo [PIXEL_NOISE] [OUTLIER RATIO] [ROBUST_KERNEL] [STRUCTURE_ONLY] [DENSE]" << endl;
        cout << endl;
        cout << "PIXEL_NOISE: noise in image space (E.g.: 1)" << endl;
        cout << "OUTLIER_RATIO: probability of spuroius observation  (default: 0.0)" << endl;
        cout << "ROBUST_KERNEL: use robust kernel (0 or 1; default: 0==false)" << endl;
        cout << "STRUCTURE_ONLY: performe structure-only BA to get better point initializations (0 or 1; default: 0==false)" << endl;
        cout << "DENSE: Use dense solver (0 or 1; default: 0==false)" << endl;
        cout << endl;
        cout << "Note, if OUTLIER_RATIO is above 0, ROBUST_KERNEL should be set to 1==true." << endl;
        cout << endl;
        exit(0);
    }

    double PIXEL_NOISE = atof(argv[1]);
    double OUTLIER_RATIO = 0.0;

    if (argc > 2)
    {
        OUTLIER_RATIO = atof(argv[2]);
    }

    bool ROBUST_KERNEL = false;
    if (argc > 3)
    {
        ROBUST_KERNEL = atoi(argv[3]) != 0;
    }
    bool STRUCTURE_ONLY = false;
    if (argc > 4)
    {
        STRUCTURE_ONLY = atoi(argv[4]) != 0;
    }

    bool DENSE = false;
    if (argc > 5)
    {
        DENSE = atoi(argv[5]) != 0;
    }

    cout << "PIXEL_NOISE: " << PIXEL_NOISE << endl;
    cout << "OUTLIER_RATIO: " << OUTLIER_RATIO << endl;
    cout << "ROBUST_KERNEL: " << ROBUST_KERNEL << endl;
    cout << "STRUCTURE_ONLY: " << STRUCTURE_ONLY << endl;
    cout << "DENSE: " << DENSE << endl;

    // g2o::SparseOptimizer optimizer;
    // optimizer.setVerbose(false);
    // std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    // if (DENSE)
    // {
    //   linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    // }
    // else
    // {
    //   linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
    // }

    // g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
    //     g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
    // optimizer.setAlgorithm(solver);

    vector<Vector3d> true_points;
    for (size_t i = 0; i < 500; ++i)
    {
        true_points.push_back(Vector3d((Sample::uniform() - 0.5) * 3,
                                       Sample::uniform() - 0.5,
                                       Sample::uniform() + 3));
    }

    double focal_length = 1000.;
    Vector2d principal_point(320., 240.);

    vector<Vector6d, aligned_allocator<Vector6d>> true_poses;

    int vertex_id = 0;
    for (size_t i = 0; i < 15; ++i)
    {
        Vector3d trans(i * 0.04 - 1., 0, 0);

        Eigen::Quaterniond q;
        q.setIdentity();
        // g2o::SE3Quat pose(q, trans);
        // g2o::VertexSE3Expmap *v_se3 = new g2o::VertexSE3Expmap();
        Vector6d pose = Vector6d::Zero();
        pose.tail<3>() = trans;

        // v_se3->setId(vertex_id);
        // if (i < 2)
        // {
        //   v_se3->setFixed(true);
        // }
        // v_se3->setEstimate(pose);
        // optimizer.addVertex(v_se3);
        true_poses.push_back(pose);
        vertex_id++;
    }
    int point_id = vertex_id;
    int point_num = 0;
    double sum_diff2 = 0;

    cout << endl;
    unordered_map<int, int> pointid_2_trueid;
    unordered_set<int> inliers;

    for (size_t i = 0; i < true_points.size(); ++i)
    {
        g2o::VertexSBAPointXYZ *v_p = new g2o::VertexSBAPointXYZ();
        v_p->setId(point_id);
        v_p->setMarginalized(true);
        v_p->setEstimate(true_points.at(i) + Vector3d(Sample::gaussian(1),
                                                      Sample::gaussian(1),
                                                      Sample::gaussian(1)));
        int num_obs = 0;
        for (size_t j = 0; j < true_poses.size(); ++j)
        {
            Vector2d z = cam_params->cam_map(true_poses.at(j).map(true_points.at(i)));
            if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480)
            {
                ++num_obs;
            }
        }
        if (num_obs >= 2)
        {
            optimizer.addVertex(v_p);
            bool inlier = true;
            for (size_t j = 0; j < true_poses.size(); ++j)
            {
                Vector2d z = cam_params->cam_map(true_poses.at(j).map(true_points.at(i)));

                if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480)
                {
                    double sam = Sample::uniform();
                    if (sam < OUTLIER_RATIO)
                    {
                        z = Vector2d(Sample::uniform(0, 640),
                                     Sample::uniform(0, 480));
                        inlier = false;
                    }
                    z += Vector2d(Sample::gaussian(PIXEL_NOISE),
                                  Sample::gaussian(PIXEL_NOISE));
                    g2o::EdgeProjectXYZ2UV *e = new g2o::EdgeProjectXYZ2UV();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(v_p));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertices().find(j)->second));
                    e->setMeasurement(z);
                    e->information() = Matrix2d::Identity();
                    if (ROBUST_KERNEL)
                    {
                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                    }
                    e->setParameterId(0, 0);
                    optimizer.addEdge(e);
                }
            }

            if (inlier)
            {
                inliers.insert(point_id);
                Vector3d diff = v_p->estimate() - true_points[i];

                sum_diff2 += diff.dot(diff);
            }
            pointid_2_trueid.insert(make_pair(point_id, i));
            ++point_id;
            ++point_num;
        }
    }
    cout << endl;
    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    if (STRUCTURE_ONLY)
    {
        g2o::StructureOnlySolver<3> structure_only_ba;
        cout << "Performing structure-only BA:" << endl;
        g2o::OptimizableGraph::VertexContainer points;
        for (g2o::OptimizableGraph::VertexIDMap::const_iterator it = optimizer.vertices().begin(); it != optimizer.vertices().end(); ++it)
        {
            g2o::OptimizableGraph::Vertex *v = static_cast<g2o::OptimizableGraph::Vertex *>(it->second);
            if (v->dimension() == 3)
                points.push_back(v);
        }
        structure_only_ba.calc(points, 10);
    }
    //optimizer.save("test.g2o");
    cout << endl;
    cout << "Performing full BA:" << endl;
    optimizer.optimize(10);
    cout << endl;
    cout << "Point error before optimisation (inliers only): " << sqrt(sum_diff2 / point_num) << endl;
    point_num = 0;
    sum_diff2 = 0;
    for (unordered_map<int, int>::iterator it = pointid_2_trueid.begin();
         it != pointid_2_trueid.end(); ++it)
    {
        g2o::HyperGraph::VertexIDMap::iterator v_it = optimizer.vertices().find(it->first);
        if (v_it == optimizer.vertices().end())
        {
            cerr << "Vertex " << it->first << " not in graph!" << endl;
            exit(-1);
        }
        g2o::VertexSBAPointXYZ *v_p = dynamic_cast<g2o::VertexSBAPointXYZ *>(v_it->second);
        if (v_p == 0)
        {
            cerr << "Vertex " << it->first << "is not a PointXYZ!" << endl;
            exit(-1);
        }
        Vector3d diff = v_p->estimate() - true_points[it->second];
        if (inliers.find(it->first) == inliers.end())
            continue;
        sum_diff2 += diff.dot(diff);
        ++point_num;
    }
    cout << "Point error after optimisation (inliers only): " << sqrt(sum_diff2 / point_num) << endl;
    cout << endl;
}
