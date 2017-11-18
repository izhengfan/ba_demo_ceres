#include "parametersse3.hpp"
#include <glog/logging.h>

using namespace Eigen;
using namespace std;

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

    google::InitGoogleLogging(argv[0]);

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

    ceres::Problem problem;

    PosePointParametersBlock states(15, 50);
    PosePointParametersBlock true_states(15, 50);

    //vector<Vector3d> true_points;
    for (int i = 0; i < 50; ++i)
    {
        Eigen::Map<Vector3d> true_pt(true_states.point(i));
        true_pt = Vector3d((Sample::uniform() - 0.5) * 3,
                           Sample::uniform() - 0.5,
                           Sample::uniform() + 3);
    }

    double focal_length = 1000.;
    double cx = 320.;
    double cy = 240.;
    CameraParameters cam(focal_length, cx, cy);

    int vertex_id = 0;
    for (int i = 0; i < 15; ++i)
    {
        Vector3d trans(i * 0.04 - 1., 0, 0);

        Eigen::Quaterniond q;
        q.setIdentity();
        Eigen::Map<Vector6d> true_pose(true_states.pose(i));
        Sophus::SE3d true_se3;
        true_se3.setQuaternion(q);
        true_se3.translation() = trans;
        true_pose = true_se3.log();

        Eigen::Map<Vector6d> pose(states.pose(i));
        pose = true_pose;

        problem.AddParameterBlock(states.pose(i), 6, new SE3Parameterization());

        if(i < 2)
        {
            problem.SetParameterBlockConstant(states.pose(i));
        }
        vertex_id++;
    }
    int point_id = vertex_id;
    int point_num = 0;
    double sum_diff2 = 0;

    cout << endl;
    unordered_map<int, int> pointid_2_trueid;
    unordered_set<int> inliers;

    for (int i = 0; i < 50; ++i)
    {
        Eigen::Map<Vector3d> true_point_i(true_states.point(i));
        Eigen::Map<Vector3d> noise_point_i(states.point(i));
        noise_point_i = true_point_i + Vector3d(Sample::gaussian(1),
                                                Sample::gaussian(1),
                                                Sample::gaussian(1));


        problem.AddParameterBlock(states.point(i), 3);

        bool inlier = true;
        for (int j = 0; j < 15; ++j)
        {
            SE3 true_pose_se3;
            true_pose_se3.setRotation(toQuaterniond(Vector3d(true_states.pose(j))));
            true_pose_se3.setTranslation(Vector3d(true_states.pose(j)+3));
            Vector3d point_cam = true_pose_se3.map(true_point_i);
            Vector2d z = cam.cam_map(point_cam);


            z += Vector2d(Sample::gaussian(PIXEL_NOISE),
                          Sample::gaussian(PIXEL_NOISE));


            Matrix2d information = Matrix2d::Identity();
            std::shared_ptr<Camera> camera = std::make_shared<Camera>(focal_length, focal_length, cx, cy);
            ceres::CostFunction* costFunc = new reprojectErr(Vector3d(states.point(i)), Vector2d(z), information, camera);

//            ceres::CostFunction* costFunc = new ReprojectionErrorSE3XYZ(focal_length, cx, cy, z[0], z[1]);
            problem.AddResidualBlock(costFunc, NULL, states.pose(j), states.point(i));

        }

        if (inlier)
        {
            inliers.insert(point_id);
            //Vector3d diff = v_p->estimate() - true_points[i];
            Vector3d diff = noise_point_i - true_point_i;

            sum_diff2 += diff.dot(diff);
        }
        pointid_2_trueid.insert(make_pair(point_id, i));
        ++point_id;
        ++point_num;

    }
    cout << endl;
    ceres::Solver::Options options;
//    options.minimizer_type = ceres::TRUST_REGION;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
//    options.trust_region_strategy_type = ceres::DOGLEG;
    options.minimizer_progress_to_stdout = true;
//    options.dogleg_type = ceres::SUBSPACE_DOGLEG;
    options.max_num_iterations = 100;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
}
