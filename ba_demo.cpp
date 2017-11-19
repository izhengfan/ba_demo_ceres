#include "parametersse3quat.hpp"

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

    int pose_num = 15;
    int point_num = 300;

    PosePointParametersBlock states(pose_num, point_num);
    PosePointParametersBlock true_states(pose_num, point_num);

    //vector<Vector3d> true_points;
    for (int i = 0; i < point_num; ++i)
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
    for (int i = 0; i < pose_num; ++i)
    {
        Vector3d trans(i * 0.04 - 1., 0, 0);

        Eigen::Quaterniond q;
        q.setIdentity();
        Eigen::Map<Vector7d> true_pose(true_states.pose(i));
        true_pose.head<4>() = Eigen::Vector4d(q.coeffs());
        true_pose.tail<3>() = trans;

        Eigen::Map<Vector7d> pose(states.pose(i));
        pose = true_pose;

        problem.AddParameterBlock(states.pose(i), 7, new PoseSE3Parameterization);

        if(i < 2)
        {
            problem.SetParameterBlockConstant(states.pose(i));
        }
        vertex_id++;
    }
    int point_id = vertex_id;
    double sum_diff2 = 0;

    cout << endl;
    unordered_map<int, int> pointid_2_trueid;
    unordered_set<int> inliers;

    for (int i = 0; i < point_num; ++i)
    {
        Eigen::Map<Vector3d> true_point_i(true_states.point(i));
        Eigen::Map<Vector3d> noise_point_i(states.point(i));
        noise_point_i = true_point_i + Vector3d(Sample::gaussian(1),
                                                Sample::gaussian(1),
                                                Sample::gaussian(1));

        Vector2d z;
        SE3 true_pose_se3;

        int num_obs = 0;
        for (int j = 0; j < pose_num; ++j)
        {
            true_pose_se3.fromVector(Eigen::Map<Vector7d>(true_states.pose(j)));
            Vector3d point_cam = true_pose_se3.map(true_point_i);
            z = cam.cam_map(point_cam);
            if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480)
            {
                ++num_obs;
            }
        }
        if (num_obs >= 2)
        {
            problem.AddParameterBlock(states.point(i), 3);

            bool inlier = true;
            for (int j = 0; j < pose_num; ++j)
            {
                true_pose_se3.fromVector(Eigen::Map<Vector7d>(true_states.pose(j)));
                Vector3d point_cam = true_pose_se3.map(true_point_i);
                z = cam.cam_map(point_cam);

                if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480)
                {
                    z += Vector2d(Sample::gaussian(PIXEL_NOISE),
                                  Sample::gaussian(PIXEL_NOISE));

                    ceres::CostFunction* costFunc = new ReprojectionErrorSE3XYZ(focal_length, cx, cy, z[0], z[1]);
                    problem.AddResidualBlock(costFunc, NULL, states.pose(j), states.point(i));
                }
            }

            if (inlier)
            {
                inliers.insert(point_id);
                Vector3d diff = noise_point_i - true_point_i;

                sum_diff2 += diff.dot(diff);
            }
            pointid_2_trueid.insert(make_pair(point_id, i));
            ++point_id;
        }
    }
    cout << endl;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
}
