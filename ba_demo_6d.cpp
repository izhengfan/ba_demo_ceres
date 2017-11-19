#include "baproblem.hpp"

using namespace Eigen;
using namespace std;

constexpr int USE_POSE_SIZE = 6;

int main(int argc, const char *argv[])
{
    if (argc < 2)
    {
        cout << endl;
        cout << "Please type: " << endl;
        cout << "ba_demo [PIXEL_NOISE] " << endl;
        cout << endl;
        cout << "PIXEL_NOISE: noise in image space (E.g.: 1)" << endl;
        cout << endl;
        exit(0);
    }

    google::InitGoogleLogging(argv[0]);

    double PIXEL_NOISE = atof(argv[1]);

    cout << "PIXEL_NOISE: " << PIXEL_NOISE << endl;

    BAProblem<USE_POSE_SIZE> baProblem(15, 300, PIXEL_NOISE, true);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;
    ceres::Solver::Summary summary;
    baProblem.solve(options, &summary);
    std::cout << summary.BriefReport() << "\n";
}
