#ifndef BAPROBLEM_HPP_
#define BAPROBLEM_HPP_

#include "parametersse3.hpp"

class Sample
{
public:
    static int uniform(int from, int to);
    static double uniform();
    static double gaussian(double sigma);
};

template <int PoseBlockSize>
class BAProblem
{
public:
    BAProblem(int pose_num_, int point_num_, double pix_noise_, bool useOrdering = false);

    void solve(ceres::Solver::Options opt, ceres::Solver::Summary* sum);

    ceres::Problem problem;
    ceres::ParameterBlockOrdering* ordering = NULL;

protected:
    PosePointParametersBlock<PoseBlockSize> states;
    PosePointParametersBlock<PoseBlockSize> true_states;

};


template<int PoseBlockSize>
BAProblem<PoseBlockSize>::BAProblem(int pose_num_, int point_num_, double pix_noise_, bool useOrdering)
{
    if(useOrdering)
        ordering = new ceres::ParameterBlockOrdering;

    int pose_num = 15;
    int point_num = 300;
    double PIXEL_NOISE = pix_noise_;

    states.create(pose_num, point_num);
    true_states.create(pose_num, point_num);

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

    for (int i = 0; i < pose_num; ++i)
    {
        Vector3d trans(i * 0.04 - 1., 0, 0);

        Eigen::Quaterniond q;
        q.setIdentity();
        if(PoseBlockSize == 7)
        {
            Eigen::Map<Vector7d> true_pose(true_states.pose(i));
            true_pose.head<4>() = Eigen::Vector4d(q.coeffs());
            true_pose.tail<3>() = trans;

            Eigen::Map<Vector7d> pose(states.pose(i));
            pose = true_pose;
        }
        else if(PoseBlockSize == 6)
        {
            Eigen::Map<Vector6d> true_pose(true_states.pose(i));
            true_pose.head<3>() = toAngleAxis(q);
            true_pose.tail<3>() = trans;

            Eigen::Map<Vector6d> pose(states.pose(i));
            pose = true_pose;
        }


        problem.AddParameterBlock(states.pose(i), PoseBlockSize, new PoseSE3Parameterization<PoseBlockSize>());

        if(i < 2)
        {
            problem.SetParameterBlockConstant(states.pose(i));
        }
    }

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
            if(PoseBlockSize == 7)
                true_pose_se3.fromVector(Eigen::Map<Vector7d>(true_states.pose(j)));
            else if(PoseBlockSize == 6)
            {
                true_pose_se3.setRotation(toQuaterniond(Vector3d(true_states.pose(j))));
                true_pose_se3.setTranslation(Vector3d(true_states.pose(j)+3));
            }
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
            if(useOrdering)
                ordering->AddElementToGroup(states.point(i), 0);

            for (int j = 0; j < pose_num; ++j)
            {
                if(PoseBlockSize == 7)
                    true_pose_se3.fromVector(Eigen::Map<Vector7d>(true_states.pose(j)));
                else if(PoseBlockSize == 6)
                {
                    true_pose_se3.setRotation(toQuaterniond(Vector3d(true_states.pose(j))));
                    true_pose_se3.setTranslation(Vector3d(true_states.pose(j)+3));
                }
                Vector3d point_cam = true_pose_se3.map(true_point_i);
                z = cam.cam_map(point_cam);

                if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480)
                {
                    z += Vector2d(Sample::gaussian(PIXEL_NOISE),
                                  Sample::gaussian(PIXEL_NOISE));

                    ceres::CostFunction* costFunc = new ReprojectionErrorSE3XYZ<PoseBlockSize>(focal_length, cx, cy, z[0], z[1]);
                    problem.AddResidualBlock(costFunc, NULL, states.pose(j), states.point(i));
                }
            }

        }
    }

    for (int i = 0; i < pose_num; ++i)
    {
        if(useOrdering)
            ordering->AddElementToGroup(states.pose(i), 1);
    }

}


template<int PoseBlockSize>
void BAProblem<PoseBlockSize>::solve(ceres::Solver::Options opt, ceres::Solver::Summary *sum)
{
    if(ordering != NULL)
        opt.linear_solver_ordering.reset(ordering);
    ceres::Solve(opt, &problem, sum);
}



#endif
