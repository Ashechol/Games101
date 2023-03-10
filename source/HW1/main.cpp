#include "Triangle.hpp"
#include "rasterizer.hpp"
#include <Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1,
        -eye_pos[2], 0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_rotation(Vector3f axis, float angle)
{
    float c = cos(angle);
    float s = sin(angle);

    axis.normalize();

    Eigen::Matrix3f n;
    n << 0, -axis.z(), axis.y(),
         axis.z(), 0, -axis.x(),
         -axis.y(), axis.x(), 0;

    Eigen::Matrix3f rot3f = c * Eigen::Matrix3f::Identity() + (1 - c) * axis * axis.transpose() + s * n;

    Eigen::Matrix4f rot4f = Eigen::Matrix4f::Identity();

    rot4f.block<3, 3>(0, 0) = rot3f;

    // std::cout << rot4f << std::endl;

    return rot4f;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    // input of cos and sin function is radian,
    // so we need convert rotation_angle from degree to radian
    rotation_angle *= MY_PI / 180;
    float c = cos(rotation_angle);
    float s = sin(rotation_angle);

    // Rotation matrices
    // Rotate around X-axis
    // model << 1, 0, 0, 0,
    //          0, c, -s, 0,
    //          0, s, c, 0,
    //          0, 0, 0, 1;

    // Rotate around Y-axis
    // model << c, 0, s, 0,
    //          0, 1, 0, 0,
    //          -s, 0, c, 0,
    //          0, 0, 0, 1;

    // Rotate around Z-axis
    // model << c, -s, 0, 0,
    //          s, c, 0, 0,
    //          0, 0, 1, 0,
    //          0, 0, 0, 1;

    Vector3f axis(3, 3, 0);
    model = get_rotation(axis, rotation_angle);


    return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float dNear, float dFar)
{
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    // dNear and dFar are positive
    float zNear = -dNear, zFar = -dFar;

    // From perspective to orthographic projection
    Eigen::Matrix4f frustum;
    frustum << zNear, 0, 0, 0,
             0, zNear, 0, 0,
             0, 0, zNear+zFar, -zNear * zFar,
             0, 0, 1, 0;

    eye_fov *= MY_PI / 180;
    float height = abs(zNear) * tan(eye_fov * 0.5f) * 2;
    float width = height * aspect_ratio;

    // Orthographic projection
    Eigen::Matrix4f ortho;
    ortho << 2 / width, 0, 0,0,
            0, 2 / height, 0, 0,
            0, 0, 2 / (zNear - zFar), (zNear + zFar) / (zFar-zNear),
            0, 0, 0, 1;

    projection = ortho * frustum;

    return projection;
}

int main(int argc, const char** argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3) {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {
            filename = std::string(argv[3]);
        }
        else
            return 0;
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0, 0, 5};

    std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};

    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;
    int frame_count = 0;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        // std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a') {
            angle += 10;
        }
        else if (key == 'd') {
            angle -= 10;
        }

        // angle -= 10;
    }

    return 0;
}
