//
// Created by LEI XU on 4/27/19.
//

#include "Texture.hpp"

Eigen::Vector3f lerp(float x, const Eigen::Vector3f& a, const Eigen::Vector3f& b)
{
    return a + x * (b - a);
}

Eigen::Vector3f Texture::getImgColor(int u, int v)
{
    auto color = image_data.at<cv::Vec3b>(v, u);
    return Eigen::Vector3f(color[0], color[1], color[2]);
}

Eigen::Vector3f Texture::getColorBilinear(float u, float v)
{
    if (u < 0) u = 0;
    if (v < 0) v = 0;

    u *= width;
    v = (1 - v) * height;

    int u_floor = std::floor(u);
    int u_ceil = std::ceil(u);
    int v_floor = std::floor(v);
    int v_ceil = std::ceil(v);

    float s = u - u_floor;
    float t = v - v_floor;

    Eigen::Vector3f cHorizontal1 = lerp(s, getImgColor(u_floor, v_floor), getImgColor(u_ceil, v_floor));
    Eigen::Vector3f cHorizontal2 = lerp(s, getImgColor(u_floor, v_ceil), getImgColor(u_ceil, v_ceil));
    Eigen::Vector3f color = lerp(t, cHorizontal1, cHorizontal2);

    return color;
}

