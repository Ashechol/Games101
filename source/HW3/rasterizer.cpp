//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_normals(const std::vector<Eigen::Vector3f>& normals)
{
    auto id = get_next_id();
    nor_buf.emplace(id, normals);

    normal_id = id;

    return {id};
}


// Bresenham's line drawing algorithm
void rst::rasterizer::draw_line(Eigen::Vector3f begin, Eigen::Vector3f end)
{
    auto x1 = begin.x();
    auto y1 = begin.y();
    auto x2 = end.x();
    auto y2 = end.y();

    Eigen::Vector3f line_color = {255, 255, 255};

    int x,y,dx,dy,dx1,dy1,px,py,xe,ye,i;

    dx=x2-x1;
    dy=y2-y1;
    dx1=fabs(dx);
    dy1=fabs(dy);
    px=2*dy1-dx1;
    py=2*dx1-dy1;

    if(dy1<=dx1)
    {
        if(dx>=0)
        {
            x=x1;
            y=y1;
            xe=x2;
        }
        else
        {
            x=x2;
            y=y2;
            xe=x1;
        }
        Eigen::Vector2i point = Eigen::Vector2i(x, y);
        set_pixel(point,line_color);
        for(i=0;x<xe;i++)
        {
            x=x+1;
            if(px<0)
            {
                px=px+2*dy1;
            }
            else
            {
                if((dx<0 && dy<0) || (dx>0 && dy>0))
                {
                    y=y+1;
                }
                else
                {
                    y=y-1;
                }
                px=px+2*(dy1-dx1);
            }
//            delay(0);
            Eigen::Vector2i point = Eigen::Vector2i(x, y);
            set_pixel(point,line_color);
        }
    }
    else
    {
        if(dy>=0)
        {
            x=x1;
            y=y1;
            ye=y2;
        }
        else
        {
            x=x2;
            y=y2;
            ye=y1;
        }
        Eigen::Vector2i point = Eigen::Vector2i(x, y);
        set_pixel(point,line_color);
        for(i=0;y<ye;i++)
        {
            y=y+1;
            if(py<=0)
            {
                py=py+2*dx1;
            }
            else
            {
                if((dx<0 && dy<0) || (dx>0 && dy>0))
                {
                    x=x+1;
                }
                else
                {
                    x=x-1;
                }
                py=py+2*(dx1-dy1);
            }
//            delay(0);
            Eigen::Vector2i point = Eigen::Vector2i(x, y);
            set_pixel(point,line_color);
        }
    }
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

static bool insideTriangle(int x, int y, const Vector4f* _v){
    Vector3f v[3];
    for(int i=0;i<3;i++)
        v[i] = {_v[i].x(),_v[i].y(), 1.0};
    Vector3f f0,f1,f2;
    f0 = v[1].cross(v[0]);
    f1 = v[2].cross(v[1]);
    f2 = v[0].cross(v[2]);
    Vector3f p(x,y,1.);
    if((p.dot(f0)*f0.dot(v[2])>0) && (p.dot(f1)*f1.dot(v[0])>0) && (p.dot(f2)*f2.dot(v[1])>0))
        return true;
    return false;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector4f* v){
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(std::vector<Triangle *> &TriangleList) {

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (const auto& t:TriangleList)
    {
        Triangle newtri = *t;

        std::array<Eigen::Vector4f, 3> mm {
                (view * model * t->v[0]),
                (view * model * t->v[1]),
                (view * model * t->v[2])
        };

        std::array<Eigen::Vector3f, 3> viewspace_pos;

        std::transform(mm.begin(), mm.end(), viewspace_pos.begin(), [](auto& v) {
            return v.template head<3>();
        });

        Eigen::Vector4f v[] = {
                mvp * t->v[0],
                mvp * t->v[1],
                mvp * t->v[2]
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec.x()/=vec.w();
            vec.y()/=vec.w();
            vec.z()/=vec.w();
        }

        Eigen::Matrix4f inv_trans = (view * model).inverse().transpose();
        Eigen::Vector4f n[] = {
                inv_trans * to_vec4(t->normal[0], 0.0f),
                inv_trans * to_vec4(t->normal[1], 0.0f),
                inv_trans * to_vec4(t->normal[2], 0.0f)
        };

        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = -vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            //screen space coordinates
            newtri.setVertex(i, v[i]);
        }

        for (int i = 0; i < 3; ++i)
        {
            //view space normal
            newtri.setNormal(i, n[i].head<3>());
        }

        newtri.setColor(0, 148,121.0,92.0);
        newtri.setColor(1, 148,121.0,92.0);
        newtri.setColor(2, 148,121.0,92.0);

        // Also pass view space vertice position
        rasterize_triangle(newtri, viewspace_pos);
    }
}

static Eigen::Vector3f interpolate(float alpha, float beta, float gamma, const Eigen::Vector3f& vert1, const Eigen::Vector3f& vert2, const Eigen::Vector3f& vert3, float weight)
{
    return (alpha * vert1 + beta * vert2 + gamma * vert3) / weight;
}

static Eigen::Vector2f interpolate(float alpha, float beta, float gamma, const Eigen::Vector2f& vert1, const Eigen::Vector2f& vert2, const Eigen::Vector2f& vert3, float weight)
{
    auto u = (alpha * vert1[0] + beta * vert2[0] + gamma * vert3[0]);
    auto v = (alpha * vert1[1] + beta * vert2[1] + gamma * vert3[1]);

    u /= weight;
    v /= weight;

    return Eigen::Vector2f(u, v);
}

// perspective-correct interpolation
template <typename T>
T interpolate(std::tuple<float, float, float>baryCoords, const float* w, float reverse, const T* attribs)
{
    T interpolated = std::get<0>(baryCoords) * attribs[0] / w[0] +
                     std::get<1>(baryCoords) * attribs[1] / w[1] +
                     std::get<2>(baryCoords) * attribs[2] / w[2];

    return interpolated * reverse;
}

std::tuple<float, float, float> find_nearest(float x, float y, const Vector4f (&v)[3])
{
    auto [alpha, beta, gamma] = computeBarycentric2D(x + .5f, y + .5f, v);

    if (alpha >= 0 && beta >= 0 && gamma >= 0)
        return std::tuple<float, float, float>{alpha, beta, gamma};

    if (alpha < 0) return computeBarycentric2D(v[0].x(), v[0].y(), v);
    if (beta < 0) return computeBarycentric2D(v[1].x(), v[1].y(), v);

    return computeBarycentric2D(v[2].x(), v[2].y(), v);
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t, const std::array<Eigen::Vector3f, 3>& view_pos) 
{
    const Vector4f (&v)[3] = t.v;
    const float depth[3] = {v[0].z(), v[1].z(), v[2].z()};
    const float w[3] = {v[0].w(), v[1].w(), v[2].w()};

    int xMin = floor(std::min(v[0].x(), std::min(v[1].x(), v[2].x())));
    int xMax = ceil(std::max(v[0].x(), std::max(v[1].x(), v[2].x())));
    int yMin = floor(std::min(v[0].y(), std::min(v[1].y(), v[2].y())));
    int yMax = ceil(std::max(v[0].y(), std::max(v[1].y(), v[2].y())));

    Vector2f x4[4] = {{0.25, 0.25},
                      {0.25, 0.75},
                      {0.75, 0.25},
                      {0.75, 0.75}};

    for (int i = xMin; i < xMax; i++)
    {
        auto x = static_cast<float>(i);

        for (int j = yMin; j < yMax; j++)
        {
            auto y = static_cast<float>(j);

            bool depth_test = false;

            for (int k = 0; k < 4; k++)
            {
                auto baryCoords = computeBarycentric2D(x + x4[k].x(), y + x4[k].y(), v);
                auto [a, b, c] = baryCoords;

                // vertex should pass the inside test!
                if (a < 0 || b < 0 || c < 0) continue;

                // following code to get the interpolated z value.
                // interpolate depth
                float reverseZ = 1 / (a / v[0].w() + b / v[1].w() + c / v[2].w());
                float interpolate_depth = interpolate(baryCoords, w, reverseZ, depth);

                int index = get_subsample_index(i, j, k);
                if (interpolate_depth < subsample_depth_buf[index])
                {
                    // interpolate attributes
                    Vector3f interpolated_color = interpolate(baryCoords, w, reverseZ, t.color);

                    Vector3f interpolated_normal = interpolate(baryCoords, w, reverseZ, t.normal);

                    Vector2f interpolated_texcoords = interpolate(baryCoords, w, reverseZ, t.tex_coords);

                    Vector3f vp[3]{view_pos[0], view_pos[1], view_pos[2]};
                    Vector3f interpolated_shadingcoords = interpolate(baryCoords, w, reverseZ, vp);

                    fragment_shader_payload payload( interpolated_color,
                                                     interpolated_normal.normalized(),
                                                     interpolated_texcoords,
                                                     texture ? &*texture : nullptr);
                    payload.view_pos = interpolated_shadingcoords;

                    Vector3f color = fragment_shader(payload);

                    subsample_color_buf[index] = color;
                    subsample_depth_buf[index] = interpolate_depth;
                    depth_test = true;
                }
            }

            if (depth_test)
            {
                set_pixel(Vector2i(i, j), get_sample_color(i, j));
            }
        }
    }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
        std::fill(subsample_color_buf.begin(), subsample_color_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
        std::fill(subsample_depth_buf.begin(), subsample_depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
    subsample_color_buf.resize(w * h * 4);
    subsample_depth_buf.resize(w * h * 4);

    texture = std::nullopt;
}

int rst::rasterizer::get_index(int x, int y) const
{
    return (height-y)*width + x;
}

void rst::rasterizer::set_pixel(const Vector2i &point, const Eigen::Vector3f &color)
{
    //old index: auto ind = point.y() + point.x() * width;
    int ind = (height-point.y())*width + point.x();
    frame_buf[ind] = color;
}

void rst::rasterizer::set_vertex_shader(std::function<Eigen::Vector3f(vertex_shader_payload)> vert_shader)
{
    vertex_shader = vert_shader;
}

void rst::rasterizer::set_fragment_shader(std::function<Eigen::Vector3f(fragment_shader_payload)> frag_shader)
{
    fragment_shader = frag_shader;
}

int rst::rasterizer::get_subsample_index(int x, int y, int k) const
{
    return (height - 1- y) * width * 4 + (width - 1 - x) * 4 + k;
}

float rst::rasterizer::get_sample_depth(int x, int y) const
{
    int index = get_subsample_index(x, y, 0);
    float min_depth = std::numeric_limits<float>::infinity();
    for (int i = 0; i < 4; i++)
        min_depth = std::min(min_depth, subsample_depth_buf[index + i]);

    return min_depth;
}

Vector3f rst::rasterizer::get_sample_color(int x, int y) const
{
    int index = get_subsample_index(x, y, 0);
    Vector3f sum{0, 0, 0};
    for (int i = 0; i < 4; i++)
        sum += subsample_color_buf[index + i];

    return sum / 4.0f;
}

