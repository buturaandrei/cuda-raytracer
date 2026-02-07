/**
 * triangle.cuh - Geometria Triangolo per Ray Tracing (Möller-Trumbore)
 */

#ifndef TRIANGLE_CUH
#define TRIANGLE_CUH

#include "vec3.cuh"
#include "ray.cuh"
#include "materials.cuh"

struct Triangle {
    Vec3 v0, v1, v2;
    Vec3 normal;
    Material material;
    
    __host__ __device__ Triangle() {}
    
    __host__ __device__ Triangle(Vec3 vertex0, Vec3 vertex1, Vec3 vertex2, Material mat)
        : v0(vertex0), v1(vertex1), v2(vertex2), material(mat) {
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        normal = edge1.cross(edge2).normalize();
    }
    
    __host__ __device__ Triangle(Vec3 vertex0, Vec3 vertex1, Vec3 vertex2, Vec3 norm, Material mat)
        : v0(vertex0), v1(vertex1), v2(vertex2), normal(norm), material(mat) {}
    
    __host__ __device__ bool intersect(const Ray& ray, float t_min, float t_max, float& t) const {
        const float EPSILON = 0.0000001f;
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        Vec3 h = ray.direction.cross(edge2);
        float a = edge1.dot(h);
        if (a > -EPSILON && a < EPSILON) return false;
        float f = 1.0f / a;
        Vec3 s = ray.origin - v0;
        float u = f * s.dot(h);
        if (u < 0.0f || u > 1.0f) return false;
        Vec3 q = s.cross(edge1);
        float v = f * ray.direction.dot(q);
        if (v < 0.0f || u + v > 1.0f) return false;
        float t_hit = f * edge2.dot(q);
        if (t_hit > t_min && t_hit < t_max) { t = t_hit; return true; }
        return false;
    }
    
    __host__ __device__ Vec3 getNormal(const Vec3& point) const { return normal; }
};

#endif // TRIANGLE_CUH
