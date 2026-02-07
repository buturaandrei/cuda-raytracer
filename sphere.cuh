/**
 * sphere.cuh - Geometria della Sfera per Ray Tracing
 */

#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "vec3.cuh"
#include "ray.cuh"
#include "materials.cuh"

struct Sphere {
    Vec3 center;
    float radius;
    Material material;

    __host__ __device__ Sphere() {}
    __host__ __device__ Sphere(Vec3 c, float r, Material m) : center(c), radius(r), material(m) {}
    
    __host__ __device__ bool intersect(const Ray& ray, float t_min, float t_max, float& t) const {
        Vec3 oc = ray.origin - center;
        float a = ray.direction.dot(ray.direction);
        float b = 2.0f * oc.dot(ray.direction);
        float c = oc.dot(oc) - radius * radius;
        float disc = b * b - 4.0f * a * c;
        if (disc < 0) return false;

        float sqrtd = sqrtf(disc);
        float root = (-b - sqrtd) / (2.0f * a);
        if (root < t_min || root > t_max) {
            root = (-b + sqrtd) / (2.0f * a);
            if (root < t_min || root > t_max) return false;
        }
        t = root;
        return true;
    }

    __host__ __device__ Vec3 getNormal(const Vec3& p) const { return (p - center).normalize(); }
};

#endif // SPHERE_CUH
