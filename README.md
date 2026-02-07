# CUDA Real-Time Raytracer

A GPU-accelerated real-time raytracer built with **CUDA** and **OpenGL**, featuring BVH acceleration, OBJ mesh loading, reflections, and interactive camera controls.

![CUDA](https://img.shields.io/badge/CUDA-13.1-green)
![OpenGL](https://img.shields.io/badge/OpenGL-4.x-blue)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)

## Features

- **Real-time GPU Raytracing** - Parallel computation on NVIDIA GPUs using CUDA
- **BVH Acceleration** - Grid-based Bounding Volume Hierarchy for fast ray-triangle intersection
- **OBJ Mesh Loading** - Support for 3D models in Wavefront OBJ format
- **Interactive Camera** - FPS-style controls with mouse look
- **Material System** - Diffuse, reflective, and emissive materials
- **Tone Mapping & Gamma Correction** - HDR to LDR conversion with Reinhard operator
- **Debug Overlay** - Real-time GPU info, FPS counter, and camera position
- **CPU Fallback** - Compare GPU vs CPU rendering performance

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         GPU                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │ CUDA Kernel  │ ──→  │     PBO      │ ──→  │  Texture  │ │
│  │ (ray trace)  │      │ (buffer)     │      │  OpenGL   │ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
└─────────────────────────────────────────────────────────────┘
                                                      │
                                                      ↓
                                                 ┌─────────┐
                                                 │ Display │
                                                 └─────────┘
```

The raytracer uses **CUDA-OpenGL interop** for zero-copy rendering:
- CUDA writes directly to a Pixel Buffer Object (PBO)
- OpenGL displays the texture without CPU-GPU transfer overhead

## Controls

| Key | Action |
|-----|--------|
| **WASD** | Move camera |
| **Space/Shift** | Move up/down |
| **Ctrl** | Fast movement |
| **Mouse** | Look around (click to capture) |
| **TAB** | Toggle CPU/GPU rendering |
| **F1** | Toggle debug overlay |
| **F2** | Toggle 3D mesh visibility |
| **F3** | Toggle reflections |
| **R** | Reset camera position |
| **ESC** | Exit |

## Requirements

- **NVIDIA GPU** with CUDA Compute Capability 7.5+ (RTX 20 series or newer recommended)
- **CUDA Toolkit 13.1** or compatible version
- **Visual Studio 2022** with C++ and CUDA development workload
- **Windows 10/11**

## Dependencies

The project uses NuGet packages (automatically restored):
- **GLFW 3.4.0** - Window and input management
- **GLEW 1.12.0** - OpenGL extension loading

## Building

1. Clone the repository
2. Open `CudaRuntime1.sln` in Visual Studio 2022
3. Right-click the project → **Manage NuGet Packages** → Restore packages
4. Select **Release | x64** configuration
5. Build and run (F5)

## Project Structure

```
├── kernel.cu           # Main file: CUDA kernels, OpenGL setup, game loop
├── raytracer.cuh       # Core raytracing algorithms (traceRay, findClosestHit)
├── bvh.cuh             # Bounding Volume Hierarchy for acceleration
├── obj_loader.cuh      # Wavefront OBJ file parser
├── vec3.cuh            # 3D vector math operations
├── ray.cuh             # Ray structure and utilities
├── sphere.cuh          # Sphere primitive with intersection test
├── triangle.cuh        # Triangle primitive (Möller-Trumbore algorithm)
├── materials.cuh       # Material system (diffuse, reflective, emissive)
├── light.cuh           # Point light structure
├── hit_record.cuh      # Intersection record structure
├── debug_overlay.cuh   # Debug text rendering with bitmap font
└── scene.cuh           # Scene configuration (not used in current version)
```

## Technical Highlights

### BVH Grid Acceleration
The raytracer uses a uniform grid BVH to accelerate ray-triangle intersection:
- Divides the scene into an 8×8×8 voxel grid
- Each cell stores indices of overlapping triangles
- Uses 3D-DDA algorithm for efficient grid traversal
- Achieves ~10x speedup over brute-force on complex meshes

### CUDA-OpenGL Interop
Zero-copy rendering pipeline:
1. Register OpenGL PBO with CUDA (`cudaGraphicsGLRegisterBuffer`)
2. Map PBO for CUDA access (`cudaGraphicsMapResources`)
3. CUDA kernel writes pixels directly to PBO
4. Unmap and use PBO as texture source

### Parallel Ray Tracing
- Each pixel is processed by an independent CUDA thread
- ~1 million threads execute in parallel (1280×720 resolution)
- Optimal block size (32×8) for memory coalescing

## Performance

| Configuration | Resolution | FPS (approx) |
|--------------|------------|--------------|
| GPU (RTX 2050) | 1280×720 | 16-30 FPS |
| CPU (single-threaded) | 1280×720 | 2-5 FPS |

*Performance varies based on scene complexity and reflections enabled.*

## License

This project is available for educational purposes. Feel free to use it as a learning resource or portfolio piece.

---

*Built as a learning project to explore GPU parallel computing and real-time graphics.*
