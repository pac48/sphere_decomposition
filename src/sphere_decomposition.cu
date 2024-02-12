#include "sphere_decomposition.hpp"
#include "memory"
#include "chrono"
#include "iostream"

namespace sphere_decomposition {
  template<typename T>
  struct BufferGPU {
    T *buffer;
    size_t size;

    explicit BufferGPU(size_t size_in) : size{size_in} {
      cudaMalloc(&buffer, size * sizeof(T));
      cudaMemset(&buffer, 0, size * sizeof(T));
    }

    BufferGPU(const BufferGPU &other) {
      size = other.size;
      cudaMalloc(&buffer, size * sizeof(T));
      cudaMemcpy(&buffer, &other, size * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    }

    BufferGPU &operator=(const BufferGPU &other) {
      cudaFree(buffer);
      size = other.size;
      cudaMalloc(&buffer, size * sizeof(T));
      cudaMemcpy(&buffer, &other, size * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
      return *this;
    }

    ~BufferGPU() {
      cudaFree(buffer);
    }

    std::vector<T> toCPU() {
      std::vector<T> out(size);
      cudaMemcpy(out.data(), buffer, size * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost);

      return out;
    }

  };

  std::shared_ptr<BufferGPU<unsigned char>> gpu_pixel_buffer = nullptr;
  std::shared_ptr<BufferGPU<float>> gpu_triangle_buffer = nullptr;


  struct Vertex {
    float x;
    float y;
    float z;
  };

  typedef Vertex Vector;

  inline __device__ float dot_product(const Vertex &a, const Vertex &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }

  inline __device__ Vertex vec_minus(const Vertex &vert1, const Vertex &vert2) {
    Vertex out;
    out.x = vert1.x - vert2.x;
    out.y = vert1.y - vert2.y;
    out.z = vert1.z - vert2.z;
    return out;
  }

  struct Triangle {
    Vertex v1;
    Vertex v2;
    Vertex v3;
  };

  struct Vertex2 {
    float x;
    float y;
  };

  inline __device__ Vertex2 vec_minus(const Vertex2 &vert1, const Vertex2 &vert2) {
    Vertex2 out;
    out.x = vert1.x - vert2.x;
    out.y = vert1.y - vert2.y;
    return out;
  }

  inline __device__ float dot_product2(const Vertex2 &a, const Vertex2 &b) {
    return a.x * b.x + a.y * b.y;
  }


  struct Triangle2D {
    Vertex2 v1;
    Vertex2 v2;
    Vertex2 v3;
  };


  inline __device__ void
  bary_centric(const Vertex2 &a, const Vertex2 &b, const Vertex2 &c, const Vertex2 &p, float &u, float &v, float &w) {
    Vertex2 v0 = vec_minus(b, a);
    Vertex2 v1 = vec_minus(c, a);
    Vertex2 v2 = vec_minus(p, a);
    float d00 = dot_product2(v0, v0);
    float d01 = dot_product2(v0, v1);
    float d11 = dot_product2(v1, v1);
    float d20 = dot_product2(v2, v0);
    float d21 = dot_product2(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0f - v - w;
  }

  inline Vector __device__ cross(const Vector &a, const Vector &b) {
    Vector out;
    out.x = a.y * b.z - a.z * b.y;
    out.y = a.z * b.x - b.z * a.x;
    out.z = a.x * b.y - a.y * b.x;

    return out;
  }

  inline float __device__ cross_Z2(const Vertex2 &a, const Vertex2 &b) {
    return a.x * b.y - a.y * b.x;

  }

  __device__ int is_point_in_triangle(const Vertex2 &a, const Vertex2 &b, const Vertex2 &c, const Vertex2 &p) {
    // if z component of cross product is positive, then the point is inside for convex mesh
    float val1 = cross_Z2(vec_minus(b, a), vec_minus(p, a));
    float val2 = cross_Z2(vec_minus(c, b), vec_minus(p, b));
    float val3 = cross_Z2(vec_minus(a, c), vec_minus(p, c));
    return (val1 < 0 && val2 < 0 && val3 < 0);
  }

  __device__ void project_triangle(float fx, float fy, Triangle &triangle) {
    triangle.v1.x = fx * triangle.v1.x / triangle.v1.z;
    triangle.v1.y = fy * triangle.v1.y / triangle.v1.z;
    triangle.v1.z = 1;

    triangle.v2.x = fx * triangle.v2.x / triangle.v2.z;
    triangle.v2.y = fy * triangle.v2.y / triangle.v2.z;
    triangle.v2.z = 1;

    triangle.v3.x = fx * triangle.v3.x / triangle.v3.z;
    triangle.v3.y = fy * triangle.v3.y / triangle.v3.z;
    triangle.v3.z = 1;
  }

  constexpr float MAX_DEPTH = 1E10;


  __global__ void render_kernel(const float *triangles, size_t size, unsigned char *image, size_t image_size,
                                float fx, float fy, unsigned int res_x, unsigned int res_y) {
    // Get the index of the current thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Ensure we don't go out of bounds
    if (idx < image_size / 4) {
      unsigned int ind_x = idx % res_x;
      unsigned int ind_y = idx / res_x;
      float depth = MAX_DEPTH;
      image[idx * 4] = 0;
      image[idx * 4 + 1] = 0;
      image[idx * 4 + 2] = 0;
      image[idx * 4 + 3] = 255;

#pragma unroll
      for (size_t ind = 0; ind < size; ind += (9 + 6)) {
        const Triangle &triangle = *(Triangle *) &triangles[ind];
        const Triangle2D &triangle2d = *(Triangle2D *) &triangles[ind + 9];
        Vertex2 point2;
        point2.x = 2.0 * (ind_x - res_x / 2.0) / res_x;
        point2.y = 2.0 * (ind_y - res_y / 2.0) / res_y;
        if (is_point_in_triangle(triangle2d.v1, triangle2d.v2, triangle2d.v3, point2) == true) {
          // ax + by + cz + d = 0;
          // z = -(ax + by + d)/c;
          float u, v, w;
          bary_centric(triangle2d.v1, triangle2d.v2, triangle2d.v3, point2, u, v, w);
          float new_dist = triangle.v1.z * u + triangle.v2.z * v + triangle.v3.z * w;

          // now normalize norm vector
          Vector normal = cross(vec_minus(triangle.v2, triangle.v1), vec_minus(triangle.v3, triangle.v1));
          float length = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
          normal.x = normal.x / length;
          normal.y = normal.y / length;
          normal.z = normal.z / length;

          if (new_dist < depth) { // && normal.z > 0 && normal.z < 0
//            printf("depth: %f\n", new_depth);
//            printf("thread id: %d\n", idx);
            depth = new_dist;
            image[idx * 4] = (float) -normal.z * 200;
            image[idx * 4 + 1] = (float) -normal.z * 200;
            image[idx * 4 + 2] = (float) -normal.z * 200;
            image[idx * 4 + 3] = 255;
          }
        }
      }
    }
  }


  std::vector<unsigned char> render(float fx, float fy, unsigned int res_x, unsigned int res_y, CArray triangles) {
    if (gpu_pixel_buffer == nullptr || res_x * res_y * 4 > gpu_pixel_buffer->size) {
      gpu_pixel_buffer = std::make_shared<BufferGPU<unsigned char>>(res_x * res_y * 4);
    }
    if (gpu_triangle_buffer == nullptr || triangles.size > gpu_triangle_buffer->size) {
      gpu_triangle_buffer = std::make_shared<BufferGPU<float>>(triangles.size);
    }
    cudaMemcpy(gpu_triangle_buffer->buffer, triangles.data, triangles.size * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(256);
    dim3 numBlocks((gpu_pixel_buffer->size / 4 + threadsPerBlock.x - 1) / threadsPerBlock.x);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    auto start = std::chrono::high_resolution_clock::now();
    render_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        gpu_triangle_buffer->buffer, triangles.size, gpu_pixel_buffer->buffer, res_x * res_y * 4, fx,
        fy, res_x, res_y);
    cudaStreamSynchronize(stream);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken by render: " << (double) duration.count() << " microseconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    auto out = gpu_pixel_buffer->toCPU();
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken by copy data: " << (double) duration.count() << " microseconds" << std::endl;

    cudaStreamDestroy(stream);

    return out;
  }

}
