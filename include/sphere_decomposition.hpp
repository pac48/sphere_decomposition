#pragma once

#include "vector"

struct CArray {
  const double *data;
  long size;
};
namespace sphere_decomposition {
  std::vector<unsigned char> render(float fx, float fy, unsigned int res_x, unsigned int res_y, CArray triangles);
}
