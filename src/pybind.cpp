#include "sphere_decomposition.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <variant>
#include "gui.hpp"

pybind11::array_t<uint8_t>
render(float fx, float fy, unsigned int res_x, unsigned int res_y, const pybind11::array_t<double>& triangles) {
  CArray triangles_arr{triangles.data(), triangles.size()};
  auto img_vec = sphere_decomposition::render(fx, fy, res_x, res_y, triangles_arr);
  pybind11::array_t<uint8_t> img(img_vec.size(), img_vec.data());
  img.resize({{res_y, res_x, 4}});
  return img;
}

PYBIND11_MODULE(sphere_decomposition_py, m) {
  m.def("render", render);
  pybind11::class_<ImguiController>(m, "ImguiController", R"(
    ImguiController used for rendering.
									     )")
      .def(pybind11::init([]() {
             auto controller = ImguiController();
             return controller;
           }),
           R"(
                 Init.
           )").def("get_width", [](ImguiController &controller) { return controller.get_width(); })
      .def("get_height", [](ImguiController &controller) { return controller.get_height(); })
      .def("get_camera_transform", [](ImguiController &controller) { return controller.get_camera_transform(); })
      .def("set_img",
           [](ImguiController &controller, const pybind11::array_t<uint8_t> &img) { return controller.set_img(img); });
}