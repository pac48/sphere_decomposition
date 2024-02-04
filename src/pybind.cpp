#include "sphere_decomposition.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <variant>
#include "gui.hpp"

pybind11::array_t<uint8_t>
render(float fx, float fy, unsigned int res_x, unsigned int res_y,
       std::variant<SDFSpherePy, SDFPolynomialPy, SDFRadialPy> &sdf_object) {

    std::shared_ptr<SDFObject> object;
    if (SDFSpherePy *obj_py = std::get_if<SDFSpherePy>(&sdf_object)) {
        object = obj_py->operator()();
    } else if (SDFPolynomialPy *obj_py = std::get_if<SDFPolynomialPy>(&sdf_object)) {
        object = obj_py->operator()();
    } else if (SDFRadialPy *obj_py = std::get_if<SDFRadialPy>(&sdf_object)) {
        object = obj_py->operator()();
    } else {
        throw pybind11::type_error();
    }

    auto img_vec = internal::render(fx, fy, res_x, res_y, *object);
    pybind11::array_t<uint8_t> img(img_vec.size(), img_vec.data());
    img.resize({{res_y, res_x, 4}});

    return img;
}

PYBIND11_MODULE(sdf_experiments_py, m) {
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
            .def("set_img", [](ImguiController &controller, const pybind11::array_t<uint8_t> &img) { return controller.set_img(img); });
    pybind11::class_<SDFSpherePy>(m, "SDFSphere", R"(
    SDFSphere contains parameters of SDF.)")
            .def(pybind11::init([]() {
                     auto sdf_object = SDFSpherePy();
                     return sdf_object;
                 }),
                 R"(
                 Init.
           )").def("__str__", [](const SDFSpherePy &sdf_object) {
                std::stringstream ss;
                for (int r = 0; r < 3; r++) {
                    ss << "T:\n[";
                    for (int c = 0; c < 4; c++) {
                        ss << sdf_object.T.data()[r * 4 + c] << ", ";
                    }
                    ss << "]\n";
                }
                ss << "radius: " << sdf_object.radius;
                return ss.str();
            }).def_readwrite("T", &SDFSpherePy::T)
            .def_readwrite("radius", &SDFSpherePy::radius);
    pybind11::class_<SDFPolynomialPy>(m, "SDFPolynomial", R"(
    SDFPolynomial contains parameters of SDF.
									     )")

            .def(pybind11::init([](int num_coefficients) {
                     auto sdf_object = SDFPolynomialPy(num_coefficients);
                     return sdf_object;
                 }),
                 R"(
                 Init.
           )").def("__str__", [](const SDFPolynomialPy &sdf_object) {
                std::stringstream ss;
                for (int r = 0; r < 3; r++) {
                    ss << "T:\n[";
                    for (int c = 0; c < 4; c++) {
                        ss << sdf_object.T.data()[r * 4 + c] << ", ";
                    }
                    ss << "]\n";
                }
                ss << "p :\n[";
                for (int i = 0; i < sdf_object.coefficients.size(); i++) {
                    ss << sdf_object.coefficients.data()[i] << ", ";
                }
                ss << "]\n";

                return ss.str();
            }).def_readwrite("T", &SDFPolynomialPy::T)
            .def_readwrite("coefficients", &SDFPolynomialPy::coefficients);
    pybind11::class_<SDFRadialPy>(m, "SDFRadial", R"(
    SDFRadial contains parameters of SDF.
									     )")

            .def(pybind11::init([](const pybind11::array_t<float> &centers) {
                     auto sdf_object = SDFRadialPy(centers);
                     return sdf_object;
                 }),
                 R"(
                 Init.
           )").def("__str__", [](const SDFRadialPy &sdf_object) {
                std::stringstream ss;
                for (int r = 0; r < 3; r++) {
                    ss << "T:\n[";
                    for (int c = 0; c < 4; c++) {
                        ss << sdf_object.T.data()[r * 4 + c] << ", ";
                    }
                    ss << "]\n";
                }
                ss << "p :\n[";
                for (int i = 0; i < sdf_object.coefficients.size(); i++) {
                    ss << sdf_object.coefficients.data()[i] << ", ";
                }
                ss << "]\n";

                ss << "centers :\n[";
                for (int i = 0; i < sdf_object.centers.size(); i++) {
                    ss << sdf_object.centers.data()[i] << ", ";
                }
                ss << "]\n";

                return ss.str();
            }).def_readwrite("T", &SDFRadialPy::T)
            .def_readwrite("coefficients", &SDFRadialPy::coefficients)
            .def_readwrite("centers", &SDFRadialPy::centers);
}