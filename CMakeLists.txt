cmake_minimum_required(VERSION 3.27)
set(CMAKE_CUDA_ARCHITECTURES 62)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
project(sphere_decomposition
    LANGUAGES CXX CUDA)
set(CMAKE_MODULE_PATH "/usr/local/share/cmake-3.22/Modules")

#include(cmake/FindCUDA.cmake)
#find_package(CUDA REQUIRED)
#enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 REQUIRED)
find_package(PythonInterp REQUIRED)
if (PYTHON_INCLUDE_DIRS)
  include_directories(${PYTHON_INCLUDE_DIRS})
  message("Using python version: " ${PYTHONLIBS_VERSION_STRING})
else ()
  message(FATAL_ERROR "python not found!")
endif ()

# set CUSTOM_PYTHON_EXE to use virtual environment
if (CUSTOM_PYTHON_EXE)
  set(PYTHON_EXECUTABLE ${CUSTOM_PYTHON_EXE})
endif ()
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "from distutils import sysconfig; print(sysconfig.get_python_lib())"
    OUTPUT_VARIABLE SITE_PACKAGES_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_library(sphere_decomposition SHARED src/sphere_decomposition.cu)
target_include_directories(sphere_decomposition PUBLIC include)
set_target_properties(sphere_decomposition PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_SEPARABLE_COMPILATION ON)
#target_compile_options(sphere_decomposition PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G -lineinfo>)

pybind11_add_module(sphere_decomposition_py src/pybind.cpp)
target_link_libraries(sphere_decomposition_py PUBLIC sphere_decomposition)
set_target_properties(sphere_decomposition_py PROPERTIES
    INSTALL_RPATH ${CMAKE_BINARY_DIR}
    PREFIX ""
)
install(TARGETS sphere_decomposition_py LIBRARY DESTINATION ${SITE_PACKAGES_DIR}/sphere_decomposition)



add_subdirectory(gui)
target_link_libraries(sphere_decomposition_py PUBLIC gui)
