cmake_minimum_required(VERSION 3.27)
project(sphere_decomposition)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
enable_language(CUDA)
set(CMAKE_MODULE_PATH "/usr/share/cmake-3.22/Modules")
find_package(CUDA REQUIRED)
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

pybind11_add_module(sphere_decomposition_py src/pybind.cpp)
add_library(sphere_decomposition SHARED src/sphere_decomposition.cu)
target_include_directories(sphere_decomposition PUBLIC include)


set_target_properties(sphere_decomposition PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_SEPARABLE_COMPILATION ON)
#target_compile_options(sdf_renderer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G -lineinfo>)

target_link_libraries(sphere_decomposition_py PUBLIC sphere_decomposition)

install(TARGETS sphere_decomposition_py LIBRARY DESTINATION ${SITE_PACKAGES_DIR}/sphere_decomposition)

add_subdirectory(gui)
target_link_libraries(sphere_decomposition_py PUBLIC gui)
