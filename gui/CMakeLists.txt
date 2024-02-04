set(imgui_src imgui/imgui.cpp imgui/imgui_demo.cpp imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp imgui/imgui_widgets.cpp imgui/backends/imgui_impl_opengl3.cpp
    imgui/backends/imgui_impl_glfw.cpp)
add_library(gui src/gui.cpp ${imgui_src})
target_include_directories(gui PUBLIC include)
find_package(glfw3 REQUIRED)
find_package(OpenGL)
find_package(Eigen3)
target_include_directories(gui PUBLIC ${EIGEN3_INCLUDE_DIR})

target_link_libraries(gui glfw)
target_link_libraries(gui OpenGL::GL)
target_link_libraries(gui Eigen3::Eigen)
target_include_directories(gui PUBLIC imgui/backends)
target_include_directories(gui PUBLIC imgui)
