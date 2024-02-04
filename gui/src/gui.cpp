// Dear ImGui: standalone example application for GLFW + OpenGL 3, using programmable pipeline
// (GLFW is a cross-platform general purpose library for handling windows, inputs, OpenGL/Vulkan/Metal graphics context creation, etc.)

// Learn about Dear ImGui:
// - FAQ                  https://dearimgui.com/faq
// - Getting Started      https://dearimgui.com/getting-started
// - Documentation        https://dearimgui.com/docs (same as your local docs/ folder).
// - Introduction, links and more at the top of imgui.cpp
#include "gui.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>

#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif

#include <GLFW/glfw3.h> // Will drag system OpenGL headers
#include <iostream>

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to maximize ease of testing and compatibility with old VS compilers.
// To link with VS2010-era libraries, VS2015+ requires linking with legacy_stdio_definitions.lib, which we do using this pragma.
// Your own project should not be affected, as you are likely to link with a newer binary of GLFW that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

// This example can also compile and run with Emscripten! See 'Makefile.emscripten' for details.
#ifdef __EMSCRIPTEN__
#include "../libs/emscripten/emscripten_mainloop_stub.h"
#endif

static void glfw_error_callback(int error, const char *description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

struct RenderData {
    GLFWwindow *window;
    ImGuiIO io;
};


using namespace imgui_rendering;
// Main code
imgui_rendering::State state;
Eigen::Matrix3f last_rotation;
Eigen::Vector3f last_pos;

void imgui_rendering::init() {
    state.data = std::make_shared<RenderData>();
    last_pos = state.pos;
    last_rotation = state.rotation;

    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return;

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100
    const char* glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    // GL 3.0 + GLSL 130
    const char *glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

    // Create window with graphics context
    state.data->window = glfwCreateWindow(state.width, state.height, "Dear ImGui GLFW+OpenGL3 example", nullptr,
                                          nullptr);
    if (state.data->window == nullptr)
        return;
    glfwMakeContextCurrent(state.data->window);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    state.data->io = ImGui::GetIO();
    state.data->io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    state.data->io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(state.data->window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
    // - If the file cannot be loaded, the function will return a nullptr. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
    // - Use '#define IMGUI_ENABLE_FREETYPE' in your imconfig file to use Freetype for higher quality font rendering.
    // - Read 'docs/FONTS.md' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
    // - Our Emscripten build process allows embedding fonts to be accessible at runtime from the "fonts/" folder. See Makefile.emscripten for details.
    //io.Fonts->AddFontDefault();
    //io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\segoeui.ttf", 18.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
    //ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, nullptr, io.Fonts->GetGlyphRangesJapanese());
    //IM_ASSERT(font != nullptr);

    // Our state
}


void imgui_rendering::render_frame() {
    if (!glfwWindowShouldClose(state.data->window)) {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();

        float speed = 0.03 * 7;
        if (ImGui::IsKeyDown(ImGuiKey::ImGuiKey_W)) {
            state.pos += speed * state.rotation.col(2);
        }
        if (ImGui::IsKeyDown(ImGuiKey::ImGuiKey_S)) {
            state.pos -= speed * state.rotation.col(2);
        }
        if (ImGui::IsKeyDown(ImGuiKey::ImGuiKey_D)) {
            state.pos += speed * state.rotation.col(0);
        }
        if (ImGui::IsKeyDown(ImGuiKey::ImGuiKey_A)) {
            state.pos -= speed * state.rotation.col(0);
        }
        if (ImGui::IsKeyDown(ImGuiKey::ImGuiKey_Q)) {
            state.pos += speed * state.rotation.col(1);
        }
        if (ImGui::IsKeyDown(ImGuiKey::ImGuiKey_E)) {
            state.pos -= speed * state.rotation.col(1);
        }

        if (state.pos.norm() > 20) {
            state.pos = 20 * state.pos / state.pos.norm();
        }

        if (ImGui::IsMouseDown(0)) {
            auto mouse_delta = ImGui::GetMouseDragDelta();
            float delta_x = 3*mouse_delta.x / state.width;
            float delta_y = 3*mouse_delta.y / state.height;

            Eigen::Matrix3f rotx = Eigen::Matrix3f::Identity();
            Eigen::Matrix3f roty= Eigen::Matrix3f::Identity();
            Eigen::Matrix3f rot= Eigen::Matrix3f::Identity();
            rotx(1, 1) = 1;
            rotx(0, 0) = cos(-delta_x);
            rotx(0, 2) = -sin(-delta_x);
            rotx(2, 2) = cos(-delta_x);
            rotx(2, 0) = sin(-delta_x);

            roty(0, 0) = 1;
            roty(1, 1) = cos(-delta_y);
            roty(1, 2) = -sin(-delta_y);
            roty(2, 2) = cos(-delta_y);
            roty(2, 1) = sin(-delta_y);

            rot = rotx * roty;

            state.rotation = last_rotation*rot;

            Eigen::AngleAxisf aa(state.rotation);    // RotationMatrix to AxisAngle
            state.rotation = aa.toRotationMatrix();  // AxisAngle      to RotationMatrix

//            std::cout << rot << "\n";
            state.pos = state.rotation*rot * state.rotation.transpose()*last_pos;
        } else {
            last_rotation = state.rotation;
            last_pos = state.pos;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin(
                    "Hello, world!");                          // Create a window called "Hello, world!" and append into it.

            ImGui::Text(
                    "This is some useful text.");               // Display some text (you can use a format strings too)
            static bool show_demo_window;
            static bool show_another_window;
            ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
            ImGui::Checkbox("Another Window", &show_another_window);

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3("clear color", (float *) &clear_color); // Edit 3 floats representing a color

            if (ImGui::Button(
                    "Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / state.data->io.Framerate,
                        state.data->io.Framerate);
            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(state.data->window, &display_w, &display_h);
        state.width = display_w;
        state.height = display_h;
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w,
                     clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        {
            std::scoped_lock lock(state.mut);
            if (state.img != nullptr && state.img_size == display_h * display_w * 3) {
                glDrawPixels(display_w, display_h, GL_RGB, GL_UNSIGNED_BYTE, state.img);
            }
        }

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(state.data->window);
    }
}

void imgui_rendering::cleanup() {
    // Cleanup
//    ImGui_ImplOpenGL3_Shutdown();
//    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(state.data->window);
    glfwTerminate();
}

namespace imgui_rendering {
    void stop_rendering() {
        state.t->join();
        cleanup();
    }

    void start_rendering() {
        state.t = std::make_shared<std::thread>([]() {
            init();
            while (state.rendering != 0) {
                render_frame();
            }
        });
    }
}