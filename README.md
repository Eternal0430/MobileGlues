> [!WARNING]
> **此项目完全由ai驱动开发**
# MobileGLESWrapper

> **注意：这是非官方版本（Unofficial Fork）**
>
> 本项目基于 [MobileGlues](https://github.com/MobileGL-Dev/MobileGlues) 修改而来，并非官方发布版本。
> 官方项目请访问：https://github.com/MobileGL-Dev/MobileGlues
>
> 本版本将项目名称及相关标识从 `MobileGlues` 修改为 `MobileGLESWrapper`，其余内容可能与官方版本存在差异。请勿向官方项目提交与本版本相关的 Issue 或 PR。

---

**MobileGLESWrapper**（原 MobileGlues，意为 "(on) Mobile, GL uses ES"）是一个运行在 OpenGL ES 3.2 之上的 GL 实现，主要面向 Minecraft: Java Edition 的运行场景。

> **注意：本项目已放弃对 OpenGL ES 3.1 及更低版本的支持，仅支持 OpenGL ES 3.2。**

> [!IMPORTANT]
> **关于 ANGLE 支持**
>
> 本 Fork 版本**不支持启用 ANGLE**。`enableANGLE` 配置项已被移除，ANGLE 后端始终处于关闭状态。如需 ANGLE 支持，请使用 [官方版本](https://github.com/MobileGL-Dev/MobileGlues-release)。

# 面向着色器开发者

1. MobileGLESWrapper 会自动：
   - 将桌面版 GLSL 转换为 GLSL ES
   - 移除 `layout(binding)` 语法
   - 处理版本指令
   - 始终显式声明精度：
     ```glsl
     precision highp float;
     precision highp int;
     ```

2. MobileGLESWrapper（自 V1.2.6 起）会在着色器中注入以下宏：
   ```glsl
   #define MG_MOBILEGLUES                   // 标识 MobileGlues 环境
   #define MG_MOBILEGLUES_VERSION 1260      // 版本号（如 1260 = V1.2.6）
   ```

   使用这些宏进行平台特定逻辑判断：
   ```glsl
   #ifdef MG_MOBILEGLUES
       #if MG_MOBILEGLUES_VERSION >= 1270
           // MobileGlues 版本 >= V1.2.7 的逻辑
       #else
           // MobileGlues 版本 < V1.2.7 的逻辑
       #endif
   #else
       // ...
   #endif
   ```

3. 如遇到问题：
   - 启用 `Ignore shader/program error`，并检查日志（位于 `/sdcard/MG/latest.log`）。

# 许可证

本项目基于 **GNU LGPL-2.1 License** 许可。

请参阅原始项目 [LICENSE](https://github.com/MobileGL-Dev/MobileGlues/blob/main/LICENSE)。

# 第三方组件

**SPIRV-Cross** by **KhronosGroup** - [Apache License 2.0](https://github.com/KhronosGroup/SPIRV-Cross/blob/master/LICENSE): [github](https://github.com/KhronosGroup/SPIRV-Cross)

**glslang** by **KhronosGroup** - [Various Licenses](https://github.com/KhronosGroup/glslang/blob/main/LICENSE.txt): [github](https://github.com/KhronosGroup/glslang)

**cJSON** by **DaveGamble** - [MIT License](https://github.com/DaveGamble/cJSON/blob/master/LICENSE): [github](https://github.com/DaveGamble/cJSON)

**OpenGL Mathematics (*GLM*)** by **G-Truc Creation** - [The Happy Bunny License](https://github.com/g-truc/glm/blob/master/copying.txt): [github](https://github.com/g-truc/glm)

**FidelityFX-FSR** by **AMD** - [MIT License](https://github.com/GPUOpen-Effects/FidelityFX-FSR/blob/master/license.txt): [github](https://github.com/GPUOpen-Effects/FidelityFX-FSR)

**Perfetto** by **Google** - [Apache License 2.0](https://github.com/google/perfetto/blob/main/LICENSE): [github](https://github.com/google/perfetto)

**xxHash** by **Yann Collet** - [BSD 2-Clause License](https://github.com/Cyan4973/xxHash/blob/dev/LICENSE): [github](https://github.com/Cyan4973/xxHash)

# 项目目录结构

```
MobileGLESWrapper/
├── .github/workflows/
│   ├── build-android.yml
│   └── main.yml
├── MobileGlues-cpp/                  # 核心 C++ 实现
│   ├── config/                       # 配置管理
│   │   ├── cJSON.c / cJSON.h         # JSON 解析
│   │   ├── config.cpp / config.h     # 配置核心
│   │   ├── gpu_utils.cpp / .h        # GPU 工具
│   │   └── settings.cpp / .h         # 设置项
│   ├── egl/                          # EGL 层
│   │   ├── egl.cpp / .h              # EGL 接口实现
│   │   └── loader.cpp / .h           # 符号加载
│   ├── gl/                           # OpenGL 实现
│   │   ├── backend/                  # 后端抽象
│   │   │   ├── BackendObject.h
│   │   │   ├── Init.cpp
│   │   │   └── DirectGLES/           # DirectGLES 后端
│   │   ├── glsl/                     # GLSL 转换与缓存
│   │   │   ├── glsl_for_es.cpp / .h  # 桌面 GLSL → GLSL ES
│   │   │   ├── cache.cpp / .h        # 着色器缓存
│   │   │   └── benchmark_parser.cpp
│   │   ├── impl/                     # GL 接口实现
│   │   │   ├── Buffer/               # 缓冲区对象
│   │   │   ├── Drawing/              # 绘制调用
│   │   │   ├── Framebuffer/          # 帧缓冲对象
│   │   │   ├── Getter/               # 状态查询
│   │   │   ├── Program/              # 着色器程序
│   │   │   ├── RenderState/          # 渲染状态
│   │   │   ├── Texture/              # 纹理对象
│   │   │   ├── VertexArray/          # 顶点数组
│   │   │   └── Exporting.cpp         # 符号导出
│   │   ├── state/                    # 状态管理
│   │   ├── ExtWrappers/              # 扩展封装
│   │   │   ├── DSAWrapper.cpp / .h
│   │   │   └── MultiBindWrapper.cpp / .h
│   │   ├── FSR1/                     # FSR 超分辨率
│   │   ├── gl.cpp / gl.h             # GL 入口
│   │   ├── gl_native.cpp             # native GL 调用
│   │   ├── gl_stub.cpp               # stub 实现
│   │   ├── shader.cpp / .h           # 着色器管理
│   │   ├── program.cpp / .h          # 程序管理
│   │   ├── texture.cpp / .h          # 纹理管理
│   │   ├── buffer.cpp / .h           # 缓冲管理
│   │   ├── framebuffer.cpp / .h      # 帧缓冲管理
│   │   ├── state.cpp / .h            # 状态管理
│   │   ├── drawing.cpp / .h          # 绘制管理
│   │   ├── getter.cpp / .h           # 查询管理
│   │   ├── vertexattrib.cpp / .h     # 顶点属性
│   │   ├── multidraw.cpp / .h        # 多绘制调用
│   │   ├── pixel.cpp / .h            # 像素操作
│   │   ├── envvars.cpp / .h          # 环境变量
│   │   ├── log.cpp / .h              # 日志
│   │   ├── mg.cpp / .h               # MobileGLES 管理
│   │   ├── loader.h                  # 符号加载声明
│   │   └── random_string_gen.cpp / .h
│   ├── gles/                         # GLES 层
│   │   ├── gles.h
│   │   └── loader.cpp / .h
│   ├── glx/                          # GLX 查找
│   │   └── lookup.cpp / .h
│   ├── include/                      # 第三方头文件
│   │   ├── EGL/                      # EGL 头文件
│   │   ├── GL/                       # GL 头文件
│   │   ├── GLES3/                    # GLES3 头文件
│   │   ├── KHR/                      # KHR 头文件
│   │   ├── MG/                       # MobileGLES 扩展头文件
│   │   ├── SPIRV/                    # SPIRV 工具链
│   │   ├── glslang/                  # glslang 编译器
│   │   ├── spirv_cross/              # SPIRV-Cross 转换
│   │   ├── ankerl/                   # unordered_dense 哈希
│   │   └── vulkan/                   # Vulkan 头文件
│   ├── shaders/                      # 着色器文件
│   │   └── multidraw_compute.comp
│   ├── scripts/                      # 辅助脚本
│   │   ├── format_code.sh
│   │   └── update-source-file-header.py
│   ├── CMakeLists.txt
│   ├── main.cpp                      # 入口
│   ├── init.cpp                      # 初始化
│   └── version.h                     # 版本信息
├── android_stub/                     # Android 存根
│   └── android/log.h
├── .gitignore
├── .gitmodules
├── LICENSE
├── README.md
├── build.gradle.kts                  # Gradle 构建
├── mg.pbtx                           # Perfetto 配置
├── mg_full.pbtx
├── record_android_trace.py           # 性能追踪
└── trace.bat
```
