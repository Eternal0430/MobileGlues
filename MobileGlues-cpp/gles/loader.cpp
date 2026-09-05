// MobileGlues - gles/loader.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

// ============================================================================
// GLES 3.2 Loader - ES 3.2-Only Architecture
//
// This file targets OpenGL ES 3.2 exclusively. All ES 3.2 core features
// (compute shaders, geometry/tessellation shaders, multisample textures,
//  texture buffer, debug output, etc.) are always available — no capability
//  checks are needed for them.
//
// Only optional extensions beyond ES 3.2 core (e.g. EXT_buffer_storage,
//  EXT_disjoint_timer_query, EXT_multi_draw_indirect, OES_mapbuffer)
//  are detected at runtime and stored in g_gles_caps.
// ============================================================================

#include <cstring>
#include <cstdio>
#include "loader.h"
#include "../includes.h"
#include "loader.h"
#include <GL/gl.h>
#include "../gl/glext.h"
#include "../gl/envvars.h"
#include "../gl/log.h"
#include "../gl/mg.h"
#include "../gl/buffer.h"
#include "../gl/getter.h"
#include "../config/settings.h"
#include "../gl/texture.h"
#include "../gl/framebuffer.h"

#define DEBUG 0

void *gles = nullptr, *egl = nullptr;

struct gles_func_t g_gles_func;

// ---------------------------------------------------------------------------
// Library search paths
// ---------------------------------------------------------------------------

static const char* path_prefix[] = {
    "", "/opt/vc/lib/", "/usr/local/lib/", "/usr/lib/", nullptr,
};

static const char* lib_ext[] = {
#ifndef NO_GBM
    "so.19",
#endif
    "so",    "so.1", "so.2", "dylib", "dll", nullptr,
};

static const char* gles3_lib[] = {"libGLESv3_CM", "libGLESv3", nullptr};

static const char* egl_lib[] = {
#if defined(BCMHOST)
    "libbrcmEGL",
#endif
    "libEGL", nullptr};

const char* GLES_ANGLE = "libGLESv2_angle.so";
const char* EGL_ANGLE = "libEGL_angle.so";

// ---------------------------------------------------------------------------
// Dynamic library loading
// ---------------------------------------------------------------------------

void* open_lib(const char** names, const char* override) {
    void* lib = nullptr;

    char path_name[PATH_MAX + 1];
    int flags = RTLD_LOCAL | RTLD_NOW;
    if (override) {
        if ((lib = dlopen(override, flags))) {
            strncpy(path_name, override, PATH_MAX);
            LOG_D("LIBGL:loaded: %s\n", path_name)
            return lib;
        } else {
            LOG_E("LIBGL_GLES override failed: %s\n", dlerror())
        }
    }
    for (int p = 0; path_prefix[p]; p++) {
        for (int i = 0; names[i]; i++) {
            for (int e = 0; lib_ext[e]; e++) {
                snprintf(path_name, PATH_MAX, "%s%s.%s", path_prefix[p], names[i], lib_ext[e]);
                if ((lib = dlopen(path_name, flags))) {
                    return lib;
                }
            }
        }
    }
    return lib;
}

void load_libs() {
    const char* gles_override = global_settings.angle == AngleMode::Enabled ? GLES_ANGLE : nullptr;
    const char* egl_override = global_settings.angle == AngleMode::Enabled ? EGL_ANGLE : nullptr;
    gles = open_lib(gles3_lib, gles_override);
    egl = open_lib(egl_lib, egl_override);
}

void* proc_address(void* lib, const char* name) {
    return dlsym(lib, name);
}

// ---------------------------------------------------------------------------
// Hardware & GL State Setup (ES 3.2 target)
// ---------------------------------------------------------------------------

void set_hardware() {
    // ES 3.2-only: always set version to 320 — no per-version branching needed.
    GLState.esVersion = 320;
    // Log the driver's GLES version string into the log file (latest.log),
    // matching upstream's startup renderer info output.
    set_es_version();
}

void init_gl_state() {
    // Initialize CPU+GPU symbiotic context
    // CPU-side: GLStateManager (via GLState alias)
    // GPU-side: BackendObject (set later via InitWindowSurface/InitPbufferSurface)
    GLState.Initialize();
    GLState.proxyWidth = 0;
    GLState.proxyHeight = 0;
    GLState.proxyInternalFormat = 0;

    InitTextureMap(1024);
    InitBufferMap(4096);
    InitVertexArrayMap(512);
    InitFramebufferMap(512);
}

// ---------------------------------------------------------------------------
// Extension logging
// ---------------------------------------------------------------------------

void LogOpenGLExtensions() {
    const GLubyte* raw_extensions = glGetString(GL_EXTENSIONS);
    LOG_D("Extensions list using glGetString:\n%s", raw_extensions ? (const char*)raw_extensions : "(nullptr)")
    GLint num_extensions = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &num_extensions);
    LOG_D("Extensions list using glGetStringi:\n")
    for (GLint i = 0; i < num_extensions; ++i) {
        const GLubyte* extension = glGetStringi(GL_EXTENSIONS, i);
        if (extension) {
            LOG_D("%s", (const char*)extension)
        } else {
            LOG_D("(nullptr)")
        }
    }
}

// ---------------------------------------------------------------------------
// Capability Detection (ES 3.2 core assumed, optional extensions only)
// ---------------------------------------------------------------------------

struct gles_caps_t g_gles_caps;

// =============================================================================
// Persistent mapping probe
// =============================================================================
namespace {

// Minecraft 26.3 maps its dynamic uniform buffers like this:
//     target : GlUtil.selectBufferBindTarget(130) = 35345 = GL_UNIFORM_BUFFER
//     storage: GlConst.bufferUsageToGlFlag(130)   = 0xC2 = WRITE|PERSISTENT|COHERENT
//     mapping: GlBuffer$Direct builds 34 | 192
//              = 0xE2 = WRITE|UNSYNCHRONIZED|PERSISTENT|COHERENT
//     size   : roundToward(TRANSFORM_UBO_SIZE=160, 256) * capacity(2) = 512
//
// Advertising GL_ARB_buffer_storage promises all of that works. A driver can
// publish the extension string and resolve every entry point and still refuse
// the mapping — which only surfaces much later as "IllegalStateException:
// Failed to map buffer", with no GL error in between and no surviving trace.
//
// So ask the driver directly, once, at start-up, using the game's own target
// and size. Every combination is tried independently rather than stopping at
// the first success: the earlier probe only reported the first rung and left
// the others as "FAILED", which made a working driver look broken.
//
// The numbers are written out rather than spelled with GL_ names because the
// mapping flags the game uses are not a combination any header defines.
constexpr GLenum kProbeTargets[] = {0x8892 /* GL_ARRAY_BUFFER */, 0x8A11 /* GL_UNIFORM_BUFFER */};
constexpr const char* kProbeTargetNames[] = {"ARRAY_BUFFER", "UNIFORM_BUFFER"};
constexpr int kProbeTargetCount = 2;

constexpr GLbitfield kProbeStorageFlags = 0xC2; // WRITE|PERSISTENT|COHERENT
constexpr GLbitfield kProbeFlagSets[] = {
    0xE2, // exactly what the game asks for
    0xC2, // with UNSYNCHRONIZED stripped (what buffer.cpp sends)
    0x02, // plain WRITE, no persistence at all
};
constexpr const char* kProbeFlagNames[] = {"E2", "C2", "W"};
constexpr int kProbeFlagCount = 3;
constexpr GLsizeiptr kProbeSize = 512; // the game's actual ring-buffer size

struct PersistentProbe {
    bool ran = false;
    bool storageOk[kProbeTargetCount] = {false, false};
    GLenum storageErr[kProbeTargetCount] = {GL_NO_ERROR, GL_NO_ERROR};
    bool mapOk[kProbeTargetCount][kProbeFlagCount] = {};
    GLenum mapErr[kProbeTargetCount][kProbeFlagCount] = {};

    // Whether persistent mapping (either flag set) works on the target the
    // game actually uses. This is what decides the extension advertisement.
    bool WorksOnGameTarget() const { return mapOk[1][0] || mapOk[1][1]; }
};

PersistentProbe ProbePersistentMapping() {
    PersistentProbe probe;

    if (!GLES.glGenBuffers || !GLES.glBindBuffer || !GLES.glDeleteBuffers || !GLES.glBufferStorageEXT ||
        !GLES.glMapBufferRange || !GLES.glUnmapBuffer || !GLES.glGetError || !GLES.glGetIntegerv)
        return probe;

    probe.ran = true;

    for (int t = 0; t < kProbeTargetCount; ++t) {
        const GLenum target = kProbeTargets[t];

        GLuint buffer = 0;
        GLES.glGenBuffers(1, &buffer);
        if (buffer == 0) continue;

        GLint previous = 0;
        GLES.glGetIntegerv(target == 0x8892 ? GL_ARRAY_BUFFER_BINDING : 0x8A28 /* GL_UNIFORM_BUFFER_BINDING */,
                           &previous);
        GLES.glBindBuffer(target, buffer);
        while (GLES.glGetError() != GL_NO_ERROR) {
        }

        GLES.glBufferStorageEXT(target, kProbeSize, nullptr, kProbeStorageFlags);
        probe.storageErr[t] = GLES.glGetError();
        probe.storageOk[t] = (probe.storageErr[t] == GL_NO_ERROR);
        if (!probe.storageOk[t]) {
            GLES.glBindBuffer(target, (GLuint)previous);
            GLES.glDeleteBuffers(1, &buffer);
            while (GLES.glGetError() != GL_NO_ERROR) {
            }
            continue;
        }

        for (int f = 0; f < kProbeFlagCount; ++f) {
            // A fresh buffer per attempt: some drivers keep a buffer mapped
            // after a failed call, which would poison every later attempt.
            GLuint b = 0;
            GLES.glGenBuffers(1, &b);
            if (b == 0) continue;
            GLES.glBindBuffer(target, b);
            GLES.glBufferStorageEXT(target, kProbeSize, nullptr, kProbeStorageFlags);
            while (GLES.glGetError() != GL_NO_ERROR) {
            }

            void* ptr = GLES.glMapBufferRange(target, 0, kProbeSize, kProbeFlagSets[f]);
            probe.mapErr[t][f] = GLES.glGetError();
            probe.mapOk[t][f] = (ptr != nullptr);
            if (ptr) GLES.glUnmapBuffer(target);

            GLES.glDeleteBuffers(1, &b);
            while (GLES.glGetError() != GL_NO_ERROR) {
            }
        }

        GLES.glBindBuffer(target, (GLuint)previous);
        GLES.glDeleteBuffers(1, &buffer);
        while (GLES.glGetError() != GL_NO_ERROR) {
        }
    }
    return probe;
}

} // namespace

void InitGLESCapabilities() {
    memset(&g_gles_caps, 0, sizeof(struct gles_caps_t));

    // Initialize the base GL extension list (GL_ARB_*, GL_KHR_*, etc.)
    // These are the extensions we report to the desktop GL layer.
    InitGLESBaseExtensions();

    GLES.glGetIntegerv(GL_MAJOR_VERSION, &g_gles_caps.major);
    GLES.glGetIntegerv(GL_MINOR_VERSION, &g_gles_caps.minor);

    // ES 3.2 core features are always available — no runtime checks needed for:
    //   - compute shaders (glDispatchCompute, etc.)
    //   - geometry/tessellation shaders
    //   - multisample/multisample array textures
    //   - texture buffer objects (glTexBuffer, glTexBufferRange)
    //   - debug output (glDebugMessageCallback, etc.)
    //   - per-blend-draw-buffer (glBlendFuncSeparatei, etc.)
    //   - robust buffer access (glGetGraphicsResetStatus, etc.)
    //   - depth_texture, depth24, texture_rg, texture_norm16, etc.
    //
    // We only detect optional extensions that are NOT part of ES 3.2 core.

    GLint num_es_extensions = 0;
    GLES.glGetIntegerv(GL_NUM_EXTENSIONS, &num_es_extensions);
    LOG_D("Detected %d OpenGL ES extensions.", num_es_extensions)
    for (GLint i = 0; i < num_es_extensions; ++i) {
        const char* extension = (const char*)GLES.glGetStringi(GL_EXTENSIONS, i);
        if (extension) {
            LOG_D("%s", (const char*)extension)
            // ---- Optional extensions beyond ES 3.2 core ----
            if (strcmp(extension, "GL_EXT_buffer_storage") == 0) {
                g_gles_caps.GL_EXT_buffer_storage = 1;
            } else if (strcmp(extension, "GL_EXT_disjoint_timer_query") == 0) {
                g_gles_caps.EXT_disjoint_timer_query = 1;
            } else if (strcmp(extension, "GL_OES_mapbuffer") == 0) {
                g_gles_caps.GL_OES_mapbuffer = 1;
            } else if (strcmp(extension, "GL_EXT_multi_draw_indirect") == 0) {
                g_gles_caps.GL_EXT_multi_draw_indirect = 1;
            } else if (strcmp(extension, "GL_KHR_texture_compression_astc_ldr") == 0) {
                g_gles_caps.KHR_texture_compression_astc_ldr = 1;
            } else if (strcmp(extension, "GL_EXT_texture_filter_anisotropic") == 0) {
                g_gles_caps.EXT_texture_filter_anisotropic = 1;
            } else if (strcmp(extension, "GL_OES_draw_elements_base_vertex") == 0) {
                g_gles_caps.GL_OES_draw_elements_base_vertex = 1;
            } else if (strcmp(extension, "GL_EXT_draw_elements_base_vertex") == 0) {
                g_gles_caps.GL_EXT_draw_elements_base_vertex = 1;
            } else if (strcmp(extension, "GL_EXT_multisample_compatibility") == 0) {
                g_gles_caps.GL_EXT_multisample_compatibility = 1;
            } else if (strcmp(extension, "GL_EXT_clip_cull_distance") == 0) {
                g_gles_caps.GL_EXT_clip_cull_distance = 1;
            } else if (strcmp(extension, "GL_EXT_depth_clamp") == 0) {
                g_gles_caps.GL_EXT_depth_clamp = 1;
            } else if (strcmp(extension, "GL_EXT_sRGB_write_control") == 0) {
                g_gles_caps.GL_EXT_sRGB_write_control = 1;
            } else if (strcmp(extension, "GL_NV_polygon_mode") == 0) {
                g_gles_caps.GL_NV_polygon_mode = 1;
            } else if (strcmp(extension, "GL_OES_sample_shading") == 0) {
                g_gles_caps.GL_OES_sample_shading = 1;
            } else if (strcmp(extension, "GL_EXT_color_buffer_float") == 0) {
                // Required for RGBA32F / R11F_G11F_B10F / RGB32F / RG32F / R32F
                // as color-renderable FBO attachments. Without it, GLES drivers
                // reject these formats with GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT
                // even though desktop GL accepts them as core.
                g_gles_caps.GL_EXT_color_buffer_float = 1;
            } else if (strcmp(extension, "GL_EXT_color_buffer_half_float") == 0) {
                // Required for RGBA16F / RG16F / R16F / RGB16F as
                // color-renderable FBO attachments. Advertised separately from
                // GL_EXT_color_buffer_float on many mobile drivers.
                g_gles_caps.GL_EXT_color_buffer_half_float = 1;
            } else if (strcmp(extension, "GL_EXT_texture_norm16") == 0) {
                // Required to keep GL_RGBA16 / GL_RG16 / GL_R16 / GL_RGB16
                // as-is instead of downgrading them to float formats in
                // internal_convert(). Was declared in gles_caps_t but never
                // detected before, so the GL_RGBA16 -> GL_RGBA16F downgrade
                // always ran even on devices that natively support norm16.
                g_gles_caps.GL_EXT_texture_norm16 = 1;
            } else if (strcmp(extension, "GL_EXT_texture_rg") == 0) {
                g_gles_caps.GL_EXT_texture_rg = 1;
            } else if (strcmp(extension, "GL_OES_depth_texture") == 0) {
                g_gles_caps.GL_OES_depth_texture = 1;
            } else if (strcmp(extension, "GL_OES_depth24") == 0) {
                g_gles_caps.GL_OES_depth24 = 1;
            } else if (strcmp(extension, "GL_OES_depth_texture_float") == 0) {
                g_gles_caps.GL_OES_depth_texture_float = 1;
            }
        } else {
            LOG_D("(nullptr)")
        }
    }

    LOG_I("%sDetected GL_EXT_multi_draw_indirect!", g_gles_caps.GL_EXT_multi_draw_indirect ? "" : "Not ")
    LOG_I("Enable-capability extensions: multisample_compatibility=%d clip_cull_distance=%d depth_clamp=%d "
          "sRGB_write_control=%d NV_polygon_mode=%d OES_sample_shading=%d",
          g_gles_caps.GL_EXT_multisample_compatibility, g_gles_caps.GL_EXT_clip_cull_distance,
          g_gles_caps.GL_EXT_depth_clamp, g_gles_caps.GL_EXT_sRGB_write_control, g_gles_caps.GL_NV_polygon_mode,
          g_gles_caps.GL_OES_sample_shading)
    LOG_I("%sDetected GL_EXT_color_buffer_float!", g_gles_caps.GL_EXT_color_buffer_float ? "" : "Not ")
    LOG_I("%sDetected GL_EXT_color_buffer_half_float!", g_gles_caps.GL_EXT_color_buffer_half_float ? "" : "Not ")
    LOG_I("%sDetected GL_EXT_texture_norm16!", g_gles_caps.GL_EXT_texture_norm16 ? "" : "Not ")

    // ---- Map optional ES extensions to desktop GL extensions ----

    // Advertise ARB_buffer_storage only when the driver can actually back it
    // up. Three things have to hold, and they are independent of each other:
    //   1. the extension string (from glGetStringi)
    //   2. the entry point (from eglGetProcAddress)
    //   3. a working persistent mapping (proven, not assumed)
    // Failing 1 or 2 makes glBufferStorage a silent no-op: no storage, no GL
    // error. Failing 3 does allocate the storage but leaves the first
    // glMapBufferRange returning NULL. Either way the game dies in
    // RenderSystem.initRenderer with "Failed to map buffer".
    LOG_I("%sDetected GL_EXT_buffer_storage! (glBufferStorageEXT %s)",
          g_gles_caps.GL_EXT_buffer_storage ? "" : "Not ", GLES.glBufferStorageEXT ? "resolved" : "MISSING")

    const bool haveStorageExt = g_gles_caps.GL_EXT_buffer_storage && GLES.glBufferStorageEXT;
    if (g_gles_caps.GL_EXT_buffer_storage && !GLES.glBufferStorageEXT) {
        LOG_W_FORCE("GL_EXT_buffer_storage is advertised but glBufferStorageEXT could not be resolved; "
                    "withholding GL_ARB_buffer_storage so callers use mutable buffers")
    } else if (haveStorageExt) {
        const PersistentProbe probe = ProbePersistentMapping();
        for (int t = 0; t < kProbeTargetCount; ++t) {
            LOG_I("Persistent map probe [%s]: storage=%s(0x%x) E2=%s(0x%x) C2=%s(0x%x) W=%s(0x%x)",
                  kProbeTargetNames[t], probe.storageOk[t] ? "ok" : "FAILED", probe.storageErr[t],
                  probe.mapOk[t][0] ? "ok" : "FAILED", probe.mapErr[t][0], probe.mapOk[t][1] ? "ok" : "FAILED",
                  probe.mapErr[t][1], probe.mapOk[t][2] ? "ok" : "FAILED", probe.mapErr[t][2])
        }

        if (probe.WorksOnGameTarget()) {
            if (!probe.mapOk[1][0] && probe.mapOk[1][1]) {
                LOG_I("Persistent mapping needs UNSYNCHRONIZED stripped; buffer.cpp already does this")
            }
            AppendExtension("GL_ARB_buffer_storage");
        } else {
            LOG_W_FORCE("Persistent mapping is unusable on GL_UNIFORM_BUFFER (storage %s, E2 and C2 both failed); "
                        "withholding GL_ARB_buffer_storage so the game allocates mutable buffers",
                        probe.storageOk[1] ? "ok" : "FAILED")
        }
    }

    if (g_gles_caps.EXT_disjoint_timer_query && global_settings.ext_timer_query) {
        AppendExtension("GL_ARB_timer_query");
        AppendExtension("GL_EXT_timer_query");
        AppendExtension("GL_EXT_disjoint_timer_query");
    }

    // ASTC texture compression is core in ES 3.2 — always advertise it
    if (g_gles_caps.major > 3 || (g_gles_caps.major == 3 && g_gles_caps.minor >= 2)) {
        AppendExtension("GL_KHR_texture_compression_astc_ldr");
    }

    // Anisotropic texture filtering — only if detected on the GLES side
    if (g_gles_caps.EXT_texture_filter_anisotropic) {
        // Report the real ceiling. Minecraft 26.3 validates the sampler's
        // anisotropy against it and throws when the range is empty:
        //     IllegalArgumentException: maxAnisotropy out of range;
        //     must be >= 1 and <= 0, but was 1
        // so a host that leaves it at 0 is worth naming here, in the part of
        // the log that survives — gl/getter.cpp substitutes 16 in that case.
        GLint max_anisotropy = 0;
        if (GLES.glGetIntegerv) GLES.glGetIntegerv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &max_anisotropy);
        LOG_I("%sDetected GL_EXT_texture_filter_anisotropic! (maxAnisotropy=%d%s)",
              g_gles_caps.EXT_texture_filter_anisotropic ? "" : "Not ", max_anisotropy,
              max_anisotropy > 0 ? "" : " — host did not answer, 16 will be substituted")
        AppendExtension("GL_EXT_texture_filter_anisotropic");
    }

    if (global_settings.ext_compute_shader) {
        // Compute shaders are core in ES 3.2 — always advertise them.
        AppendExtension("GL_ARB_compute_shader");
    }

    if (global_settings.ext_direct_state_access) {
        AppendExtension("GL_ARB_direct_state_access");
        AppendExtension("GL_EXT_direct_state_access");
    }

    // Append OpenGL version strings for the range 32..GLVersion
    int glVersion = GLVersion.toInt(2);
    for (int ver = 32; ver <= glVersion; ++ver) {
        if (ver > 33 && ver < 40) continue;
        LOG_D("Appending OpenGL extension for version %d", ver)
        AppendExtension(("OpenGL" + std::to_string(ver)).c_str());
    }

    // ES 3.2 includes vertex attrib binding as core — always advertise it.
    if (g_gles_caps.major > 3 || (g_gles_caps.major == 3 && g_gles_caps.minor >= 1)) {
        AppendExtension("GL_ARB_vertex_attrib_binding");
    }
}

// ---------------------------------------------------------------------------
// ES 3.2 Function Pointer Initialization
// All ES 3.2 core entry points are loaded here at startup.
// ---------------------------------------------------------------------------

void init_target_gles() {
    init_gl_state();

    memset(&g_gles_func, 0, sizeof(g_gles_func));

    // ---- ES 2.0 core functions ----
    INIT_GLES_FUNC(glActiveTexture)
    INIT_GLES_FUNC(glAttachShader)
    INIT_GLES_FUNC(glBindAttribLocation)
    INIT_GLES_FUNC(glBindBuffer)
    INIT_GLES_FUNC(glBindFramebuffer)
    INIT_GLES_FUNC(glBindRenderbuffer)
    INIT_GLES_FUNC(glBindTexture)
    INIT_GLES_FUNC(glBlendColor)
    INIT_GLES_FUNC(glBlendEquation)
    INIT_GLES_FUNC(glBlendEquationSeparate)
    INIT_GLES_FUNC(glBlendFunc)
    INIT_GLES_FUNC(glBlendFuncSeparate)
    INIT_GLES_FUNC(glBufferData)
    INIT_GLES_FUNC(glBufferSubData)
    INIT_GLES_FUNC(glCheckFramebufferStatus)
    INIT_GLES_FUNC(glClear)
    INIT_GLES_FUNC(glClearColor)
    INIT_GLES_FUNC(glClearDepthf)
    INIT_GLES_FUNC(glClearStencil)
    INIT_GLES_FUNC(glColorMask)
    INIT_GLES_FUNC(glCompileShader)
    INIT_GLES_FUNC(glCompressedTexImage2D)
    INIT_GLES_FUNC(glCompressedTexSubImage2D)
    INIT_GLES_FUNC(glCopyTexImage2D)
    INIT_GLES_FUNC(glCopyTexSubImage2D)
    INIT_GLES_FUNC(glCreateProgram)
    INIT_GLES_FUNC(glCreateShader)
    INIT_GLES_FUNC(glCullFace)
    INIT_GLES_FUNC(glDeleteBuffers)
    INIT_GLES_FUNC(glDeleteFramebuffers)
    INIT_GLES_FUNC(glDeleteProgram)
    INIT_GLES_FUNC(glDeleteRenderbuffers)
    INIT_GLES_FUNC(glDeleteShader)
    INIT_GLES_FUNC(glDeleteTextures)
    INIT_GLES_FUNC(glDepthFunc)
    INIT_GLES_FUNC(glDepthMask)
    INIT_GLES_FUNC(glDepthRangef)
    INIT_GLES_FUNC(glDetachShader)
    INIT_GLES_FUNC(glDisable)
    INIT_GLES_FUNC(glDisableVertexAttribArray)
    INIT_GLES_FUNC(glDrawArrays)
    INIT_GLES_FUNC(glDrawElements)
    INIT_GLES_FUNC(glEnable)
    INIT_GLES_FUNC(glEnableVertexAttribArray)
    INIT_GLES_FUNC(glFinish)
    INIT_GLES_FUNC(glFlush)
    INIT_GLES_FUNC(glFramebufferRenderbuffer)
    INIT_GLES_FUNC(glFramebufferTexture2D)
    INIT_GLES_FUNC(glFrontFace)
    INIT_GLES_FUNC(glGenBuffers)
    INIT_GLES_FUNC(glGenerateMipmap)
    INIT_GLES_FUNC(glGenFramebuffers)
    INIT_GLES_FUNC(glGenRenderbuffers)
    INIT_GLES_FUNC(glGenTextures)
    INIT_GLES_FUNC(glGetActiveAttrib)
    INIT_GLES_FUNC(glGetActiveUniform)
    INIT_GLES_FUNC(glGetAttachedShaders)
    INIT_GLES_FUNC(glGetAttribLocation)
    INIT_GLES_FUNC(glGetBooleanv)
    INIT_GLES_FUNC(glGetBufferParameteriv)
    INIT_GLES_FUNC(glGetError)
    INIT_GLES_FUNC(glGetString)
    INIT_GLES_FUNC(glGetStringi)
    INIT_GLES_FUNC(glGetFloatv)
    INIT_GLES_FUNC(glGetFramebufferAttachmentParameteriv)
    INIT_GLES_FUNC(glGetIntegerv)
    INIT_GLES_FUNC(glGetProgramiv)
    INIT_GLES_FUNC(glGetProgramInfoLog)
    INIT_GLES_FUNC(glGetRenderbufferParameteriv)
    INIT_GLES_FUNC(glGetShaderiv)
    INIT_GLES_FUNC(glGetShaderInfoLog)
    INIT_GLES_FUNC(glGetShaderPrecisionFormat)
    INIT_GLES_FUNC(glGetShaderSource)
    INIT_GLES_FUNC(glGetTexParameterfv)
    INIT_GLES_FUNC(glGetTexParameteriv)
    INIT_GLES_FUNC(glGetUniformfv)
    INIT_GLES_FUNC(glGetUniformiv)
    INIT_GLES_FUNC(glGetUniformLocation)
    INIT_GLES_FUNC(glGetVertexAttribfv)
    INIT_GLES_FUNC(glGetVertexAttribiv)
    INIT_GLES_FUNC(glGetVertexAttribPointerv)
    INIT_GLES_FUNC(glHint)
    INIT_GLES_FUNC(glIsBuffer)
    INIT_GLES_FUNC(glIsEnabled)
    INIT_GLES_FUNC(glIsFramebuffer)
    INIT_GLES_FUNC(glIsProgram)
    INIT_GLES_FUNC(glIsRenderbuffer)
    INIT_GLES_FUNC(glIsShader)
    INIT_GLES_FUNC(glIsTexture)
    INIT_GLES_FUNC(glLineWidth)
    INIT_GLES_FUNC(glLinkProgram)
    INIT_GLES_FUNC(glPixelStorei)
    INIT_GLES_FUNC(glPolygonOffset)
    INIT_GLES_FUNC(glReadPixels)
    INIT_GLES_FUNC(glReleaseShaderCompiler)
    INIT_GLES_FUNC(glRenderbufferStorage)
    INIT_GLES_FUNC(glSampleCoverage)
    INIT_GLES_FUNC(glScissor)
    INIT_GLES_FUNC(glShaderBinary)
    INIT_GLES_FUNC(glShaderSource)
    INIT_GLES_FUNC(glStencilFunc)
    INIT_GLES_FUNC(glStencilFuncSeparate)
    INIT_GLES_FUNC(glStencilMask)
    INIT_GLES_FUNC(glStencilMaskSeparate)
    INIT_GLES_FUNC(glStencilOp)
    INIT_GLES_FUNC(glStencilOpSeparate)
    INIT_GLES_FUNC(glTexImage2D)
    INIT_GLES_FUNC(glTexParameterf)
    INIT_GLES_FUNC(glTexParameterfv)
    INIT_GLES_FUNC(glTexParameteri)
    INIT_GLES_FUNC(glTexParameteriv)
    INIT_GLES_FUNC(glTexSubImage2D)
    INIT_GLES_FUNC(glUniform1f)
    INIT_GLES_FUNC(glUniform1fv)
    INIT_GLES_FUNC(glUniform1i)
    INIT_GLES_FUNC(glUniform1iv)
    INIT_GLES_FUNC(glUniform2f)
    INIT_GLES_FUNC(glUniform2fv)
    INIT_GLES_FUNC(glUniform2i)
    INIT_GLES_FUNC(glUniform2iv)
    INIT_GLES_FUNC(glUniform3f)
    INIT_GLES_FUNC(glUniform3fv)
    INIT_GLES_FUNC(glUniform3i)
    INIT_GLES_FUNC(glUniform3iv)
    INIT_GLES_FUNC(glUniform4f)
    INIT_GLES_FUNC(glUniform4fv)
    INIT_GLES_FUNC(glUniform4i)
    INIT_GLES_FUNC(glUniform4iv)
    INIT_GLES_FUNC(glUniformMatrix2fv)
    INIT_GLES_FUNC(glUniformMatrix3fv)
    INIT_GLES_FUNC(glUniformMatrix4fv)
    INIT_GLES_FUNC(glUseProgram)
    INIT_GLES_FUNC(glValidateProgram)
    INIT_GLES_FUNC(glVertexAttrib1f)
    INIT_GLES_FUNC(glVertexAttrib1fv)
    INIT_GLES_FUNC(glVertexAttrib2f)
    INIT_GLES_FUNC(glVertexAttrib2fv)
    INIT_GLES_FUNC(glVertexAttrib3f)
    INIT_GLES_FUNC(glVertexAttrib3fv)
    INIT_GLES_FUNC(glVertexAttrib4f)
    INIT_GLES_FUNC(glVertexAttrib4fv)
    INIT_GLES_FUNC(glVertexAttribPointer)
    INIT_GLES_FUNC(glViewport)

    // ---- ES 3.0 core functions ----
    INIT_GLES_FUNC(glReadBuffer)
    INIT_GLES_FUNC(glDrawRangeElements)
    INIT_GLES_FUNC(glTexImage3D)
    INIT_GLES_FUNC(glTexSubImage3D)
    INIT_GLES_FUNC(glCopyTexSubImage3D)
    INIT_GLES_FUNC(glCompressedTexImage3D)
    INIT_GLES_FUNC(glCompressedTexSubImage3D)
    INIT_GLES_FUNC(glGenQueries)
    INIT_GLES_FUNC(glDeleteQueries)
    INIT_GLES_FUNC(glIsQuery)
    INIT_GLES_FUNC(glBeginQuery)
    INIT_GLES_FUNC(glEndQuery)
    INIT_GLES_FUNC(glGetQueryiv)
    INIT_GLES_FUNC(glGetQueryObjectuiv)
    INIT_GLES_FUNC(glUnmapBuffer)
    INIT_GLES_FUNC(glGetBufferPointerv)
    INIT_GLES_FUNC(glDrawBuffers)
    INIT_GLES_FUNC(glUniformMatrix2x3fv)
    INIT_GLES_FUNC(glUniformMatrix3x2fv)
    INIT_GLES_FUNC(glUniformMatrix2x4fv)
    INIT_GLES_FUNC(glUniformMatrix4x2fv)
    INIT_GLES_FUNC(glUniformMatrix3x4fv)
    INIT_GLES_FUNC(glUniformMatrix4x3fv)
    INIT_GLES_FUNC(glBlitFramebuffer)
    INIT_GLES_FUNC(glRenderbufferStorageMultisample)
    INIT_GLES_FUNC(glFramebufferTextureLayer)
    INIT_GLES_FUNC(glFlushMappedBufferRange)
    INIT_GLES_FUNC(glBindVertexArray)
    INIT_GLES_FUNC(glDeleteVertexArrays)
    INIT_GLES_FUNC(glGenVertexArrays)
    INIT_GLES_FUNC(glIsVertexArray)
    INIT_GLES_FUNC(glGetIntegeri_v)
    INIT_GLES_FUNC(glBeginTransformFeedback)
    INIT_GLES_FUNC(glEndTransformFeedback)
    INIT_GLES_FUNC(glBindBufferRange)
    INIT_GLES_FUNC(glBindBufferBase)
    INIT_GLES_FUNC(glTransformFeedbackVaryings)
    INIT_GLES_FUNC(glGetTransformFeedbackVarying)
    INIT_GLES_FUNC(glVertexAttribIPointer)
    INIT_GLES_FUNC(glGetVertexAttribIiv)
    INIT_GLES_FUNC(glGetVertexAttribIuiv)
    INIT_GLES_FUNC(glVertexAttribI4i)
    INIT_GLES_FUNC(glVertexAttribI4ui)
    INIT_GLES_FUNC(glVertexAttribI4iv)
    INIT_GLES_FUNC(glVertexAttribI4uiv)
    INIT_GLES_FUNC(glGetUniformuiv)
    INIT_GLES_FUNC(glGetFragDataLocation)
    INIT_GLES_FUNC(glUniform1ui)
    INIT_GLES_FUNC(glUniform2ui)
    INIT_GLES_FUNC(glUniform3ui)
    INIT_GLES_FUNC(glUniform4ui)
    INIT_GLES_FUNC(glUniform1uiv)
    INIT_GLES_FUNC(glUniform2uiv)
    INIT_GLES_FUNC(glUniform3uiv)
    INIT_GLES_FUNC(glUniform4uiv)
    INIT_GLES_FUNC(glClearBufferiv)
    INIT_GLES_FUNC(glClearBufferuiv)
    INIT_GLES_FUNC(glClearBufferfv)
    INIT_GLES_FUNC(glClearBufferfi)
    INIT_GLES_FUNC(glCopyBufferSubData)
    INIT_GLES_FUNC(glGetUniformIndices)
    INIT_GLES_FUNC(glGetActiveUniformsiv)
    INIT_GLES_FUNC(glGetUniformBlockIndex)
    INIT_GLES_FUNC(glGetActiveUniformBlockiv)
    INIT_GLES_FUNC(glGetActiveUniformBlockName)
    INIT_GLES_FUNC(glUniformBlockBinding)
    INIT_GLES_FUNC(glDrawArraysInstanced)
    INIT_GLES_FUNC(glDrawElementsInstanced)
    INIT_GLES_FUNC(glFenceSync)
    INIT_GLES_FUNC(glIsSync)
    INIT_GLES_FUNC(glDeleteSync)
    INIT_GLES_FUNC(glClientWaitSync)
    INIT_GLES_FUNC(glWaitSync)
    INIT_GLES_FUNC(glGetInteger64v)
    INIT_GLES_FUNC(glGetSynciv)
    INIT_GLES_FUNC(glGetInteger64i_v)
    INIT_GLES_FUNC(glGetBufferParameteri64v)
    INIT_GLES_FUNC(glGenSamplers)
    INIT_GLES_FUNC(glDeleteSamplers)
    INIT_GLES_FUNC(glIsSampler)
    INIT_GLES_FUNC(glBindSampler)
    INIT_GLES_FUNC(glSamplerParameteri)
    INIT_GLES_FUNC(glSamplerParameteriv)
    INIT_GLES_FUNC(glSamplerParameterf)
    INIT_GLES_FUNC(glSamplerParameterfv)
    INIT_GLES_FUNC(glGetSamplerParameteriv)
    INIT_GLES_FUNC(glGetSamplerParameterfv)
    INIT_GLES_FUNC(glVertexAttribDivisor)
    INIT_GLES_FUNC(glBindTransformFeedback)
    INIT_GLES_FUNC(glDeleteTransformFeedbacks)
    INIT_GLES_FUNC(glGenTransformFeedbacks)
    INIT_GLES_FUNC(glIsTransformFeedback)
    INIT_GLES_FUNC(glPauseTransformFeedback)
    INIT_GLES_FUNC(glResumeTransformFeedback)
    INIT_GLES_FUNC(glGetProgramBinary)
    INIT_GLES_FUNC(glProgramBinary)
    INIT_GLES_FUNC(glProgramParameteri)
    INIT_GLES_FUNC(glInvalidateFramebuffer)
    INIT_GLES_FUNC(glInvalidateSubFramebuffer)
    INIT_GLES_FUNC(glTexStorage2D)
    INIT_GLES_FUNC(glTexStorage3D)
    INIT_GLES_FUNC(glGetInternalformativ)

    // ---- ES 3.1 core functions (compute shaders, indirect draw, SSBO) ----
    INIT_GLES_FUNC(glDispatchCompute)
    INIT_GLES_FUNC(glDispatchComputeIndirect)
    INIT_GLES_FUNC(glDrawArraysIndirect)
    INIT_GLES_FUNC(glDrawElementsIndirect)
    INIT_GLES_FUNC(glFramebufferParameteri)
    INIT_GLES_FUNC(glGetFramebufferParameteriv)
    INIT_GLES_FUNC(glGetProgramInterfaceiv)
    INIT_GLES_FUNC(glGetProgramResourceIndex)
    INIT_GLES_FUNC(glGetProgramResourceName)
    INIT_GLES_FUNC(glGetProgramResourceiv)
    INIT_GLES_FUNC(glGetProgramResourceLocation)
    INIT_GLES_FUNC(glUseProgramStages)
    INIT_GLES_FUNC(glActiveShaderProgram)
    INIT_GLES_FUNC(glCreateShaderProgramv)
    INIT_GLES_FUNC(glBindProgramPipeline)
    INIT_GLES_FUNC(glDeleteProgramPipelines)
    INIT_GLES_FUNC(glGenProgramPipelines)
    INIT_GLES_FUNC(glIsProgramPipeline)
    INIT_GLES_FUNC(glGetProgramPipelineiv)
    INIT_GLES_FUNC(glProgramUniform1i)
    INIT_GLES_FUNC(glProgramUniform2i)
    INIT_GLES_FUNC(glProgramUniform3i)
    INIT_GLES_FUNC(glProgramUniform4i)
    INIT_GLES_FUNC(glProgramUniform1ui)
    INIT_GLES_FUNC(glProgramUniform2ui)
    INIT_GLES_FUNC(glProgramUniform3ui)
    INIT_GLES_FUNC(glProgramUniform4ui)
    INIT_GLES_FUNC(glProgramUniform1f)
    INIT_GLES_FUNC(glProgramUniform2f)
    INIT_GLES_FUNC(glProgramUniform3f)
    INIT_GLES_FUNC(glProgramUniform4f)
    INIT_GLES_FUNC(glProgramUniform1iv)
    INIT_GLES_FUNC(glProgramUniform2iv)
    INIT_GLES_FUNC(glProgramUniform3iv)
    INIT_GLES_FUNC(glProgramUniform4iv)
    INIT_GLES_FUNC(glProgramUniform1uiv)
    INIT_GLES_FUNC(glProgramUniform2uiv)
    INIT_GLES_FUNC(glProgramUniform3uiv)
    INIT_GLES_FUNC(glProgramUniform4uiv)
    INIT_GLES_FUNC(glProgramUniform1fv)
    INIT_GLES_FUNC(glProgramUniform2fv)
    INIT_GLES_FUNC(glProgramUniform3fv)
    INIT_GLES_FUNC(glProgramUniform4fv)
    INIT_GLES_FUNC(glProgramUniformMatrix2fv)
    INIT_GLES_FUNC(glProgramUniformMatrix3fv)
    INIT_GLES_FUNC(glProgramUniformMatrix4fv)
    INIT_GLES_FUNC(glProgramUniformMatrix2x3fv)
    INIT_GLES_FUNC(glProgramUniformMatrix3x2fv)
    INIT_GLES_FUNC(glProgramUniformMatrix2x4fv)
    INIT_GLES_FUNC(glProgramUniformMatrix4x2fv)
    INIT_GLES_FUNC(glProgramUniformMatrix3x4fv)
    INIT_GLES_FUNC(glProgramUniformMatrix4x3fv)
    INIT_GLES_FUNC(glValidateProgramPipeline)
    INIT_GLES_FUNC(glGetProgramPipelineInfoLog)
    INIT_GLES_FUNC(glBindImageTexture)
    INIT_GLES_FUNC(glGetBooleani_v)
    INIT_GLES_FUNC(glMemoryBarrier)
    INIT_GLES_FUNC(glMemoryBarrierByRegion)
    INIT_GLES_FUNC(glTexStorage2DMultisample)
    INIT_GLES_FUNC(glGetMultisamplefv)
    INIT_GLES_FUNC(glSampleMaski)
    INIT_GLES_FUNC(glGetTexLevelParameteriv)
    INIT_GLES_FUNC(glGetTexLevelParameterfv)
    INIT_GLES_FUNC(glBindVertexBuffer)
    INIT_GLES_FUNC(glVertexAttribFormat)
    INIT_GLES_FUNC(glVertexAttribIFormat)
    INIT_GLES_FUNC(glVertexAttribBinding)
    INIT_GLES_FUNC(glVertexBindingDivisor)
    INIT_GLES_FUNC(glBlendBarrier)
    INIT_GLES_FUNC(glCopyImageSubData)

    // ---- ES 3.2 core functions (debug, geometry/tess, multisample, etc.) ----
    INIT_GLES_FUNC(glDebugMessageControl)
    INIT_GLES_FUNC(glDebugMessageInsert)
    INIT_GLES_FUNC(glDebugMessageCallback)
    INIT_GLES_FUNC(glGetDebugMessageLog)
    INIT_GLES_FUNC(glPushDebugGroup)
    INIT_GLES_FUNC(glPopDebugGroup)
    INIT_GLES_FUNC(glObjectLabel)
    INIT_GLES_FUNC(glGetObjectLabel)
    INIT_GLES_FUNC(glObjectPtrLabel)
    INIT_GLES_FUNC(glGetObjectPtrLabel)
    INIT_GLES_FUNC(glGetPointerv)
    INIT_GLES_FUNC(glEnablei)
    INIT_GLES_FUNC(glDisablei)
    INIT_GLES_FUNC(glBlendEquationi)
    INIT_GLES_FUNC(glBlendEquationSeparatei)
    INIT_GLES_FUNC(glBlendFunci)
    INIT_GLES_FUNC(glBlendFuncSeparatei)
    INIT_GLES_FUNC(glColorMaski)
    INIT_GLES_FUNC(glIsEnabledi)
    INIT_GLES_FUNC(glDrawElementsBaseVertex)
    INIT_GLES_FUNC(glDrawRangeElementsBaseVertex)
    INIT_GLES_FUNC(glDrawElementsInstancedBaseVertex)
    INIT_GLES_FUNC(glFramebufferTexture)
    INIT_GLES_FUNC(glPrimitiveBoundingBox)
    INIT_GLES_FUNC(glGetGraphicsResetStatus)
    INIT_GLES_FUNC(glReadnPixels)
    INIT_GLES_FUNC(glGetnUniformfv)
    INIT_GLES_FUNC(glGetnUniformiv)
    INIT_GLES_FUNC(glGetnUniformuiv)
    INIT_GLES_FUNC(glMinSampleShading)
    INIT_GLES_FUNC(glPatchParameteri)
    INIT_GLES_FUNC(glTexParameterIiv)
    INIT_GLES_FUNC(glTexParameterIuiv)
    INIT_GLES_FUNC(glGetTexParameterIiv)
    INIT_GLES_FUNC(glGetTexParameterIuiv)
    INIT_GLES_FUNC(glSamplerParameterIiv)
    INIT_GLES_FUNC(glSamplerParameterIuiv)
    INIT_GLES_FUNC(glGetSamplerParameterIiv)
    INIT_GLES_FUNC(glGetSamplerParameterIuiv)
    INIT_GLES_FUNC(glTexBuffer)
    INIT_GLES_FUNC(glTexBufferRange)
    INIT_GLES_FUNC(glTexStorage3DMultisample)
    INIT_GLES_FUNC(glMapBufferRange)

    // ---- Optional extension functions (may not be present on all ES 3.2 drivers) ----
    INIT_GLES_FUNC(glBufferStorageEXT)
    INIT_GLES_FUNC(glGetQueryObjectivEXT)
    INIT_GLES_FUNC(glGetQueryObjecti64vEXT)
    INIT_GLES_FUNC(glBindFragDataLocationEXT)
    INIT_GLES_FUNC(glMapBufferOES)

    INIT_GLES_FUNC(glMultiDrawArraysIndirectEXT)
    INIT_GLES_FUNC(glMultiDrawElementsIndirectEXT)
    INIT_GLES_FUNC(glMultiDrawElementsBaseVertexEXT)

    LOG_D("glMultiDrawArraysIndirectEXT() @ 0x%x", GLES.glMultiDrawArraysIndirectEXT)
    LOG_D("glMultiDrawElementsIndirectEXT() @ 0x%x", GLES.glMultiDrawElementsIndirectEXT)
    LOG_D("glMultiDrawElementsBaseVertexEXT() @ 0x%x", GLES.glMultiDrawElementsBaseVertexEXT)

    // ---- Hardware & capability initialization ----
    LOG_D("Initializing %s @ hardware", RENDERERNAME)
    set_hardware();

    InitGLESCapabilities();
    LogOpenGLExtensions();
}