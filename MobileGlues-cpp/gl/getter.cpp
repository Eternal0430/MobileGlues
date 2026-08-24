// MobileGLESWrapper - gl/getter.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header
//
// Architecture: "ES 3.2 native → native, ES 3.2 not native → CPU simulation"
//   - Queries that exist in ES 3.2 core are forwarded directly to GLES.
//   - Queries that do NOT exist in ES 3.2 core (desktop GL queries) are
//     simulated via CPU-side logic, caching, or synthetic responses.

#include "getter.h"
#include "buffer.h"
#include <string>
#include <format>
#include <vector>
#include <random>
#include "log.h"
#include "random_string_gen.h"

#define DEBUG 0

// =============================================================================
// Section: Global State
// =============================================================================

Version GLVersion;

// =============================================================================
// Section: glGetIntegerv
//   ES 3.2 native queries → forwarded to GLES.glGetIntegerv
//   Non-native (desktop) queries → CPU simulation
// =============================================================================

void glGetIntegerv(GLenum pname, GLint* params) {
    LOG()
    LOG_D("glGetIntegerv, pname: %s", glEnumToString(pname))
    switch (pname) {

    // -------------------------------------------------------------------------
    // GL_BACKEND_GETTER_MG offset: strip wrapper prefix and forward to GLES
    // -------------------------------------------------------------------------
    case GL_NUM_EXTENSIONS + GL_BACKEND_GETTER_MG:
        GLES.glGetIntegerv(pname - GL_BACKEND_GETTER_MG, params);
        return;

    // -------------------------------------------------------------------------
    // Desktop GL queries — CPU simulation (not in ES 3.2 core)
    // -------------------------------------------------------------------------

    case GL_CONTEXT_PROFILE_MASK:
        (*params) = GL_CONTEXT_CORE_PROFILE_BIT;
        break;

    case GL_NUM_EXTENSIONS: {
        static GLint num_extensions = -1;
        if (num_extensions == -1) {
            const GLubyte* ext_str = glGetString(GL_EXTENSIONS);
            if (ext_str) {
                std::string ext((const char*)ext_str);
                num_extensions = 0;
                // O(n) single-pass: find(char, offset) avoids the O(n²)
                // erase(0, pos+1) that was used previously.
                size_t offset = 0;
                while (true) {
                    size_t pos = ext.find(' ', offset);
                    if (pos == std::string::npos) {
                        if (offset < ext.size()) num_extensions++;
                        break;
                    }
                    num_extensions++;
                    offset = pos + 1;
                }
            } else {
                num_extensions = 0;
            }
        }
        (*params) = num_extensions;
        break;
    }

    case GL_MAJOR_VERSION:
        (*params) = GLVersion.Major;
        break;

    case GL_MINOR_VERSION:
        (*params) = GLVersion.Minor;
        break;

    case GL_MAX_TEXTURE_IMAGE_UNITS: {
        // The GLES driver's max texture image units is a static property
        // of the context and never changes after creation. Cache it to
        // avoid a GLES round-trip on every query (some mods query this
        // during per-frame state setup).
        static GLint cached = -1;
        if (cached == -1) [[unlikely]] {
            int es_params = 16;
            GLES.glGetIntegerv(pname, &es_params);
            cached = es_params * 2;
        }
        (*params) = cached;
        break;
    }

    case GL_MAX_VERTEX_ATTRIBS: {
        static GLint cached = -1;
        if (cached == -1) [[unlikely]] {
            GLES.glGetIntegerv(pname, &cached);
        }
        (*params) = cached;
        break;
    }

    case GL_MAX_DRAW_BUFFERS: {
        static GLint cached = -1;
        if (cached == -1) [[unlikely]] {
            GLES.glGetIntegerv(pname, &cached);
        }
        (*params) = cached;
        break;
    }

    case GL_MAX_VERTEX_UNIFORM_VECTORS: {
        static GLint cached = -1;
        if (cached == -1) [[unlikely]] {
            GLES.glGetIntegerv(pname, &cached);
        }
        (*params) = cached;
        break;
    }

    case GL_MAX_FRAGMENT_UNIFORM_VECTORS: {
        static GLint cached = -1;
        if (cached == -1) [[unlikely]] {
            GLES.glGetIntegerv(pname, &cached);
        }
        (*params) = cached;
        break;
    }

    case GL_MAX_VARYING_VECTORS: {
        static GLint cached = -1;
        if (cached == -1) [[unlikely]] {
            GLES.glGetIntegerv(pname, &cached);
        }
        (*params) = cached;
        break;
    }

    case GL_MAX_TEXTURE_SIZE: {
        static GLint cached = -1;
        if (cached == -1) [[unlikely]] {
            GLES.glGetIntegerv(pname, &cached);
        }
        (*params) = cached;
        break;
    }

    case GL_MAX_CUBE_MAP_TEXTURE_SIZE: {
        static GLint cached = -1;
        if (cached == -1) [[unlikely]] {
            GLES.glGetIntegerv(pname, &cached);
        }
        (*params) = cached;
        break;
    }

    case GL_MAX_RENDERBUFFER_SIZE: {
        static GLint cached = -1;
        if (cached == -1) [[unlikely]] {
            GLES.glGetIntegerv(pname, &cached);
        }
        (*params) = cached;
        break;
    }

    case GL_MAX_SAMPLES: {
        static GLint cached = -1;
        if (cached == -1) [[unlikely]] {
            GLES.glGetIntegerv(pname, &cached);
        }
        (*params) = cached;
        break;
    }

    case GL_CONTEXT_FLAGS:
        (*params) = GL_CONTEXT_FLAG_ROBUST_ACCESS_BIT
                  | GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT
                  | GL_CONTEXT_FLAG_NO_ERROR_BIT;
        break;

    // -------------------------------------------------------------------------
    // Buffer binding queries — CPU simulation via our own binding tracking
    // -------------------------------------------------------------------------
    case GL_ARRAY_BUFFER_BINDING:
    case GL_ATOMIC_COUNTER_BUFFER_BINDING:
    case GL_COPY_READ_BUFFER_BINDING:
    case GL_COPY_WRITE_BUFFER_BINDING:
    case GL_DRAW_INDIRECT_BUFFER_BINDING:
    case GL_DISPATCH_INDIRECT_BUFFER_BINDING:
    case GL_ELEMENT_ARRAY_BUFFER_BINDING:
    case GL_PIXEL_PACK_BUFFER_BINDING:
    case GL_PIXEL_UNPACK_BUFFER_BINDING:
    case GL_SHADER_STORAGE_BUFFER_BINDING:
    case GL_TRANSFORM_FEEDBACK_BUFFER_BINDING:
    case GL_UNIFORM_BUFFER_BINDING:
        (*params) = (int)find_bound_buffer(pname);
        LOG_D("  -> %d", *params)
        break;

    // -------------------------------------------------------------------------
    // VAO binding query — CPU simulation via our own binding tracking
    // -------------------------------------------------------------------------
    case GL_VERTEX_ARRAY_BINDING:
        (*params) = (int)find_bound_array();
        break;

    // -------------------------------------------------------------------------
    // All other queries — forward to native ES 3.2 GLES
    // -------------------------------------------------------------------------
    default:
        GLES.glGetIntegerv(pname, params);
        LOG_D("  -> %d", *params)
        CHECK_GL_ERROR
    }
}

// =============================================================================
// Section: glGetError
//   Always returns GL_NO_ERROR. Internal GLES errors are silently consumed.
// =============================================================================

GLenum glGetError() {
    LOG()
#if GLOBAL_DEBUG
    // In debug mode, consume and report real GLES errors.
    // In release mode, skip the glGetError() GPU round-trip entirely —
    // glGetError is an implicit glFinish on many drivers, causing a
    // full pipeline stall. Since we always return GL_NO_ERROR to the
    // caller, there is no need to query the real error queue.
    GLenum err = GLES.glGetError();
    if (err != GL_NO_ERROR) {
        LOG_W("glGetError\n -> %d", err)
        LOG_W("Now try to cheat.")
    }
#endif
    return GL_NO_ERROR;
}

// =============================================================================
// Section: Extension Management
// =============================================================================

static std::string es_ext;

const std::string& GetExtensionsList() {
    // es_ext is built once in InitGLESBaseExtensions() and is stable thereafter;
    // return by reference so callers can read its buffer without a per-query copy.
    return es_ext;
}

void InitGLESBaseExtensions() {
    std::vector<std::string> extensions;

    if (global_settings.hide_mg_env_level == HideMGEnvLevel::Disabled) {
        extensions.push_back("GL_MG_mobileglues");
        extensions.push_back("GL_MG_backend_string_getter_access");
        extensions.push_back("GL_MG_settings_string_dump");
    }

    const char* base_exts[] = {
        "GL_ARB_fragment_program",
        "GL_ARB_vertex_buffer_object",
        "GL_ARB_vertex_array_object",
        "GL_ARB_vertex_buffer",
        "GL_EXT_vertex_array",
        "GL_ARB_ES2_compatibility",
        "GL_ARB_ES3_compatibility",
        "GL_EXT_packed_depth_stencil",
        "GL_EXT_depth_texture",
        "GL_ARB_depth_texture",
        "GL_ARB_shading_language_100",
        "GL_ARB_imaging",
        "GL_ARB_draw_buffers_blend",
        "OpenGL15",
        "GL_ARB_shader_storage_buffer_object",
        "GL_ARB_shader_image_load_store",
        "GL_ARB_clear_texture",
        "GL_ARB_get_program_binary",
        "GL_ARB_separate_shader_objects",
        "GL_ARB_multi_bind",
        "GL_KHR_no_error",
    };

    extensions.insert(extensions.end(), std::begin(base_exts), std::end(base_exts));

    if (global_settings.hide_mg_env_level >= HideMGEnvLevel::Level1) {
        for (int i = extensions.size() - 1; i > 0; --i) {
            int j = rand() % (i + 1);
            std::swap(extensions[i], extensions[j]);
        }
    }

    // Pre-allocate capacity to avoid reallocations during AppendExtension calls
    es_ext.clear();
    es_ext.reserve(4096);
    for (const auto& ext : extensions) {
        es_ext += ext;
        es_ext += " ";
    }
}

void AppendExtension(const char* ext) {
    es_ext += ext;
    es_ext += ' ';
}

// =============================================================================
// Section: GPU Name Helpers
// =============================================================================

static std::string getBeforeThirdSpace(const std::string& str) {
    int spaceCount = 0;
    size_t endPos = 0;
    for (size_t i = 0; i < str.length(); ++i) {
        if (str[i] == ' ') {
            spaceCount++;
            if (spaceCount == 3) {
                endPos = i;
                break;
            }
        }
        if (spaceCount < 3) endPos = str.length();
    }
    return str.substr(0, endPos);
}

static std::string getGpuName() {
    std::string gpuName = std::string((char*)GLES.glGetString(GL_RENDERER));

    if (gpuName.empty()) {
        return "<unknown>";
    }

    // MetalANGLE, ANGLE (Metal Renderer: Apple * GPU)
    if (gpuName.find("MetalANGLE, ANGLE") != std::string::npos) {
        if (gpuName.length() < 25) {
            return gpuName;
        }
        std::string gpu = gpuName.substr(23, gpuName.length() - 24);
        return gpu + " | MetalANGLE | Metal";
    }

    // Vulkan ANGLE
    if (gpuName.rfind("ANGLE", 0) == 0 && gpuName.find("Vulkan") != std::string::npos) {
        size_t firstParen = gpuName.find('(');
        size_t secondParen = gpuName.find('(', firstParen + 1);
        size_t lastParen = gpuName.rfind('(');
        std::string gpu = gpuName.substr(secondParen + 1, lastParen - secondParen - 2);

        size_t vulkanStart = gpuName.find("Vulkan ");
        size_t vulkanEnd = gpuName.find(' ', vulkanStart + 7);
        std::string vulkanVersion = gpuName.substr(vulkanStart + 7, vulkanEnd - (vulkanStart + 7));

        return gpu + " | ANGLE | Vulkan " + vulkanVersion;
    }

    return gpuName;
}

static std::string getGLESName() {
    return getBeforeThirdSpace(std::string((char*)GLES.glGetString(GL_VERSION)));
}

void set_es_version() {
    std::string ESVersionStr = getBeforeThirdSpace(std::string((const char*)GLES.glGetString(GL_VERSION)));
    hardware->es_version = 320;
    LOG_I("OpenGL ES Version: %s (%d)", ESVersionStr.c_str(), hardware->es_version)
}

// =============================================================================
// Section: glGetString
//   Custom vendor / version / renderer / extensions strings (CPU simulation).
//   Non-overridden names → forwarded to GLES.glGetString (native).
// =============================================================================

static std::string rendererString;
static std::string vendorString;
static std::string versionString;

const GLubyte* glGetString(GLenum name) {
    LOG()
    LOG_D("glGetString, %s", glEnumToString(name))

    switch (name) {

    // -------------------------------------------------------------------------
    // GL_VENDOR — synthetic vendor string
    // -------------------------------------------------------------------------
    case GL_VENDOR: {
        if (vendorString.empty()) {
            if (global_settings.hide_mg_env_level == HideMGEnvLevel::Disabled) {
                vendorString = "Swung0x48, BZLZHH, Tungsten, EternityQwQ";
            } else {
                const char choices[] = "AINM";
                vendorString = choices[rand() % 4];

                RandomStringOptions randStrOpts;
                randStrOpts.includeDigits = false;
                randStrOpts.minLength = 3;
                randStrOpts.maxLength = 8;
                randStrOpts.includeLowercase = false;
                randStrOpts.includeUppercase = false;
                randStrOpts.customChars = "IMenaNtMseAVlD";
                vendorString += GenerateRandomString(randStrOpts);
            }
        }
        return (const GLubyte*)vendorString.c_str();
    }

    // -------------------------------------------------------------------------
    // GL_VERSION — synthetic version string
    // -------------------------------------------------------------------------
    case GL_VERSION: {
        if (versionString.empty()) {
            versionString = GLVersion.toString();
            if (global_settings.hide_mg_env_level == HideMGEnvLevel::Disabled) {
                if (GLVersion.toInt(2) == DEFAULT_GL_VERSION) {
                    versionString += " MobileGLESWrapper ";
                } else {
                    Version defaultVersion = Version(DEFAULT_GL_VERSION);
                    versionString += " §4§l(" + defaultVersion.toString() + ") MobileGLESWrapper§r ";
                }
                versionString += std::to_string(MAJOR) + "." + std::to_string(MINOR) + "." + std::to_string(REVISION);
#if PATCH != 0
                versionString += "." + std::to_string(PATCH);
#endif
#if defined(VERSION_TYPE)
#if VERSION_TYPE == VERSION_ALPHA
                versionString += "·Alpha";
#elif VERSION_TYPE == VERSION_BETA
                versionString += "·Beta";
#elif VERSION_TYPE == VERSION_DEVELOPMENT
                versionString += "·Dev";
#elif VERSION_TYPE == VERSION_RC
                versionString += "·RC" + std::to_string(VERSION_RC_NUMBER);
#endif
#endif
                versionString += VERSION_SUFFIX;
            } else {
                const char choices[] = "AIN";
                versionString += " ";
                versionString += choices[rand() % 3];

                RandomStringOptions randStrOpts;
                randStrOpts.includeDigits = false;
                randStrOpts.customChars = " ";
                versionString += GenerateRandomString(randStrOpts);

                RandomStringOptions randStrOpts2;
                randStrOpts2.includeDigits = false;
                randStrOpts2.includeUppercase = false;
                randStrOpts2.minLength = 1;
                randStrOpts2.maxLength = 4;

                versionString += std::to_string(MAJOR) + GenerateRandomString(randStrOpts2)
                               + std::to_string(MINOR) + GenerateRandomString(randStrOpts2)
                               + std::to_string(REVISION) + GenerateRandomString(randStrOpts2)
                               + std::to_string(PATCH) + GenerateRandomString(randStrOpts2);
            }
        }
        return (const GLubyte*)versionString.c_str();
    }

    // -------------------------------------------------------------------------
    // GL_RENDERER — synthetic renderer string from GPU + GLES names
    // -------------------------------------------------------------------------
    case GL_RENDERER: {
        if (rendererString.empty()) {
            if (global_settings.hide_mg_env_level == HideMGEnvLevel::Disabled) {
                std::string gpuName = getGpuName();
                std::string glesName = getGLESName();
                rendererString = gpuName + " | " + glesName;
            } else {
                const char choices[] = "AINM";
                rendererString = choices[rand() % 4];

                RandomStringOptions randStrOpts;
                randStrOpts.includeDigits = true;
                randStrOpts.minLength = 6;
                randStrOpts.maxLength = 12;
                randStrOpts.includeLowercase = false;
                randStrOpts.includeUppercase = false;
                randStrOpts.customChars = "IRMenaNtfsoerAceVlDG";
                rendererString += GenerateRandomString(randStrOpts);

                int junkInfoTime = rand() % 3 + 1;
                for (int i = 0; i < junkInfoTime; ++i) {
                    rendererString += " ";
                    RandomStringOptions randStrOpts2;
                    randStrOpts2.minLength = 3;
                    randStrOpts2.maxLength = 6;
                    randStrOpts2.includeLowercase = false;
                    randStrOpts2.includeUppercase = false;
                    randStrOpts2.customChars = "IRenaNtfsoerAcieVDcsG";
                    rendererString += GenerateRandomString(randStrOpts2);
                }
            }
        }
        return (const GLubyte*)rendererString.c_str();
    }

    // -------------------------------------------------------------------------
    // GL_SHADING_LANGUAGE_VERSION — synthetic shading language version
    // -------------------------------------------------------------------------
    case GL_SHADING_LANGUAGE_VERSION: {
        static std::string shadingLangString;
        if (shadingLangString.empty()) {
            std::string baseVer = "4.60";
            if (global_settings.hide_mg_env_level >= HideMGEnvLevel::Level1) {
                shadingLangString = baseVer;
                int junkCount = rand() % 2 + 1;
                for (int i = 0; i < junkCount; ++i) {
                    shadingLangString += " ";
                    RandomStringOptions junkOpts;
                    junkOpts.minLength = 2;
                    junkOpts.maxLength = 5;
                    junkOpts.includeLowercase = false;
                    junkOpts.includeUppercase = false;
                    junkOpts.customChars = "IAneNDtVsaMIl";
                    shadingLangString += GenerateRandomString(junkOpts);
                }
            } else {
                shadingLangString = baseVer + " MobileGLESWrapper with glslang and SPIRV-Cross";
            }
        }
        return reinterpret_cast<const GLubyte*>(shadingLangString.c_str());
    }

    // -------------------------------------------------------------------------
    // GL_EXTENSIONS — from our managed extension list
    // -------------------------------------------------------------------------
    case GL_EXTENSIONS: {
        // es_ext is stable after init; return its buffer directly, avoiding the
        // per-query std::string copy (and potential reallocation) the previous
        // static-cache assignment performed.
        return (const GLubyte*)GetExtensionsList().c_str();
    }

    // -------------------------------------------------------------------------
    // GL_SETTINGS_MG — dump settings string (MG custom query)
    // -------------------------------------------------------------------------
    case GL_SETTINGS_MG: {
        if (global_settings.hide_mg_env_level >= HideMGEnvLevel::Level1)
            return GLES.glGetString(name);

        static char* settings_string = nullptr;
        std::string tmp = dump_settings_string("  ");
        settings_string = strdup(tmp.c_str());
        return reinterpret_cast<const GLubyte*>(settings_string);
    }

    // -------------------------------------------------------------------------
    // GL_BACKEND_GETTER_MG offset queries — strip prefix and forward to GLES
    // -------------------------------------------------------------------------
    case GL_VERSION + GL_BACKEND_GETTER_MG:
    case GL_VENDOR + GL_BACKEND_GETTER_MG:
    case GL_RENDERER + GL_BACKEND_GETTER_MG:
    case GL_EXTENSIONS + GL_BACKEND_GETTER_MG:
    case GL_SHADING_LANGUAGE_VERSION + GL_BACKEND_GETTER_MG:
        if (global_settings.hide_mg_env_level == HideMGEnvLevel::Disabled)
            return GLES.glGetString(name - GL_BACKEND_GETTER_MG);
        else
            return GLES.glGetString(name);

    // -------------------------------------------------------------------------
    // All other string queries → forward to native GLES
    // -------------------------------------------------------------------------
    default:
        return GLES.glGetString(name);
    }
}

// =============================================================================
// Section: glGetStringi
//   CPU simulation: tokenizes the synthetic glGetString output into parts,
//   then returns the requested part by index.
// =============================================================================

const GLubyte* glGetStringi(GLenum name, GLuint index) {
    LOG()

    if (name == GL_EXTENSIONS + GL_BACKEND_GETTER_MG && global_settings.hide_mg_env_level == HideMGEnvLevel::Disabled) {
        return GLES.glGetStringi(name - GL_BACKEND_GETTER_MG, index);
    }

    typedef struct {
        GLenum name;
        const char** parts;
        GLuint count;
    } StringCache;

    static StringCache caches[] = {
        {GL_EXTENSIONS, nullptr, 0},
        {GL_VENDOR, nullptr, 0},
        {GL_VERSION, nullptr, 0},
        {GL_SHADING_LANGUAGE_VERSION, nullptr, 0},
    };

    static int initialized = 0;
    if (!initialized) {
        for (auto& cache : caches) {
            GLenum target = cache.name;
            const GLubyte* str = nullptr;
            const char* delimiter = " ";

            switch (target) {
            case GL_VENDOR:
                str = glGetString(GL_VENDOR);
                delimiter = ", ";
                break;
            case GL_VERSION:
                str = glGetString(GL_VERSION);
                delimiter = " .";
                break;
            case GL_SHADING_LANGUAGE_VERSION:
                str = glGetString(GL_SHADING_LANGUAGE_VERSION);
                break;
            case GL_EXTENSIONS:
                str = glGetString(GL_EXTENSIONS);
                break;
            default:
                return GLES.glGetStringi(name, index);
            }

            if (!str) continue;

            std::string copy_str((const char*)str);

            // First pass: count tokens so we can allocate the parts array
            // in a single malloc instead of O(n) realloc calls.
            GLuint token_count = 0;
            size_t pos = 0;
            while (true) {
                token_count++;
                size_t next = copy_str.find_first_of(delimiter, pos);
                if (next == std::string::npos) break;
                pos = next + 1;
            }

            // Single allocation for the pointer array
            cache.parts = (const char**)malloc(token_count * sizeof(char*));
            cache.count = token_count;

            // Second pass: extract tokens
            size_t start = 0;
            size_t end = copy_str.find_first_of(delimiter);
            GLuint idx = 0;

            while (end != std::string::npos) {
                std::string token = copy_str.substr(start, end - start);
                cache.parts[idx++] = strdup(token.c_str());
                start = end + 1;
                end = copy_str.find_first_of(delimiter, start);
            }
            std::string token = copy_str.substr(start);
            cache.parts[idx++] = strdup(token.c_str());
        }
        initialized = 1;
    }

    for (auto& cache : caches) {
        if (cache.name == name) {
            if (index >= cache.count) {
                return nullptr;
            }
            return (const GLubyte*)cache.parts[index];
        }
    }

    return nullptr;
}

// =============================================================================
// Section: glGetQueryObject — ES 3.2 native extensions
//   Forwarded to GLES if the extension is available.
// =============================================================================

void glGetQueryObjectiv(GLuint id, GLenum pname, GLint* params) {
    LOG()
    if (GLES.glGetQueryObjectivEXT) {
        GLES.glGetQueryObjectivEXT(id, pname, params);
        CHECK_GL_ERROR
    }
}

void glGetQueryObjecti64v(GLuint id, GLenum pname, GLint64* params) {
    LOG()
    if (GLES.glGetQueryObjecti64vEXT) {
        GLES.glGetQueryObjecti64vEXT(id, pname, params);
        CHECK_GL_ERROR
    }
}