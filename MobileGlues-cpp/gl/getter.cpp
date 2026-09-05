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
#include <mutex>
#include "log.h"
#include "random_string_gen.h"
#include "../egl/loader.h"

#define DEBUG 0

// ScopedHostContext now lives in egl/loader.h so that gl/buffer.cpp can use
// the same mechanism for host buffer operations.

// =============================================================================
// Section: Global State
// =============================================================================

Version GLVersion;

// =============================================================================
// Section: Host Limit Query Fallbacks
//
//   Minecraft 26.3's renderer sizes its dynamic GPU buffers with
//   Mth.roundToward(capacity, alignment), which is positiveCeilDiv(cap, align)
//   * align. If a device-limit query yields 0, that becomes Math.floorDiv(x, 0)
//   and the game dies with "ArithmeticException: / by zero" during
//   RenderSystem.initRenderer.
//
//   A host driver legitimately leaves params untouched when it does not know
//   the enum (GL_INVALID_ENUM, command ignored) or when the calling thread has
//   no current EGL context. The application's buffer is then still zero — it
//   never sees an error, just a nonsense limit.
//
//   The guards below catch that at the boundary: retry under MobileGLES'
//   fallback context if this thread had none, and otherwise substitute a safe
//   value. Every repair is written to the log so the offending enum is
//   identifiable from latest.log.
// =============================================================================

namespace limitguard {

// One entry per limit we are willing to synthesize. `substitute` is a query
// GLES 3.2 always supports, tried before the hard-coded number — so the value
// still reflects the real GPU when only the fancy enum is missing.
// The numbers are conservative order-of-magnitude values for ES 3.2 class
// hardware, not guaranteed spec minima.
struct LimitFallback {
    GLenum pname;
    const char* name;
    GLenum substitute; // 0 = none
    GLint fallback;
};

// Notes on the values: anything in the ES 3.2 core set uses that spec's
// mandated minimum, so a substitute can never claim more than the host is
// guaranteed to deliver. The desktop-only limits use conservative values that
// are typical rather than maximal — they only have to be plausible, since the
// real number is used whenever the host answers at all.
const LimitFallback kLimitFallbacks[] = {
    // --- Buffer offset alignment. Minecraft 26.3 feeds minUniformOffsetAlignment
    // --- straight into Mth.roundToward(), so a 0 here is a hard crash.
    {GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, "GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT", 0, 256},
    {GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, "GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT",
     GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, 256},
    {GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT, "GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT", GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, 256},

    // --- Sizes and binding counts (ES 3.2 mandated minima unless noted).
    {GL_MAX_UNIFORM_BLOCK_SIZE, "GL_MAX_UNIFORM_BLOCK_SIZE", 0, 16384},
    {GL_MAX_SHADER_STORAGE_BLOCK_SIZE, "GL_MAX_SHADER_STORAGE_BLOCK_SIZE", GL_MAX_UNIFORM_BLOCK_SIZE, 16777216},
    {GL_MAX_TEXTURE_BUFFER_SIZE, "GL_MAX_TEXTURE_BUFFER_SIZE", 0, 65536},
    {GL_MAX_UNIFORM_BUFFER_BINDINGS, "GL_MAX_UNIFORM_BUFFER_BINDINGS", 0, 24},
    {GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, "GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS", GL_MAX_UNIFORM_BUFFER_BINDINGS,
     8},

    // --- Limits MobileGlues caches. These are queried once at start-up and the
    // --- result was kept forever, so a query that happened to run without a
    // --- current context used to poison them with 0 for the whole session.
    {GL_MAX_VERTEX_ATTRIBS, "GL_MAX_VERTEX_ATTRIBS", 0, 16},
    {GL_MAX_DRAW_BUFFERS, "GL_MAX_DRAW_BUFFERS", 0, 8},
    {GL_MAX_SAMPLES, "GL_MAX_SAMPLES", 0, 4},
    {GL_MAX_TEXTURE_IMAGE_UNITS, "GL_MAX_TEXTURE_IMAGE_UNITS", 0, 16},
    {GL_MAX_VERTEX_UNIFORM_VECTORS, "GL_MAX_VERTEX_UNIFORM_VECTORS", 0, 256},
    {GL_MAX_FRAGMENT_UNIFORM_VECTORS, "GL_MAX_FRAGMENT_UNIFORM_VECTORS", 0, 224},
    {GL_MAX_VARYING_VECTORS, "GL_MAX_VARYING_VECTORS", 0, 15},
    {GL_MAX_TEXTURE_SIZE, "GL_MAX_TEXTURE_SIZE", 0, 4096},
    {GL_MAX_CUBE_MAP_TEXTURE_SIZE, "GL_MAX_CUBE_MAP_TEXTURE_SIZE", 0, 4096},
    {GL_MAX_RENDERBUFFER_SIZE, "GL_MAX_RENDERBUFFER_SIZE", 0, 4096},

    // --- Anisotropic filtering. Minecraft 26.3 validates the sampler's
    // --- anisotropy against this limit and throws outright when the range is
    // --- empty:
    // ---     IllegalArgumentException: maxAnisotropy out of range;
    // ---     must be >= 1 and <= 0, but was 1
    // --- So unlike the entries above, a 0 here is a hard crash rather than
    // --- degraded visuals. 16 is what virtually every ES 3.2 mobile GPU
    // --- reports; a host that supports less clamps the value itself, which
    // --- the extension explicitly allows.
    {GL_MAX_TEXTURE_MAX_ANISOTROPY, "GL_MAX_TEXTURE_MAX_ANISOTROPY", 0, 16},
};

const LimitFallback* FindLimitFallback(GLenum pname) {
    for (const LimitFallback& entry : kLimitFallbacks) {
        if (entry.pname == pname) return &entry;
    }
    return nullptr;
}

// Consumes the host error queue and reports whether anything was in it. Used to
// tell "driver rejected this enum" apart from "driver answered, and 0 is real".
bool ConsumeHostErrors() {
    if (!GLES.glGetError) return false;
    bool any = false;
    while (GLES.glGetError() != GL_NO_ERROR) any = true;
    return any;
}

GLint ResolveLimitFallback(const LimitFallback& entry) {
    GLint value = 0;
    if (entry.substitute && GLES.glGetIntegerv) {
        GLES.glGetIntegerv(entry.substitute, &value);
        if (value > 0) {
            ConsumeHostErrors();
            return value;
        }
        value = 0;
    }
    return entry.fallback;
}

// Diagnostics: report each offending pname once. Races can at worst produce a
// duplicate line, which is harmless for a log file.
constexpr int kMaxReportedPnames = 32;
GLenum g_reported_pnames[kMaxReportedPnames];
int g_reported_count = 0;
bool g_reported_cap_hit = false;

bool ReportOnce(GLenum pname) {
    for (int i = 0; i < g_reported_count; ++i) {
        if (g_reported_pnames[i] == pname) return false;
    }
    if (g_reported_count >= kMaxReportedPnames) {
        if (!g_reported_cap_hit) {
            g_reported_cap_hit = true;
            LOG_W_FORCE("glGetIntegerv: more than %d distinct pnames read back as 0; further reports suppressed",
                        kMaxReportedPnames);
        }
        return false;
    }
    g_reported_pnames[g_reported_count++] = pname;
    return true;
}

void ReportContextFix(GLenum pname, long long value) {
    if (!ReportOnce(pname)) return;
    LOG_W_FORCE("glGetIntegerv(0x%x) read 0 until MobileGLES bound its fallback EGL context, then returned %lld. "
                "The calling thread had no current EGL context.",
                pname, value);
}

void ReportUntracked(GLenum pname, bool rejected) {
    if (!ReportOnce(pname)) return;
    LOG_W_FORCE("glGetIntegerv(0x%x) read back as 0 and the host %s. No fallback is registered for this enum, "
                "so it is left at 0.",
                pname, rejected ? "reported an error" : "reported no error");
}

void ReportSubstitution(GLenum pname, const LimitFallback& entry, GLint value, bool rejected) {
    if (!ReportOnce(pname)) return;
    LOG_W_FORCE("glGetIntegerv(%s / 0x%x) was not answered by the host GLES driver (%s); using %d instead of 0.",
                entry.name, pname, rejected ? "GL error raised" : "value stayed 0", value);
}

// Asks the host for an integer limit, repairing the answer if the host never
// gave one. Always returns something usable, or 0 when the limit is one we
// have no opinion about.
GLint QueryHostInt(GLenum pname) {
    if (GLES.glGetIntegerv) {
        GLint value = 0;
        GLES.glGetIntegerv(pname, &value);
        if (value > 0) return value;
    }

    // Maybe the thread simply had no context: re-read with ours bound.
    {
        ScopedHostContext scoped;
        if (scoped.Bound() && GLES.glGetIntegerv) {
            GLint retry = 0;
            GLES.glGetIntegerv(pname, &retry);
            if (retry > 0) {
                ReportContextFix(pname, retry);
                return retry;
            }
        }
    }

    const bool rejected = ConsumeHostErrors();
    const LimitFallback* entry = FindLimitFallback(pname);
    if (!entry) {
        ReportUntracked(pname, rejected);
        return 0;
    }

    const GLint value = ResolveLimitFallback(*entry);
    ReportSubstitution(pname, *entry, value, rejected);
    return value;
}

// Memoized QueryHostInt. Only a successful answer is stored: a query that ran
// while this thread had no current context must not poison the cache with 0
// for the rest of the process, which is exactly what the old per-pname
// `static GLint cached = -1` blocks did.
constexpr int kMaxCacheEntries = 32;
GLenum g_cache_pnames[kMaxCacheEntries];
GLint g_cache_values[kMaxCacheEntries];
int g_cache_count = 0;
std::mutex g_cache_mutex;

GLint CachedHostInt(GLenum pname) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);

    for (int i = 0; i < g_cache_count; ++i) {
        if (g_cache_pnames[i] == pname) return g_cache_values[i];
    }

    const GLint value = QueryHostInt(pname);
    if (value > 0 && g_cache_count < kMaxCacheEntries) {
        g_cache_pnames[g_cache_count] = pname;
        g_cache_values[g_cache_count] = value;
        ++g_cache_count;
    }
    return value;
}

} // namespace limitguard

void mg_guard_host_limit_i(GLenum pname, GLint* params) {
    if (!params || *params > 0) return;
    *params = limitguard::QueryHostInt(pname);
}

void mg_guard_host_limit_f(GLenum pname, GLfloat* params) {
    if (!params || *params > 0.0f) return;

    // Check the table first. QueryHostInt() unconditionally asks the host for
    // an int, and glGetFloatv also serves queries that are legitimately 0 or
    // negative (ranges, clear values) — asking those as ints would raise
    // GL_INVALID_ENUM for no reason. Only limits we actually track are touched.
    if (!limitguard::FindLimitFallback(pname)) return;

    const GLint value = limitguard::QueryHostInt(pname);
    if (value > 0) *params = static_cast<GLfloat>(value);
}

void mg_guard_host_limit_i64(GLenum pname, GLint64* params) {
    using namespace limitguard;
    if (!params || *params > 0) return;

    {
        ScopedHostContext scoped;
        if (scoped.Bound() && GLES.glGetInteger64v) {
            GLint64 retry = 0;
            GLES.glGetInteger64v(pname, &retry);
            if (retry > 0) {
                *params = retry;
                ReportContextFix(pname, (long long)retry);
                return;
            }
        }
    }

    const bool rejected = ConsumeHostErrors();
    const LimitFallback* entry = FindLimitFallback(pname);
    if (!entry) {
        ReportUntracked(pname, rejected);
        return;
    }

    const GLint value = ResolveLimitFallback(*entry);
    *params = (GLint64)value;
    ReportSubstitution(pname, *entry, value, rejected);
}

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
        // during per-frame state setup). Desktop GL guarantees at least
        // twice the ES 3.2 minimum, so the doubled value is what callers
        // expect to see.
        static const GLint cached = limitguard::CachedHostInt(pname) * 2;
        (*params) = cached;
        break;
    }

    // -------------------------------------------------------------------------
    // Static device limits — queried once and cached.
    //
    // These are properties of the GPU, so the answer never changes. But they
    // were cached unconditionally: a query that ran while this thread had no
    // current EGL context stored 0 and returned it for the rest of the session.
    // CachedHostInt() retries under MobileGLES' fallback context and only
    // memorizes a value the host actually reported.
    // -------------------------------------------------------------------------
    case GL_MAX_VERTEX_ATTRIBS:
    case GL_MAX_DRAW_BUFFERS:
    case GL_MAX_VERTEX_UNIFORM_VECTORS:
    case GL_MAX_FRAGMENT_UNIFORM_VECTORS:
    case GL_MAX_VARYING_VECTORS:
    case GL_MAX_TEXTURE_SIZE:
    case GL_MAX_CUBE_MAP_TEXTURE_SIZE:
    case GL_MAX_RENDERBUFFER_SIZE:
    case GL_MAX_SAMPLES: {
        static const GLint cached = limitguard::CachedHostInt(pname);
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
        // Zero it first so a driver that ignores the query leaves a defined
        // value instead of whatever the caller's buffer happened to hold.
        *params = 0;
        if (GLES.glGetIntegerv) GLES.glGetIntegerv(pname, params);
        mg_guard_host_limit_i(pname, params);
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
// Section: Host String Query Safety Helpers
//
//   A host GLES driver is allowed to return NULL from glGetString(): the ES 3.2
//   reference page states "If an error is generated, glGetString returns 0",
//   and in practice drivers (Adreno, Mali, Mesa, ANGLE) do exactly that when the
//   calling thread has no current EGL context. Passing such a pointer into
//   std::string(const char*) is undefined behaviour and crashes inside strlen()
//   — which is precisely the reported SIGSEGV (si_addr = 0x0, R0 = 0x0) when
//   Minecraft's renderpearl GlDevice queried GL_RENDERER.
//
//   Every host string query therefore goes through these helpers, which:
//     * bind MobileGLES' fallback pbuffer context if this thread has none
//     * check the returned pointer before any strlen()/std::string construction
//     * always hand back a non-NULL string
// =============================================================================

namespace {

// Asks the host driver for a string. The first attempt is made as-is, so the
// common path costs nothing; only if that yields nothing do we bind the
// fallback context and retry — a missing current context on this thread is the
// usual reason drivers hand back NULL here.
const GLubyte* QueryHostString(GLenum name) {
    if (!GLES.glGetString) return nullptr;

    const GLubyte* str = GLES.glGetString(name);
    if (str && *str) return str;

    ScopedHostContext scoped;
    if (scoped.Bound()) {
        const GLubyte* retry = GLES.glGetString(name);
        if (retry && *retry) {
            LOG_W_FORCE("Host glGetString(0x%x) only answered after binding the fallback context: "
                        "the calling thread had no current EGL context",
                        name);
            return retry;
        }
    }

    return (str && *str) ? str : nullptr;
}

// Queries a host GL string, substituting `fallback` when the driver gives
// nothing usable back.
std::string SafeHostString(GLenum name, const char* fallback) {
    const GLubyte* str = QueryHostString(name);
    if (str) return std::string(reinterpret_cast<const char*>(str));

    LOG_W_FORCE("Host glGetString(0x%x) gave no usable string (NULL returned or symbol unavailable), "
                "using fallback \"%s\"",
                name, fallback);
    return std::string(fallback);
}

// Raw-pointer variant for queries we forward straight to the caller. Never
// returns NULL: an empty static string is used instead, so that LWJGL (and
// anything else doing strlen() on the result) cannot dereference NULL.
const GLubyte* SafeHostStringRaw(GLenum name) {
    const GLubyte* str = QueryHostString(name);
    if (str) return str;

    LOG_W_FORCE("Host glGetString(0x%x) gave no usable string (NULL returned or symbol unavailable); "
                "returning an empty string instead",
                name);
    return reinterpret_cast<const GLubyte*>("");
}

} // namespace

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
    // SafeHostString() guards against a NULL/empty host string: constructing a
    // std::string straight from the host pointer crashes in strlen() when the
    // driver returns NULL (no current context on the calling thread).
    std::string gpuName = SafeHostString(GL_RENDERER, "Unknown GPU");

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
    return getBeforeThirdSpace(SafeHostString(GL_VERSION, "OpenGL ES 3.2"));
}

void set_es_version() {
    std::string ESVersionStr = getBeforeThirdSpace(SafeHostString(GL_VERSION, "OpenGL ES 3.2"));
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
            return SafeHostStringRaw(name);

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
            return SafeHostStringRaw(name - GL_BACKEND_GETTER_MG);
        else
            return SafeHostStringRaw(name);

    // -------------------------------------------------------------------------
    // All other string queries → forward to native GLES
    // -------------------------------------------------------------------------
    default:
        return SafeHostStringRaw(name);
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