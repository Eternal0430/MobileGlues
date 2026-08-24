// MobileGlues - gl/multidraw.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header
//
// Architecture: "ES 3.2 native → native, ES 3.2 not native → CPU simulation"
//
// glMultiDrawElements is NOT part of the ES 3.2 core specification.
// It is a desktop OpenGL function that must be simulated on ES 3.2 backends.
// This file implements five CPU simulation strategies, selectable at runtime
// via global_settings.multidraw_mode.
//
// Available strategies (multidraw_mode):
//   DrawElements        — Loop over draws, call glDrawElements per draw.
//                         Works on ALL ES 3.2 devices. Fallback strategy.
//   PreferIndirect      — Pack draw commands into an indirect buffer, then
//                         call glDrawElementsIndirect per draw.
//                         Requires ES 3.1 (glDrawElementsIndirect).
//   PreferMultidrawIndirect — Pack draw commands into an indirect buffer,
//                             then call glMultiDrawElementsIndirectEXT once.
//                             Requires EXT_multi_draw_indirect.
//   PreferBaseVertex    — Loop over draws, call glDrawElementsBaseVertex.
//                         Requires ES 3.2 (glDrawElementsBaseVertex).
//   Compute             — Use a compute shader to merge all index buffers into
//                         one contiguous output, then draw once.
//                         Requires ES 3.1 compute shaders. Falls back to
//                         DrawElements for strip-like primitive modes.

#include "multidraw.h"
#include "../config/settings.h"
#include "buffer.h"
#include <cstdint>
#include <limits>
#include <vector>

#define DEBUG 0

// =============================================================================
// Section: Forward Declarations
// =============================================================================

static bool is_strip_like_mode(GLenum mode);

// =============================================================================
// Section: Strategy Dispatch — glMultiDrawElements
//   Selects a CPU simulation strategy based on global_settings.multidraw_mode.
//   glMultiDrawElements is NOT in ES 3.2 core; all paths are CPU simulation.
// =============================================================================

typedef void (*glMultiDrawElements_t)(GLenum, const GLsizei*, GLenum, const void* const*, GLsizei);

void glMultiDrawElements(GLenum mode, const GLsizei* count, GLenum type, const void* const* indices,
                         GLsizei primcount) {
    static glMultiDrawElements_t func_ptr = nullptr;

    if (func_ptr == nullptr) [[unlikely]] {
        switch (global_settings.multidraw_mode) {
        case multidraw_mode_t::PreferIndirect:
            func_ptr = mg_glMultiDrawElements_indirect;
            break;
        case multidraw_mode_t::PreferBaseVertex:
            func_ptr = mg_glMultiDrawElements_basevertex;
            break;
        case multidraw_mode_t::PreferMultidrawIndirect:
            func_ptr = mg_glMultiDrawElements_multiindirect;
            break;
        case multidraw_mode_t::DrawElements:
            func_ptr = mg_glMultiDrawElements_drawelements;
            break;
        case multidraw_mode_t::Compute:
            func_ptr = mg_glMultiDrawElements_compute;
            break;
        default:
            func_ptr = mg_glMultiDrawElements_drawelements;
            break;
        }
    }
    func_ptr(mode, count, type, indices, primcount);
}

// =============================================================================
// Section: Strategy Dispatch — glMultiDrawElementsBaseVertex
//   Same as above but with basevertex parameter.
//   glMultiDrawElementsBaseVertex is NOT in ES 3.2 core; all paths are CPU
//   simulation.
// =============================================================================

typedef void (*glMultiDrawElementsBaseVertex_t)(GLenum, const GLsizei*, GLenum, const void* const*, GLsizei, const GLint*);

void glMultiDrawElementsBaseVertex(GLenum mode, const GLsizei* counts, GLenum type, const void* const* indices,
                                   GLsizei primcount, const GLint* basevertex) {
    static glMultiDrawElementsBaseVertex_t func_ptr = nullptr;

    if (func_ptr == nullptr) [[unlikely]] {
        switch (global_settings.multidraw_mode) {
        case multidraw_mode_t::PreferIndirect:
            func_ptr = mg_glMultiDrawElementsBaseVertex_indirect;
            break;
        case multidraw_mode_t::PreferBaseVertex:
            func_ptr = mg_glMultiDrawElementsBaseVertex_basevertex;
            break;
        case multidraw_mode_t::PreferMultidrawIndirect:
            func_ptr = mg_glMultiDrawElementsBaseVertex_multiindirect;
            break;
        case multidraw_mode_t::DrawElements:
            func_ptr = mg_glMultiDrawElementsBaseVertex_drawelements;
            break;
        case multidraw_mode_t::Compute:
            func_ptr = mg_glMultiDrawElementsBaseVertex_compute;
            break;
        default:
            func_ptr = mg_glMultiDrawElementsBaseVertex_drawelements;
            break;
        }
    }

    func_ptr(mode, counts, type, indices, primcount, basevertex);
}

// =============================================================================
// Section: Indirect Buffer Management
//   Manages a reusable GL_DRAW_INDIRECT_BUFFER for indirect / multi-indirect
//   strategies. Grows by 2x when primcount exceeds the current capacity.
// =============================================================================

static bool g_indirect_cmds_inited = false;
static GLsizei g_cmdbufsize = 0;
static GLuint g_indirectbuffer = 0;
static GLuint prevIndirectBuffer = 0;
static constexpr GLsizei kDefaultIndirectCmdBufSize = 256;

void prepare_indirect_buffer(const GLsizei* counts, GLenum type, const void* const* indices, GLsizei primcount,
                             const GLint* basevertex) {
    // Use CPU-side cached binding instead of glGetIntegerv GPU round-trip
    prevIndirectBuffer = find_bound_buffer(GL_DRAW_INDIRECT_BUFFER_BINDING);

    if (!g_indirect_cmds_inited) {
        GLES.glGenBuffers(1, &g_indirectbuffer);
        GLES.glBindBuffer(GL_DRAW_INDIRECT_BUFFER, g_indirectbuffer);
        g_cmdbufsize = kDefaultIndirectCmdBufSize;
        GLES.glBufferData(GL_DRAW_INDIRECT_BUFFER, g_cmdbufsize * sizeof(draw_elements_indirect_command_t), NULL,
                          GL_DYNAMIC_DRAW);
        g_indirect_cmds_inited = true;
    }

    GLES.glBindBuffer(GL_DRAW_INDIRECT_BUFFER, g_indirectbuffer);

    if (g_cmdbufsize < primcount) {
        size_t sz = g_cmdbufsize;

        LOG_D("Before resize: %d", sz)

        // 2-exponential to reduce reallocation
        while (sz < primcount)
            sz *= 2;

        GLES.glBufferData(GL_DRAW_INDIRECT_BUFFER, sz * sizeof(draw_elements_indirect_command_t), NULL,
                          GL_DYNAMIC_DRAW);
        g_cmdbufsize = sz;
    }

    LOG_D("After resize: %d", g_cmdbufsize)

    auto* pcmds = (draw_elements_indirect_command_t*)GLES.glMapBufferRange(
        GL_DRAW_INDIRECT_BUFFER, 0, primcount * sizeof(draw_elements_indirect_command_t),
        GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

    GLsizei elementSize;
    switch (type) {
    case GL_UNSIGNED_BYTE:
        elementSize = 1;
        break;
    case GL_UNSIGNED_SHORT:
        elementSize = 2;
        break;
    case GL_UNSIGNED_INT:
        elementSize = 4;
        break;
    default:
        elementSize = 4;
    }

    for (GLsizei i = 0; i < primcount; ++i) {
        auto byteOffset = reinterpret_cast<uintptr_t>(indices[i]);
        pcmds[i].firstIndex = static_cast<GLuint>(byteOffset / elementSize);
        pcmds[i].count = counts[i];
        pcmds[i].instanceCount = 1;
        pcmds[i].baseVertex = basevertex ? basevertex[i] : 0;
        pcmds[i].reservedMustBeZero = 0;
    }

    GLES.glUnmapBuffer(GL_DRAW_INDIRECT_BUFFER);
}

// =============================================================================
// Section: Strategy — DrawElements
//   CPU simulation via per-draw glDrawElements calls.
//   Works on ALL ES 3.2 devices. Slowest but most compatible.
//   For basevertex variant: creates a temp index buffer with basevertex offset
//   applied to each index.
// =============================================================================

void mg_glMultiDrawElements_drawelements(GLenum mode, const GLsizei* count, GLenum type, const void* const* indices,
                                         GLsizei primcount) {
    // PREPARE_FOR_DRAW() is a no-op; removed to reduce call overhead on hot path.

    for (GLsizei i = 0; i < primcount; ++i) {
        const GLsizei c = count[i];
        if (c > 0) GLES.glDrawElements(mode, c, type, indices[i]);
    }
}

void mg_glMultiDrawElementsBaseVertex_drawelements(GLenum mode, const GLsizei* counts, GLenum type,
                                                   const void* const* indices, GLsizei primcount,
                                                   const GLint* basevertex) {
    // Hoist index-size computation out of the per-draw loop: `type` is loop-invariant,
    // so computing it once before the loop avoids primcount redundant switch dispatches.
    size_t indexSize;
    switch (type) {
    case GL_UNSIGNED_INT:
        indexSize = sizeof(GLuint);
        break;
    case GL_UNSIGNED_SHORT:
        indexSize = sizeof(GLushort);
        break;
    case GL_UNSIGNED_BYTE:
        indexSize = sizeof(GLubyte);
        break;
    default:
        return;
    }

    // Use CPU-side cached binding instead of glGetIntegerv GPU round-trip
    GLint prevElementBuffer = (GLint)find_bound_buffer(GL_ELEMENT_ARRAY_BUFFER_BINDING);

    // Reusable thread_local temp buffer and memory to avoid per-draw allocation
    static thread_local GLuint tempBuffer = 0;
    static thread_local size_t tempBufferSize = 0;
    static thread_local std::vector<uint8_t> tempMem;

    if (tempBuffer == 0) {
        GLES.glGenBuffers(1, &tempBuffer);
    }

    for (GLsizei i = 0; i < primcount; ++i) {
        if (counts[i] <= 0) continue;

        GLsizei currentCount = counts[i];
        const GLvoid* currentIndices = indices[i];
        GLint currentBaseVertex = basevertex[i];

        size_t neededSize = currentCount * indexSize;
        tempMem.resize(neededSize);
        void* tempIndices = tempMem.data();

        void* srcData = nullptr;
        if (prevElementBuffer != 0) {
            GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, prevElementBuffer);
            srcData = GLES.glMapBufferRange(GL_ELEMENT_ARRAY_BUFFER, (GLintptr)currentIndices, neededSize,
                                            GL_MAP_READ_BIT);
            if (!srcData) continue;
        } else {
            srcData = (void*)currentIndices;
        }

        switch (type) {
        case GL_UNSIGNED_INT:
            for (int j = 0; j < currentCount; ++j) {
                ((GLuint*)tempIndices)[j] = ((GLuint*)srcData)[j] + currentBaseVertex;
            }
            break;
        case GL_UNSIGNED_SHORT:
            for (int j = 0; j < currentCount; ++j) {
                ((GLushort*)tempIndices)[j] = ((GLushort*)srcData)[j] + currentBaseVertex;
            }
            break;
        case GL_UNSIGNED_BYTE:
            for (int j = 0; j < currentCount; ++j) {
                ((GLubyte*)tempIndices)[j] = ((GLubyte*)srcData)[j] + currentBaseVertex;
            }
            break;
        }

        if (prevElementBuffer != 0) {
            GLES.glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
        }

        GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tempBuffer);
        if (neededSize > tempBufferSize) {
            GLES.glBufferData(GL_ELEMENT_ARRAY_BUFFER, neededSize, tempIndices, GL_STREAM_DRAW);
            tempBufferSize = neededSize;
        } else {
            GLES.glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, neededSize, tempIndices);
        }
        GLES.glDrawElements(mode, currentCount, type, 0);
    }

    GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, prevElementBuffer);
}

// =============================================================================
// Section: Strategy — Indirect
//   CPU simulation: pack draw commands into an indirect buffer, then call
//   glDrawElementsIndirect once per draw.
//   Requires ES 3.1 (glDrawElementsIndirect).
// =============================================================================

void mg_glMultiDrawElements_indirect(GLenum mode, const GLsizei* count, GLenum type, const void* const* indices,
                                     GLsizei primcount) {

    prepare_indirect_buffer(count, type, indices, primcount, 0);

    for (GLsizei i = 0; i < primcount; ++i) {
        const GLvoid* offset = reinterpret_cast<GLvoid*>(i * sizeof(draw_elements_indirect_command_t));
        GLES.glDrawElementsIndirect(mode, type, offset);
    }

    GLES.glBindBuffer(GL_DRAW_INDIRECT_BUFFER, prevIndirectBuffer);
}

void mg_glMultiDrawElementsBaseVertex_indirect(GLenum mode, const GLsizei* counts, GLenum type, const void* const* indices,
                                               GLsizei primcount, const GLint* basevertex) {

    prepare_indirect_buffer(counts, type, indices, primcount, basevertex);

    for (GLsizei i = 0; i < primcount; ++i) {
        const GLvoid* offset = reinterpret_cast<GLvoid*>(i * sizeof(draw_elements_indirect_command_t));
        GLES.glDrawElementsIndirect(mode, type, offset);
    }

    GLES.glBindBuffer(GL_DRAW_INDIRECT_BUFFER, prevIndirectBuffer);
}

// =============================================================================
// Section: Strategy — Multi-Indirect
//   CPU simulation: pack draw commands into an indirect buffer, then call
//   glMultiDrawElementsIndirectEXT once.
//   Requires EXT_multi_draw_indirect.
// =============================================================================

void mg_glMultiDrawElements_multiindirect(GLenum mode, const GLsizei* count, GLenum type, const void* const* indices,
                                          GLsizei primcount) {

    prepare_indirect_buffer(count, type, indices, primcount, 0);

    GLES.glMultiDrawElementsIndirectEXT(mode, type, 0, primcount, 0);

    GLES.glBindBuffer(GL_DRAW_INDIRECT_BUFFER, prevIndirectBuffer);
}

void mg_glMultiDrawElementsBaseVertex_multiindirect(GLenum mode, const GLsizei* counts, GLenum type,
                                                    const void* const* indices, GLsizei primcount,
                                                    const GLint* basevertex) {

    prepare_indirect_buffer(counts, type, indices, primcount, basevertex);

    GLES.glMultiDrawElementsIndirectEXT(mode, type, 0, primcount, 0);

    GLES.glBindBuffer(GL_DRAW_INDIRECT_BUFFER, prevIndirectBuffer);
}

// =============================================================================
// Section: Strategy — BaseVertex
//   CPU simulation: loop over draws, call glDrawElementsBaseVertex per draw.
//   Requires ES 3.2 (glDrawElementsBaseVertex).
// =============================================================================

void mg_glMultiDrawElements_basevertex(GLenum mode, const GLsizei* count, GLenum type, const void* const* indices,
                                       GLsizei primcount) {

    for (GLsizei i = 0; i < primcount; ++i) {
        const GLsizei c = count[i];
        if (c > 0) {
            GLES.glDrawElements(mode, c, type, indices[i]);
        }
    }
}

void mg_glMultiDrawElementsBaseVertex_basevertex(GLenum mode, const GLsizei* counts, GLenum type, const void* const* indices,
                                                 GLsizei primcount, const GLint* basevertex) {

    for (GLsizei i = 0; i < primcount; ++i) {
        const GLsizei count = counts[i];
        if (count > 0) {
            GLES.glDrawElementsBaseVertex(mode, count, type, indices[i], basevertex[i]);
        }
    }
}

// =============================================================================
// Section: Strategy — Compute (without basevertex)
//   Simple fallback: loop over draws, call glDrawElements per draw.
// =============================================================================

void mg_glMultiDrawElements_compute(GLenum mode, const GLsizei* count, GLenum type, const void* const* indices,
                                    GLsizei primcount) {

    for (GLsizei i = 0; i < primcount; ++i) {
        const GLsizei c = count[i];
        if (c > 0) {
            GLES.glDrawElements(mode, c, type, indices[i]);
        }
    }
}

// =============================================================================
// Section: Primitive Mode Helper
// =============================================================================

static bool is_strip_like_mode(GLenum mode) {
    switch (mode) {
    case GL_LINE_STRIP:
    case GL_LINE_LOOP:
    case GL_TRIANGLE_STRIP:
    case GL_TRIANGLE_FAN:
    case GL_LINE_STRIP_ADJACENCY:
    case GL_TRIANGLE_STRIP_ADJACENCY:
        return true;
    default:
        return false;
    }
}

// =============================================================================
// Section: Compute Shader Pipeline
//   CPU simulation via compute shader: merges all index buffers into one
//   contiguous output buffer, then draws once with glDrawElements.
//   Requires ES 3.1 compute shaders.
//   Falls back to DrawElements strategy for strip-like primitive modes
//   (GL_LINE_STRIP, GL_TRIANGLE_STRIP, etc.) because merging strips would
//   break restart indices.
// =============================================================================

const std::string multidraw_comp_shader =
    R"(#version 310 es

layout(local_size_x = 64) in;

layout(location = 0) uniform uint uElementSize;

layout(std430, binding = 0) readonly buffer Input { uint in_indices[]; };
layout(std430, binding = 1) readonly buffer FirstIndex { uint firstIndex[]; };
layout(std430, binding = 2) readonly buffer BaseVertex { int baseVertex[]; };
layout(std430, binding = 3) readonly buffer Prefix { uint prefixSums[]; };
layout(std430, binding = 4) writeonly buffer Output { uint out_indices[]; };

uint read_index(uint elementIndex) {
    if (uElementSize == 4u) {
        return in_indices[elementIndex];
    }
    if (uElementSize == 2u) {
        uint word = in_indices[elementIndex >> 1u];
        uint shift = (elementIndex & 1u) * 16u;
        return (word >> shift) & 0xFFFFu;
    }
    uint word = in_indices[elementIndex >> 2u];
    uint shift = (elementIndex & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

void main() {
    uint outIdx = gl_GlobalInvocationID.x;
    uint drawCount = uint(prefixSums.length());
    if (drawCount == 0u) {
        return;
    }
    uint total = prefixSums[drawCount - 1u];
    if (outIdx >= total) {
        return;
    }

    int low = 0;
    int high = int(drawCount) - 1;
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (prefixSums[mid] > outIdx) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }

    uint localIdx = outIdx - ((low == 0) ? 0u : (prefixSums[low - 1]));
    uint inIndex = localIdx + firstIndex[low];

    int idx = int(read_index(inIndex));
    out_indices[outIdx] = uint(idx + baseVertex[low]);
}

)";

// =============================================================================
// Section: Compute Shader Pipeline — Persistent State
//   All compute pipeline resources are lazily initialized once and reused.
// =============================================================================

static bool g_compute_inited = false;
std::vector<GLuint> g_prefix_sum(1);
GLuint g_prefixsumbuffer = 0;
GLuint g_firstidx_ssbo = 0;
GLuint g_basevtx_ssbo = 0;
GLuint g_outputibo = 0;
GLuint g_compute_program = 0;
GLint g_element_size_loc = -1;
char g_compile_info[1024];

// =============================================================================
// Section: Compute Shader — Compilation
// =============================================================================

static GLuint compile_compute_program(const std::string& src) {
    INIT_CHECK_GL_ERROR
    auto program = GLES.glCreateProgram();
    CHECK_GL_ERROR_NO_INIT
    GLuint shader = GLES.glCreateShader(GL_COMPUTE_SHADER);
    CHECK_GL_ERROR_NO_INIT
    const char* s[] = {src.c_str()};
    const GLint length[] = {static_cast<GLint>(src.length())};
    GLES.glShaderSource(shader, 1, s, length);
    CHECK_GL_ERROR_NO_INIT
    GLES.glCompileShader(shader);
    CHECK_GL_ERROR_NO_INIT
    int success = 0;
    GLES.glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    CHECK_GL_ERROR_NO_INIT
    if (!success) {
        GLES.glGetShaderInfoLog(shader, 1024, NULL, g_compile_info);
        CHECK_GL_ERROR_NO_INIT
        LOG_E("%s: %s shader compile error: %s\nsrc:\n%s", __func__, "compute", g_compile_info, src.c_str());
#if DEBUG || GLOBAL_DEBUG
        abort();
#endif
        GLES.glDeleteShader(shader);
        GLES.glDeleteProgram(program);
        return 0;
    }

    GLES.glAttachShader(program, shader);
    CHECK_GL_ERROR_NO_INIT
    GLES.glLinkProgram(program);
    CHECK_GL_ERROR_NO_INIT

    GLES.glGetProgramiv(program, GL_LINK_STATUS, &success);
    CHECK_GL_ERROR_NO_INIT
    if (!success) {
        GLES.glGetProgramInfoLog(program, 1024, NULL, g_compile_info);
        CHECK_GL_ERROR_NO_INIT
        LOG_E("program link error: %s", g_compile_info);
#if DEBUG || GLOBAL_DEBUG
        abort();
#endif
        GLES.glDeleteShader(shader);
        GLES.glDeleteProgram(program);
        return 0;
    }

    GLES.glDeleteShader(shader);
    CHECK_GL_ERROR_NO_INIT
    g_element_size_loc = GLES.glGetUniformLocation(program, "uElementSize");
    CHECK_GL_ERROR_NO_INIT

    return program;
}

// =============================================================================
// Section: Strategy — Compute (with basevertex)
//   CPU simulation via compute shader pipeline:
//     1. Validate inputs (fall back to DrawElements on error).
//     2. Build prefix sums, first-index, and base-vertex arrays.
//     3. Dispatch compute shader to merge all index buffers into one output.
//     4. Draw once with glDrawElements using the merged output buffer.
//   Falls back to DrawElements for strip-like modes.
// =============================================================================

GLAPI GLAPIENTRY void mg_glMultiDrawElementsBaseVertex_compute(GLenum mode, const GLsizei* counts, GLenum type,
                                                               const void* const* indices, GLsizei primcount,
                                                               const GLint* basevertex) {
    LOG()

    INIT_CHECK_GL_ERROR

    if (primcount <= 0) return;
    if (!counts || !indices) {
        LOG_E("mg_glMultiDrawElementsBaseVertex_compute: counts or indices is null")
        return;
    }
    if (is_strip_like_mode(mode)) {
        LOG_D("mg_glMultiDrawElementsBaseVertex_compute: strip/loop mode, fallback")
        mg_glMultiDrawElementsBaseVertex_drawelements(mode, counts, type, indices, primcount, basevertex);
        return;
    }

    GLuint elementSize = 0;
    switch (type) {
    case GL_UNSIGNED_BYTE:
        elementSize = 1;
        break;
    case GL_UNSIGNED_SHORT:
        elementSize = 2;
        break;
    case GL_UNSIGNED_INT:
        elementSize = 4;
        break;
    default:
        LOG_E("mg_glMultiDrawElementsBaseVertex_compute: unsupported index type %s", glEnumToString(type))
        mg_glMultiDrawElementsBaseVertex_drawelements(mode, counts, type, indices, primcount, basevertex);
        return;
    }

    // -------------------------------------------------------------------------
    // Lazy initialization of compute pipeline resources
    // -------------------------------------------------------------------------
    if (!g_compute_inited) {
        LOG_D("Initializing multidraw compute pipeline...")
        GLES.glGenBuffers(1, &g_prefixsumbuffer);
        GLES.glGenBuffers(1, &g_firstidx_ssbo);
        GLES.glGenBuffers(1, &g_basevtx_ssbo);
        GLES.glGenBuffers(1, &g_outputibo);

        g_compute_program = compile_compute_program(multidraw_comp_shader);
        if (g_compute_program == 0) {
            LOG_E("mg_glMultiDrawElementsBaseVertex_compute: compute program init failed, fallback")
            GLES.glDeleteBuffers(1, &g_prefixsumbuffer);
            GLES.glDeleteBuffers(1, &g_firstidx_ssbo);
            GLES.glDeleteBuffers(1, &g_basevtx_ssbo);
            GLES.glDeleteBuffers(1, &g_outputibo);
            g_prefixsumbuffer = 0;
            g_firstidx_ssbo = 0;
            g_basevtx_ssbo = 0;
            g_outputibo = 0;
            mg_glMultiDrawElementsBaseVertex_drawelements(mode, counts, type, indices, primcount, basevertex);
            return;
        }

        g_compute_inited = true;
    }

    // -------------------------------------------------------------------------
    // Validate element array buffer
    // -------------------------------------------------------------------------
    // Use CPU-side cached binding instead of glGetIntegerv GPU round-trip
    GLint ibo = (GLint)find_bound_buffer(GL_ELEMENT_ARRAY_BUFFER_BINDING);
    CHECK_GL_ERROR_NO_INIT
    if (ibo == 0) {
        LOG_D("mg_glMultiDrawElementsBaseVertex_compute: no element array buffer bound, fallback")
        mg_glMultiDrawElementsBaseVertex_drawelements(mode, counts, type, indices, primcount, basevertex);
        return;
    }
    GLint ibo_size = 0;
    GLES.glGetBufferParameteriv(GL_ELEMENT_ARRAY_BUFFER, GL_BUFFER_SIZE, &ibo_size);
    CHECK_GL_ERROR_NO_INIT
    if (ibo_size <= 0) {
        LOG_E("mg_glMultiDrawElementsBaseVertex_compute: invalid index buffer size, fallback")
        mg_glMultiDrawElementsBaseVertex_drawelements(mode, counts, type, indices, primcount, basevertex);
        return;
    }
    if (elementSize < 4 && (ibo_size % 4) != 0) {
        LOG_E("mg_glMultiDrawElementsBaseVertex_compute: index buffer size not 4-byte aligned, fallback")
        mg_glMultiDrawElementsBaseVertex_drawelements(mode, counts, type, indices, primcount, basevertex);
        return;
    }

    // -------------------------------------------------------------------------
    // Build prefix sums, first-index, and base-vertex arrays
    // -------------------------------------------------------------------------
    g_prefix_sum.resize(primcount);
    // Reuse thread_local capacity across calls: assign() reuses the existing
    // allocation and re-zeroes, matching the original (primcount, 0) semantics
    // without a per-call heap allocation (same pattern as tempMem above).
    static thread_local std::vector<GLuint> first_index;
    static thread_local std::vector<GLint> base_vtx;
    first_index.assign(primcount, 0);
    base_vtx.assign(primcount, 0);

    uint64_t running = 0;
    for (GLsizei i = 0; i < primcount; ++i) {
        GLsizei c = counts[i];
        if (c < 0) {
            LOG_E("mg_glMultiDrawElementsBaseVertex_compute: negative count at %d", i)
            c = 0;
        }
        running += static_cast<uint64_t>(c);
        if (running > static_cast<uint64_t>(std::numeric_limits<GLuint>::max())) {
            LOG_E("mg_glMultiDrawElementsBaseVertex_compute: total index count overflow, fallback")
            mg_glMultiDrawElementsBaseVertex_drawelements(mode, counts, type, indices, primcount, basevertex);
            return;
        }
        g_prefix_sum[i] = static_cast<GLuint>(running);

        if (c > 0) {
            if (!indices[i]) {
                LOG_E("mg_glMultiDrawElementsBaseVertex_compute: indices[%d] is null", i)
                mg_glMultiDrawElementsBaseVertex_drawelements(mode, counts, type, indices, primcount, basevertex);
                return;
            }
            uintptr_t byteOffset = reinterpret_cast<uintptr_t>(indices[i]);
            uint64_t byteOffset64 = static_cast<uint64_t>(byteOffset);
            if ((byteOffset % elementSize) != 0) {
                LOG_E("mg_glMultiDrawElementsBaseVertex_compute: misaligned index offset at %d", i)
            }
            if (byteOffset64 > static_cast<uint64_t>(ibo_size)) {
                LOG_E("mg_glMultiDrawElementsBaseVertex_compute: index offset out of range at %d", i)
                mg_glMultiDrawElementsBaseVertex_drawelements(mode, counts, type, indices, primcount, basevertex);
                return;
            }
            uint64_t byteEnd = byteOffset64 + static_cast<uint64_t>(c) * elementSize;
            if (byteEnd > static_cast<uint64_t>(ibo_size)) {
                LOG_E("mg_glMultiDrawElementsBaseVertex_compute: index range out of bounds at %d", i)
                mg_glMultiDrawElementsBaseVertex_drawelements(mode, counts, type, indices, primcount, basevertex);
                return;
            }
            uint64_t elementOffset = byteOffset64 / elementSize;
            if (elementOffset > static_cast<uint64_t>(std::numeric_limits<GLuint>::max())) {
                LOG_E("mg_glMultiDrawElementsBaseVertex_compute: index offset overflow at %d", i)
                mg_glMultiDrawElementsBaseVertex_drawelements(mode, counts, type, indices, primcount, basevertex);
                return;
            }
            first_index[i] = static_cast<GLuint>(elementOffset);
        }

        if (basevertex) {
            base_vtx[i] = basevertex[i];
        }
    }

    auto total_indices = g_prefix_sum[primcount - 1];
    if (total_indices == 0) {
        return;
    }

    // -------------------------------------------------------------------------
    // Save GL state before compute dispatch
    // -------------------------------------------------------------------------
    // Use CPU-side cached bindings instead of glGetIntegerv GPU round-trips
    GLint prev_ssbo_binding = (GLint)find_bound_buffer(GL_SHADER_STORAGE_BUFFER_BINDING);
    CHECK_GL_ERROR_NO_INIT

    // -------------------------------------------------------------------------
    // Upload SSBO data
    // -------------------------------------------------------------------------
    GLES.glBindBuffer(GL_SHADER_STORAGE_BUFFER, g_firstidx_ssbo);
    CHECK_GL_ERROR_NO_INIT
    GLES.glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint) * primcount, first_index.data(), GL_DYNAMIC_DRAW);
    CHECK_GL_ERROR_NO_INIT

    GLES.glBindBuffer(GL_SHADER_STORAGE_BUFFER, g_basevtx_ssbo);
    CHECK_GL_ERROR_NO_INIT
    GLES.glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLint) * primcount, base_vtx.data(), GL_DYNAMIC_DRAW);
    CHECK_GL_ERROR_NO_INIT

    GLES.glBindBuffer(GL_SHADER_STORAGE_BUFFER, g_prefixsumbuffer);
    CHECK_GL_ERROR_NO_INIT
    GLES.glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint) * primcount, g_prefix_sum.data(), GL_DYNAMIC_DRAW);
    CHECK_GL_ERROR_NO_INIT

    GLES.glBindBuffer(GL_SHADER_STORAGE_BUFFER, g_outputibo);
    CHECK_GL_ERROR_NO_INIT
    GLES.glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint) * total_indices, nullptr, GL_DYNAMIC_DRAW);
    CHECK_GL_ERROR_NO_INIT

    GLES.glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    CHECK_GL_ERROR_NO_INIT

    // -------------------------------------------------------------------------
    // Bind SSBOs at base indices
    // -------------------------------------------------------------------------
    GLint prev_ssbo_base[5] = {};
    for (int i = 0; i < 5; ++i) {
        prev_ssbo_base[i] = (GLint)find_bound_ssbo_indexed(i);
    }

    GLES.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ibo);
    CHECK_GL_ERROR_NO_INIT
    GLES.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, g_firstidx_ssbo);
    CHECK_GL_ERROR_NO_INIT
    GLES.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, g_basevtx_ssbo);
    CHECK_GL_ERROR_NO_INIT
    GLES.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, g_prefixsumbuffer);
    CHECK_GL_ERROR_NO_INIT
    GLES.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, g_outputibo);
    CHECK_GL_ERROR_NO_INIT

    // Use CPU-side cached state instead of glGetIntegerv GPU round-trips
    GLint prev_program = GLState.currentProgram;
    CHECK_GL_ERROR_NO_INIT
    GLint prev_vb = (GLint)find_bound_buffer(GL_ARRAY_BUFFER_BINDING);
    CHECK_GL_ERROR_NO_INIT

    // -------------------------------------------------------------------------
    // Dispatch compute shader
    // -------------------------------------------------------------------------
    LOG_D("Using compute program = %d", g_compute_program)
    GLES.glUseProgram(g_compute_program);
    CHECK_GL_ERROR_NO_INIT
    if (g_element_size_loc >= 0) {
        GLES.glUniform1ui(g_element_size_loc, elementSize);
        CHECK_GL_ERROR_NO_INIT
    }
    LOG_D("Dispatch compute")
    GLES.glDispatchCompute((total_indices + 63) / 64, 1, 1);
    CHECK_GL_ERROR_NO_INIT

    // -------------------------------------------------------------------------
    // Memory barrier and draw
    // -------------------------------------------------------------------------
    LOG_D("memory barrier")
    GLES.glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);
    CHECK_GL_ERROR_NO_INIT

    LOG_D("draw")
    GLES.glUseProgram(prev_program);
    CHECK_GL_ERROR_NO_INIT
    GLES.glBindBuffer(GL_ARRAY_BUFFER, prev_vb);
    CHECK_GL_ERROR_NO_INIT
    GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_outputibo);
    CHECK_GL_ERROR_NO_INIT
    GLES.glDrawElements(mode, total_indices, GL_UNSIGNED_INT, 0);

    // -------------------------------------------------------------------------
    // Restore GL state
    // -------------------------------------------------------------------------
    for (int i = 0; i < 5; ++i) {
        GLES.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, prev_ssbo_base[i]);
        CHECK_GL_ERROR_NO_INIT
    }
    GLES.glBindBuffer(GL_SHADER_STORAGE_BUFFER, prev_ssbo_binding);
    CHECK_GL_ERROR_NO_INIT
    GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    CHECK_GL_ERROR_NO_INIT
}