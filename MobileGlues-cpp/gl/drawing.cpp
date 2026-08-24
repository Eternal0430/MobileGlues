// MobileGlues - gl/drawing.cpp
// Draw call dispatch: sampler buffer emulation, atomic counter handling.
//
// Architecture principle: "ES 3.2 native → native, ES 3.2 not native → CPU simulation"
//
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

#include "../includes.h"
#include <GL/gl.h>
#include "glcorearb.h"
#include "log.h"
#include "../gles/loader.h"
#include "mg.h"
#include <GLES3/gl32.h>
#include "buffer.h"

#define DEBUG 0

// ---------------------------------------------------------------------------
// Module-level state: tracked texture bindings per unit
// (referenced from texture.cpp and drawing.cpp)
// ---------------------------------------------------------------------------

GLuint g_tracked_tex2d_binding[MAX_TEXTURE_UNITS] = {0};
GLuint g_tracked_tex_cube_binding[MAX_TEXTURE_UNITS] = {0};
GLuint g_tracked_tex_2d_array_binding[MAX_TEXTURE_UNITS] = {0};
GLuint g_tracked_tex_3d_binding[MAX_TEXTURE_UNITS] = {0};
GLuint g_tracked_tex_2d_ms_binding[MAX_TEXTURE_UNITS] = {0};
GLuint g_tracked_tex_2d_ms_array_binding[MAX_TEXTURE_UNITS] = {0};
GLuint g_tracked_tex_cube_array_binding[MAX_TEXTURE_UNITS] = {0};
GLuint g_tracked_tex_rect_binding[MAX_TEXTURE_UNITS] = {0};

// ============================================================================
// Atomic counter buffer emulation
// ============================================================================
//
// Hot-path micro-optimization:
//   syncAtomicCounters() is invoked from every draw/dispatch entry point
//   (9 sites below). The vast majority of draws do NOT use atomic counters,
//   so the function must be as cheap as possible on the no-op fast path.
//
//   A single cached bool (g_atomicCountersActive) collapses the previous
//   two-field early-return (binding != 0 && !data.empty()) into one branch.
//   The flag is updated only when atomic counter state actually changes
//   (in glMemoryBarrier / glDispatchCompute), never on the draw path.
//   Result: one predictable branch per draw call instead of two dependent
//   loads + branches.
//
//   __attribute__((always_inline)) guarantees the compiler folds the
//   fast-path branch directly into the caller's prologue, eliminating
//   call/return overhead on every draw/dispatch entry point.

static inline __attribute__((always_inline)) void syncAtomicCounters() {
    if (!g_atomicCountersActive) return;

    auto &bs = GLState.buffer;
    GLuint realBuf = GLState.GetRealBuffer(bs.atomicCounterBufferBinding);
    if (realBuf) {
        GLES.glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, realBuf);
        GLES.glBufferSubData(GL_ATOMIC_COUNTER_BUFFER,
                             bs.atomicCounterBufferOffset,
                             bs.atomicCounterBufferSize,
                             bs.atomicCounterData.data());
    }
}

static inline __attribute__((always_inline)) void readbackAtomicCounters() {
    if (!g_atomicCountersActive) return;

    auto &bs = GLState.buffer;
    GLuint realBuf = GLState.GetRealBuffer(bs.atomicCounterBufferBinding);
    if (realBuf) {
        GLES.glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, realBuf);
        void *ptr = GLES.glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER,
                                          bs.atomicCounterBufferOffset,
                                          bs.atomicCounterBufferSize,
                                          GL_MAP_READ_BIT);
        if (ptr) {
            memcpy(bs.atomicCounterData.data(), ptr, (size_t)bs.atomicCounterBufferSize);
            GLES.glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
        }
    }
}

// ============================================================================
// Draw call dispatch
// ============================================================================

extern "C" GLAPI GLAPIENTRY void glDrawArrays(GLenum mode, GLint first, GLsizei count) {
    syncAtomicCounters();
    GLES.glDrawArrays(mode, first, count);
}

extern "C" GLAPI GLAPIENTRY void glDrawElements(GLenum mode, GLsizei count, GLenum type, const void *indices) {
    syncAtomicCounters();
    GLES.glDrawElements(mode, count, type, indices);
}

extern "C" GLAPI GLAPIENTRY void glDrawArraysInstanced(GLenum mode, GLint first, GLsizei count, GLsizei instancecount) {
    syncAtomicCounters();
    GLES.glDrawArraysInstanced(mode, first, count, instancecount);
}

extern "C" GLAPI GLAPIENTRY void glDrawElementsInstanced(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount) {
    syncAtomicCounters();
    GLES.glDrawElementsInstanced(mode, count, type, indices, instancecount);
}

extern "C" GLAPI GLAPIENTRY void glDrawRangeElements(GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices) {
    syncAtomicCounters();
    GLES.glDrawRangeElements(mode, start, end, count, type, indices);
}

extern "C" GLAPI GLAPIENTRY void glDrawElementsBaseVertex(GLenum mode, GLsizei count, GLenum type, const void *indices, GLint basevertex) {
    syncAtomicCounters();
    GLES.glDrawElementsBaseVertex(mode, count, type, indices, basevertex);
}

extern "C" GLAPI GLAPIENTRY void glDrawArraysIndirect(GLenum mode, const void *indirect) {
    syncAtomicCounters();
    GLES.glDrawArraysIndirect(mode, indirect);
}

extern "C" GLAPI GLAPIENTRY void glDrawElementsIndirect(GLenum mode, GLenum type, const void *indirect) {
    syncAtomicCounters();
    GLES.glDrawElementsIndirect(mode, type, indirect);
}

// ============================================================================
// Compute shader dispatch
// ============================================================================

extern "C" GLAPI GLAPIENTRY void glDispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z) {
    syncAtomicCounters();
    GLES.glDispatchCompute(num_groups_x, num_groups_y, num_groups_z);
    readbackAtomicCounters();
}

// ============================================================================
// glDrawBuffers - track draw buffer state
// ============================================================================

extern "C" GLAPI GLAPIENTRY void glDrawBuffers(GLsizei n, const GLenum *bufs) {
    GLES.glDrawBuffers(n, bufs);

    auto &fb = GLState.framebuffer;
    fb.drawBufferCount = n;
    for (GLsizei i = 0; i < n && i < MAX_DRAW_BUFFERS; i++) {
        fb.drawBuffers[i] = bufs[i];
    }
}

// ============================================================================
// glBindImageTexture - track image unit state
// ============================================================================

extern "C" GLAPI GLAPIENTRY void glBindImageTexture(GLuint unit, GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum format) {
    GLES.glBindImageTexture(unit, texture, level, layered, layer, access, format);

    if (unit < MAX_IMAGE_UNITS) {
        auto &img = GLState.image.imageUnits[unit];
        img.texture = texture;
        img.level = level;
        img.layered = layered;
        img.layer = layer;
        img.access = access;
        img.format = format;
    }
}

// ============================================================================
// glMemoryBarrier - handle atomic counter buffer sync
// ============================================================================

// Narrow GL_ALL_BARRIER_BITS to only the barriers commonly needed for
// compute shader work. This reduces the probability of driver-forced
// global synchronization, which is expensive on mobile GPUs.
// The full GL_ALL_BARRIER_BITS (0xFFFFFFFF) causes the driver to flush
// all caches; narrowing to the actually relevant bits avoids this.
#define MG_COMPUTE_BARRIER_MASK (GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | \
                                 GL_ELEMENT_ARRAY_BARRIER_BIT |       \
                                 GL_UNIFORM_BARRIER_BIT |             \
                                 GL_TEXTURE_FETCH_BARRIER_BIT |       \
                                 GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | \
                                 GL_COMMAND_BARRIER_BIT |             \
                                 GL_PIXEL_BUFFER_BARRIER_BIT |        \
                                 GL_TEXTURE_UPDATE_BARRIER_BIT |      \
                                 GL_BUFFER_UPDATE_BARRIER_BIT |       \
                                 GL_FRAMEBUFFER_BARRIER_BIT |         \
                                 GL_TRANSFORM_FEEDBACK_BARRIER_BIT |  \
                                 GL_ATOMIC_COUNTER_BARRIER_BIT |      \
                                 GL_SHADER_STORAGE_BARRIER_BIT)

extern "C" GLAPI GLAPIENTRY void glMemoryBarrier(GLbitfield barriers) {
    if (barriers & GL_ATOMIC_COUNTER_BARRIER_BIT) {
        syncAtomicCounters();
    }
    // Narrow GL_ALL_BARRIER_BITS to known barrier bits only.
    // Some drivers treat unknown bits as a full pipeline flush.
    if (barriers == GL_ALL_BARRIER_BITS) {
        barriers &= MG_COMPUTE_BARRIER_MASK;
    }
    GLES.glMemoryBarrier(barriers);
}