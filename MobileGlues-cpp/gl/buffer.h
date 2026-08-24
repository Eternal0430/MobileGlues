// MobileGlues - gl/buffer.h
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

#ifndef MOBILEGLUES_BUFFER_H
#define GL_GLEXT_PROTOTYPES
#include "../config/settings.h"
#include "../gles/loader.h"
#include "../includes.h"
#include "glcorearb.h"
#include "log.h"
#include "mg.h"
#include <GL/gl.h>
#include <cstddef>
#include <vector>

#ifdef __cplusplus
extern "C"
{
#endif

    GLuint gen_buffer();

    GLboolean has_buffer(GLuint key);

    void modify_buffer(GLuint key, GLuint value);

    void remove_buffer(GLuint key);

    GLuint find_real_buffer(GLuint key);

    GLuint find_bound_buffer(GLenum key);

    void set_bound_buffer_by_target(GLenum target, GLuint buffer);

    // --- PBO CPU shadow data (for BGRA swizzle without glMapBufferRange read) ---
    // MobileGL-DirectGLES keeps a CPU shadow copy of every PBO; we adopt the
    // same design but only for GL_PIXEL_UNPACK_BUFFER. This lets us do CPU-side
    // BGRA->RGBA swizzle in glTexSubImage2D/glTexImage2D without mapping the
    // source PBO for reading (which fails on many GLES drivers).
    void pbo_shadow_alloc(GLuint pbo, GLsizeiptr size, const void* data);
    void pbo_shadow_subdata(GLuint pbo, GLintptr offset, GLsizeiptr size, const void* data);
    void pbo_shadow_delete(GLuint pbo);
    // Returns a pointer to the PBO's CPU shadow data, or nullptr if none.
    // The pointer is valid until the next PBO operation on this buffer.
    const unsigned char* pbo_shadow_get(GLuint pbo);
    // Returns the size of the PBO's CPU shadow data, or 0 if none.
    GLsizeiptr pbo_shadow_size(GLuint pbo);
    // Combined lookup: returns {ptr, size} in a single locked lookup.
    const unsigned char* pbo_shadow_get_ptr_size(GLuint pbo, GLsizeiptr* outSize);
    // If the PBO is currently write-mapped, returns its shadow base pointer
    // and the mapped [offset, length) range. Returns false if not mapped.
    bool pbo_shadow_get_mapped_range(GLuint pbo, const unsigned char** outData,
                                      GLintptr* outOffset, GLsizeiptr* outLength);
    // For glMapBufferRange(GL_MAP_WRITE_BIT): returns a writable CPU pointer
    // into the shadow buffer at `offset`. Caller must call pbo_shadow_unmap
    // to flush the shadow back to GLES.
    void* pbo_shadow_map_write(GLuint pbo, GLintptr offset, GLsizeiptr length);
    void pbo_shadow_unmap(GLuint pbo);
    // Combined hot-path: ensure the shadow exists (lazily allocating if needed)
    // and write-map it in a single locked lookup. Replaces the previous 3-lock
    // sequence (pbo_shadow_get + pbo_shadow_alloc + pbo_shadow_map_write) in
    // glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, GL_MAP_WRITE_BIT). Reduces
    // lock-acquisition jitter on contended texture-upload paths, which matters
    // for frame-time stability under high CPU load.
    void* pbo_shadow_ensure_and_map_write(GLuint pbo, GLintptr offset, GLsizeiptr length);
    // Combined hot-path: snapshot the write-mapped range and clear the mapped
    // flag in a single locked lookup. Replaces the previous 2-lock sequence
    // (pbo_shadow_get_mapped_range + pbo_shadow_unmap) in
    // glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER).
    bool pbo_shadow_unmap_and_get_range(GLuint pbo, const unsigned char** outData,
                                         GLintptr* outOffset, GLsizeiptr* outLength);

    // --- Thread-local scratch buffer cache ---
    // Texture-upload swizzle path needs a tight temporary buffer per upload
    // (size = width*height*depth*4). Repeatedly calling malloc/free on every
    // glTexSubImage2D is a hot-path bottleneck. This cache keeps a per-thread
    // growable buffer that is reused across calls - acquires returns {ptr,
    // capacity}; the caller uses min(requestedSize, capacity) bytes and the
    // memory is recycled on the next acquire. NOT thread-safe across threads;
    // each thread gets its own slot.
    struct MgScratchBuffer {
        void* ptr = nullptr;
        size_t capacity = 0;
    };
    MgScratchBuffer* mg_acquire_scratch_buffer();
    // Returns true if `ptr` points into the calling thread's scratch buffer.
    // Use after swizzle_pixels_for_unpack to skip free() for scratch-owned
    // memory (the scratch buffer is reused on the next call).
    inline bool mg_scratch_owns(const void* ptr) {
        return ptr != nullptr && ptr == mg_acquire_scratch_buffer()->ptr;
    }

    GLuint find_bound_ssbo_indexed(GLuint index);

    // --- Atomic counter fast-path flag ---
    // Cached result of (atomicCounterBufferBinding != 0 && !atomicCounterData.empty()).
    // Set false at init/reset; flipped true only when atomic-counter emulation
    // is actually in use. Draw/dispatch entry points read this single bool instead
    // of performing two memory loads + branches per call.
    extern bool g_atomicCountersActive;
    void mg_update_atomic_counters_active_flag();

    GLuint gen_array();

    GLboolean has_array(GLuint key);

    void modify_array(GLuint key, GLuint value);

    void remove_array(GLuint key);

    GLuint find_real_array(GLuint key);

    GLuint find_bound_array();

    static GLenum get_binding_query(GLenum target);

    void InitBufferMap(size_t expectedSize);

    void InitVertexArrayMap(size_t expectedSize);

    GLAPI GLAPIENTRY void glGenBuffers(GLsizei n, GLuint* buffers);

    GLAPI GLAPIENTRY void glDeleteBuffers(GLsizei n, const GLuint* buffers);

    GLAPI GLAPIENTRY GLboolean glIsBuffer(GLuint buffer);

    GLAPI GLAPIENTRY void glBindBuffer(GLenum target, GLuint buffer);

    GLAPI GLAPIENTRY void glBindBufferRange(GLenum target, GLuint index, GLuint buffer, GLintptr offset,
                                            GLsizeiptr size);

    GLAPI GLAPIENTRY void glBindBufferBase(GLenum target, GLuint index, GLuint buffer);

    GLAPI GLAPIENTRY void glBindVertexBuffer(GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride);

    GLAPI GLAPIENTRY void glTexBuffer(GLenum target, GLenum internalformat, GLuint buffer);

    GLAPI GLAPIENTRY void glTexBufferRange(GLenum target, GLenum internalformat, GLuint buffer, GLintptr offset,
                                           GLsizeiptr size);

    GLAPI GLAPIENTRY GLboolean glUnmapBuffer(GLenum target);

    GLAPI GLAPIENTRY void* glMapBuffer(GLenum target, GLenum access);

    GLAPI GLAPIENTRY void* glMapBufferRange(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access);

    GLAPI GLAPIENTRY void glBufferData(GLenum target, GLsizeiptr size, const void* data, GLenum usage);

    GLAPI GLAPIENTRY void glBufferStorage(GLenum target, GLsizeiptr size, const void* data, GLbitfield flags);

    GLAPI GLAPIENTRY void glFlushMappedBufferRange(GLenum target, GLintptr offset, GLsizeiptr length);

    GLAPI GLAPIENTRY void glGenVertexArrays(GLsizei n, GLuint* arrays);

    GLAPI GLAPIENTRY void glDeleteVertexArrays(GLsizei n, const GLuint* arrays);

    GLAPI GLAPIENTRY GLboolean glIsVertexArray(GLuint array);

    GLAPI GLAPIENTRY void glBindVertexArray(GLuint array);

#ifdef __cplusplus
}
#endif

#define MOBILEGLUES_BUFFER_H

#endif // MOBILEGLUES_BUFFER_H
