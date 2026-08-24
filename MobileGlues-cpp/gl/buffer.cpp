// MobileGlues - gl/buffer.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

// ============================================================================
// ES 3.2 native → native, ES 3.2 not native → CPU simulation
// ============================================================================

#include "buffer.h"
#include "ankerl/unordered_dense.h"
#include "texture.h"

#define DEBUG 0

// ============================================================================
// Thread-local scratch buffer cache
// ============================================================================
// Hot-path texture uploads (glTexSubImage2D with BGRA swizzle) need a tight
// temporary buffer per call. Avoiding malloc/free on every call is a big
// win. We keep one growable buffer per thread; it survives across calls
// and only grows when a larger upload comes in.
static thread_local MgScratchBuffer t_scratch;

MgScratchBuffer* mg_acquire_scratch_buffer() {
    return &t_scratch;
}

// ============================================================================
// Internal State: Buffer Map Management
// ============================================================================

GLuint bound_array;
static GLint maxBufferId = 0;
static GLint maxArrayId = 0;

static std::vector<GLuint> g_gen_buffers;
static std::vector<char> g_gen_buffer_exists;
static std::vector<GLuint> g_free_buffer_ids;

static std::vector<GLuint> g_gen_arrays;
static std::vector<char> g_gen_array_exists;
static std::vector<GLuint> g_free_array_ids;

static std::vector<size_t> g_buffer_datasize;

static std::vector<GLuint> g_element_array_buffer_per_vao;

// ============================================================================
// PBO CPU shadow data (for BGRA->RGBA swizzle without glMapBufferRange read)
// ============================================================================
// Maps wrapper PBO id -> CPU shadow buffer. Only GL_PIXEL_UNPACK_BUFFERs get
// a shadow. See pbo_shadow_* API in buffer.h.
// Uses ankerl::unordered_dense::map (faster than std::unordered_map; the
// project already depends on it via DSAWrapper.cpp).
#include <mutex>
struct PboShadow {
    std::vector<unsigned char> data;
    // Active write mapping state (offset/length of the mapped region).
    GLintptr mapOffset = 0;
    GLsizeiptr mapLength = 0;
    bool mapped = false;
};
static ankerl::unordered_dense::map<GLuint, PboShadow> g_pbo_shadows;
static std::mutex g_pbo_shadow_mutex;

void pbo_shadow_alloc(GLuint pbo, GLsizeiptr size, const void* data) {
    if (pbo == 0) return;
    std::lock_guard<std::mutex> lock(g_pbo_shadow_mutex);
    // try_emplace: single hash lookup that finds-or-inserts (one fewer hash
    // than operator[], which default-constructs then is overwritten).
    auto [it, inserted] = g_pbo_shadows.try_emplace(pbo);
    auto& s = it->second;
    if (data && size > 0) {
        // assign() does resize + copy in one pass (avoids zero-fill by resize
        // then immediate overwrite by memcpy).
        s.data.assign(static_cast<const unsigned char*>(data),
                      static_cast<const unsigned char*>(data) + size);
    } else {
        s.data.resize(size);
    }
    s.mapped = false;
}

void pbo_shadow_subdata(GLuint pbo, GLintptr offset, GLsizeiptr size, const void* data) {
    if (pbo == 0 || !data || size <= 0) return;
    std::lock_guard<std::mutex> lock(g_pbo_shadow_mutex);
    auto it = g_pbo_shadows.find(pbo);
    if (it == g_pbo_shadows.end()) return;
    auto& s = it->second;
    if (offset < 0) offset = 0;
    if (offset + size > (GLsizeiptr)s.data.size()) {
        // Grow to fit (matches glBufferSubData's grow-on-overflow semantics
        // on some drivers; GLES would error, but we mirror shadow first).
        s.data.resize(offset + size);
    }
    memcpy(s.data.data() + offset, data, size);
}

void pbo_shadow_delete(GLuint pbo) {
    if (pbo == 0) return;
    std::lock_guard<std::mutex> lock(g_pbo_shadow_mutex);
    g_pbo_shadows.erase(pbo);
}

const unsigned char* pbo_shadow_get(GLuint pbo) {
    if (pbo == 0) return nullptr;
    std::lock_guard<std::mutex> lock(g_pbo_shadow_mutex);
    auto it = g_pbo_shadows.find(pbo);
    if (it == g_pbo_shadows.end()) return nullptr;
    return it->second.data.data();
}

GLsizeiptr pbo_shadow_size(GLuint pbo) {
    if (pbo == 0) return 0;
    std::lock_guard<std::mutex> lock(g_pbo_shadow_mutex);
    auto it = g_pbo_shadows.find(pbo);
    if (it == g_pbo_shadows.end()) return 0;
    return static_cast<GLsizeiptr>(it->second.data.size());
}

// Combined lookup: returns {ptr, size} in a single locked lookup. Use this
// instead of calling pbo_shadow_get() + pbo_shadow_size() separately to halve
// the mutex acquisitions on the hot texture-upload path.
const unsigned char* pbo_shadow_get_ptr_size(GLuint pbo, GLsizeiptr* outSize) {
    if (pbo == 0) { if (outSize) *outSize = 0; return nullptr; }
    std::lock_guard<std::mutex> lock(g_pbo_shadow_mutex);
    auto it = g_pbo_shadows.find(pbo);
    if (it == g_pbo_shadows.end()) { if (outSize) *outSize = 0; return nullptr; }
    if (outSize) *outSize = static_cast<GLsizeiptr>(it->second.data.size());
    return it->second.data.data();
}

// Returns the mapped region [offset, offset+length) for a write-mapped PBO.
// If the PBO is currently write-mapped, returns {ptr, offset, length} in a
// single locked lookup so glUnmapBuffer can sync only the dirty region
// instead of the whole shadow.
bool pbo_shadow_get_mapped_range(GLuint pbo, const unsigned char** outData, GLintptr* outOffset, GLsizeiptr* outLength) {
    if (pbo == 0) return false;
    std::lock_guard<std::mutex> lock(g_pbo_shadow_mutex);
    auto it = g_pbo_shadows.find(pbo);
    if (it == g_pbo_shadows.end()) return false;
    auto& s = it->second;
    if (!s.mapped) return false;
    if (outData)    *outData    = s.data.data();
    if (outOffset)  *outOffset  = s.mapOffset;
    if (outLength)  *outLength  = s.mapLength;
    return true;
}

void* pbo_shadow_map_write(GLuint pbo, GLintptr offset, GLsizeiptr length) {
    if (pbo == 0 || length <= 0) return nullptr;
    std::lock_guard<std::mutex> lock(g_pbo_shadow_mutex);
    auto it = g_pbo_shadows.find(pbo);
    if (it == g_pbo_shadows.end()) return nullptr;
    auto& s = it->second;
    if (offset < 0) offset = 0;
    if (offset + length > (GLsizeiptr)s.data.size()) {
        s.data.resize(offset + length);
    }
    s.mapOffset = offset;
    s.mapLength = length;
    s.mapped = true;
    return s.data.data() + offset;
}

// Combined: ensure the shadow exists (lazily allocating if needed) and write-map
// it in a single locked lookup. Replaces the previous 3-lock sequence in
// glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, GL_MAP_WRITE_BIT):
//   pbo_shadow_get() + pbo_shadow_alloc() + pbo_shadow_map_write()
// On contended hot paths (high CPU load) the lock-acquisition jitter from 3
// separate acquisitions was a measurable source of frame-time variance; this
// collapses them into one critical section. Returns the mapped write pointer,
// or nullptr if the (re)allocation failed.
void* pbo_shadow_ensure_and_map_write(GLuint pbo, GLintptr offset, GLsizeiptr length) {
    if (pbo == 0 || length <= 0) return nullptr;
    if (offset < 0) offset = 0;
    std::lock_guard<std::mutex> lock(g_pbo_shadow_mutex);
    // try_emplace: single hash lookup that finds-or-inserts. The previous
    // find() + emplace() sequence did two hash lookups; this collapses them
    // into one on the hot texture-upload path.
    auto [it, inserted] = g_pbo_shadows.try_emplace(pbo);
    auto& s = it->second;
    if (offset + length > (GLsizeiptr)s.data.size()) {
        s.data.resize(offset + length);
    }
    s.mapOffset = offset;
    s.mapLength = length;
    s.mapped = true;
    return s.data.data() + offset;
}

void pbo_shadow_unmap(GLuint pbo) {
    if (pbo == 0) return;
    std::lock_guard<std::mutex> lock(g_pbo_shadow_mutex);
    auto it = g_pbo_shadows.find(pbo);
    if (it == g_pbo_shadows.end()) return;
    auto& s = it->second;
    s.mapped = false;
}

// Combined: snapshot the current write-mapped range and clear the mapped flag
// in a single locked lookup. Replaces the previous 2-lock sequence in
// glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER):
//   pbo_shadow_get_mapped_range() + pbo_shadow_unmap()
// Halves lock acquisitions on the texture-upload hot path, which matters for
// frame-time stability under CPU contention. Returns false if the PBO has no
// shadow or is not currently write-mapped.
bool pbo_shadow_unmap_and_get_range(GLuint pbo, const unsigned char** outData,
                                    GLintptr* outOffset, GLsizeiptr* outLength) {
    if (pbo == 0) return false;
    std::lock_guard<std::mutex> lock(g_pbo_shadow_mutex);
    auto it = g_pbo_shadows.find(pbo);
    if (it == g_pbo_shadows.end()) return false;
    auto& s = it->second;
    if (!s.mapped) return false;
    if (outData)   *outData   = s.data.data();
    if (outOffset) *outOffset = s.mapOffset;
    if (outLength) *outLength = s.mapLength;
    s.mapped = false;
    return true;
}

// ============================================================================
// Buffer Binding Index Tracking
// ============================================================================

enum BindingIndex : int {
    BI_ARRAY_BUFFER = 0,
    BI_ATOMIC_COUNTER,
    BI_COPY_READ,
    BI_COPY_WRITE,
    BI_DRAW_INDIRECT,
    BI_DISPATCH_INDIRECT,
    BI_ELEMENT_ARRAY,
    BI_PIXEL_PACK,
    BI_PIXEL_UNPACK,
    BI_QUERY_BUFFER,
    BI_SHADER_STORAGE,
    BI_TRANSFORM_FEEDBACK,
    BI_UNIFORM_BUFFER,
    BINDING_COUNT
};
static std::array<GLuint, BINDING_COUNT> g_bound_buffers_arr = {0};

// ============================================================================
// Buffer Map Helpers: Capacity & Lifecycle
// ============================================================================

static inline __attribute__((always_inline)) int ensure_buffer_capacity(GLuint id) {
    if (id < (GLuint)g_gen_buffers.size()) [[likely]] return 0;
    g_gen_buffers.resize(id + 1, 0);
    g_gen_buffer_exists.resize(id + 1, 0);
    if (g_buffer_datasize.size() <= (size_t)id) g_buffer_datasize.resize(id + 1, 0);
    return 0;
}

static inline __attribute__((always_inline)) int ensure_array_capacity(GLuint id) {
    if (id < (GLuint)g_gen_arrays.size()) [[likely]] return 0;
    g_gen_arrays.resize(id + 1, 0);
    g_gen_array_exists.resize(id + 1, 0);
    if (g_element_array_buffer_per_vao.size() <= (size_t)id) g_element_array_buffer_per_vao.resize(id + 1, 0);
    return 0;
}

GLuint gen_buffer() {
    if (!g_free_buffer_ids.empty()) {
        GLuint id = g_free_buffer_ids.back();
        g_free_buffer_ids.pop_back();
        ensure_buffer_capacity(id);
        g_gen_buffers[id] = 0;
        g_gen_buffer_exists[id] = 1;
        g_buffer_datasize[id] = 0;
        if (id > (GLuint)maxBufferId) maxBufferId = id;
        return id;
    }
    maxBufferId++;
    ensure_buffer_capacity((GLuint)maxBufferId);
    g_gen_buffers[maxBufferId] = 0;
    g_gen_buffer_exists[maxBufferId] = 1;
    g_buffer_datasize[maxBufferId] = 0;
    return (GLuint)maxBufferId;
}

GLboolean has_buffer(GLuint key) {
    return key < g_gen_buffer_exists.size() ? (g_gen_buffer_exists[key] != 0) : 0;
}

void modify_buffer(GLuint key, GLuint value) {
    if (key >= g_gen_buffers.size()) [[unlikely]] ensure_buffer_capacity(key);
    g_gen_buffers[key] = value;
    g_gen_buffer_exists[key] = 1;
}

void remove_buffer(GLuint key) {
    if (key < g_gen_buffer_exists.size() && g_gen_buffer_exists[key]) {
        g_gen_buffer_exists[key] = 0;
        g_gen_buffers[key] = 0;
        if (key < g_buffer_datasize.size()) g_buffer_datasize[key] = 0;
        g_free_buffer_ids.push_back(key);
    }
}

GLuint find_real_buffer(GLuint key) {
    if (key < g_gen_buffers.size() && g_gen_buffer_exists[key]) return g_gen_buffers[key];
    return 0;
}

// Combined lookup: returns {real_buffer, exists} in a single bounds check.
// g_gen_buffers and g_gen_buffer_exists are always resized together,
// so checking one is sufficient for both.
static inline __attribute__((always_inline)) std::pair<GLuint, bool> find_real_buffer_with_exists(GLuint key) {
    if (key < g_gen_buffers.size() && g_gen_buffer_exists[key]) [[likely]]
        return {g_gen_buffers[key], true};
    return {0, false};
}

// ============================================================================
// Atomic counter fast-path flag
// ============================================================================
// Reflects whether GLState.buffer.atomicCounter* has any active binding +
// data. Draw/dispatch entry points consult this single bool instead of
// performing two dependent memory loads + branches per call. Updated by
// mg_update_atomic_counters_active_flag() whenever the underlying state
// changes (reset, bind, unbind).
bool g_atomicCountersActive = false;

#include "state.h"
void mg_update_atomic_counters_active_flag() {
    auto &bs = GLState.buffer;
    g_atomicCountersActive = (bs.atomicCounterBufferBinding != 0) && !bs.atomicCounterData.empty();
}

// Fast-path: direct assignment without capacity checks, for use when the caller
// has already verified the buffer exists (e.g. after find_real_buffer_with_exists).
// Avoids redundant ensure_buffer_capacity in hot paths like glBindBuffer,
// glBindBufferRange, glBindBufferBase, glBindVertexBuffer, glTexBuffer.
void modify_buffer_direct(GLuint key, GLuint value) {
    g_gen_buffers[key] = value;
}

GLuint get_ibo_by_vao(GLuint vao) {
    if (vao < g_element_array_buffer_per_vao.size()) return g_element_array_buffer_per_vao[vao];
    return 0;
}

GLuint find_bound_array() {
    return bound_array;
}

void update_vao_ibo_binding(GLuint vao, GLuint ibo) {
    ensure_array_capacity(vao);
    g_element_array_buffer_per_vao[vao] = ibo;
}

void set_buffer_data_size(GLuint buffer, size_t size) {
    ensure_buffer_capacity(buffer);
    g_buffer_datasize[buffer] = size;
}

size_t get_buffer_data_size(GLuint buffer) {
    if (buffer < g_buffer_datasize.size()) return g_buffer_datasize[buffer];
    return 0;
}

// ============================================================================
// Buffer Binding Helpers
// ============================================================================

static inline int binding_target_to_index(GLenum target) {
    switch (target) {
    case GL_ARRAY_BUFFER:             return BI_ARRAY_BUFFER;
    case GL_ATOMIC_COUNTER_BUFFER:    return BI_ATOMIC_COUNTER;
    case GL_COPY_READ_BUFFER:         return BI_COPY_READ;
    case GL_COPY_WRITE_BUFFER:        return BI_COPY_WRITE;
    case GL_DRAW_INDIRECT_BUFFER:     return BI_DRAW_INDIRECT;
    case GL_DISPATCH_INDIRECT_BUFFER: return BI_DISPATCH_INDIRECT;
    case GL_ELEMENT_ARRAY_BUFFER:     return BI_ELEMENT_ARRAY;
    case GL_PIXEL_PACK_BUFFER:        return BI_PIXEL_PACK;
    case GL_PIXEL_UNPACK_BUFFER:      return BI_PIXEL_UNPACK;
    case GL_QUERY_BUFFER:             return BI_QUERY_BUFFER;
    case GL_SHADER_STORAGE_BUFFER:    return BI_SHADER_STORAGE;
    case GL_TRANSFORM_FEEDBACK_BUFFER:return BI_TRANSFORM_FEEDBACK;
    case GL_UNIFORM_BUFFER:           return BI_UNIFORM_BUFFER;
    default:                          return -1;
    }
}

void set_bound_buffer_by_target(GLenum target, GLuint buffer) {
    int idx = binding_target_to_index(target);
    if (idx >= 0) g_bound_buffers_arr[idx] = buffer;
}

GLuint find_bound_buffer(GLenum key) {
    // Special case: ELEMENT_ARRAY_BUFFER uses VAO tracking
    if (key == GL_ELEMENT_ARRAY_BUFFER_BINDING) {
        return get_ibo_by_vao(find_bound_array());
    }
    switch (key) {
    case GL_ARRAY_BUFFER_BINDING:              return g_bound_buffers_arr[BI_ARRAY_BUFFER];
    case GL_ATOMIC_COUNTER_BUFFER_BINDING:     return g_bound_buffers_arr[BI_ATOMIC_COUNTER];
    case GL_COPY_READ_BUFFER_BINDING:          return g_bound_buffers_arr[BI_COPY_READ];
    case GL_COPY_WRITE_BUFFER_BINDING:         return g_bound_buffers_arr[BI_COPY_WRITE];
    case GL_DRAW_INDIRECT_BUFFER_BINDING:      return g_bound_buffers_arr[BI_DRAW_INDIRECT];
    case GL_DISPATCH_INDIRECT_BUFFER_BINDING:  return g_bound_buffers_arr[BI_DISPATCH_INDIRECT];
    case GL_PIXEL_PACK_BUFFER_BINDING:         return g_bound_buffers_arr[BI_PIXEL_PACK];
    case GL_PIXEL_UNPACK_BUFFER_BINDING:       return g_bound_buffers_arr[BI_PIXEL_UNPACK];
    case GL_QUERY_BUFFER_BINDING:              return g_bound_buffers_arr[BI_QUERY_BUFFER];
    case GL_SHADER_STORAGE_BUFFER_BINDING:     return g_bound_buffers_arr[BI_SHADER_STORAGE];
    case GL_TRANSFORM_FEEDBACK_BUFFER_BINDING: return g_bound_buffers_arr[BI_TRANSFORM_FEEDBACK];
    case GL_UNIFORM_BUFFER_BINDING:            return g_bound_buffers_arr[BI_UNIFORM_BUFFER];
    default:                                   return 0;
    }
}

static GLenum get_binding_query(GLenum target) {
    switch (target) {
    case GL_ARRAY_BUFFER:              return GL_ARRAY_BUFFER_BINDING;
    case GL_ELEMENT_ARRAY_BUFFER:      return GL_ELEMENT_ARRAY_BUFFER_BINDING;
    case GL_PIXEL_PACK_BUFFER:         return GL_PIXEL_PACK_BUFFER_BINDING;
    case GL_PIXEL_UNPACK_BUFFER:       return GL_PIXEL_UNPACK_BUFFER_BINDING;
    case GL_QUERY_BUFFER:              return GL_QUERY_BUFFER_BINDING;
    case GL_UNIFORM_BUFFER:            return GL_UNIFORM_BUFFER_BINDING;
    case GL_TRANSFORM_FEEDBACK_BUFFER: return GL_TRANSFORM_FEEDBACK_BUFFER_BINDING;
    case GL_COPY_READ_BUFFER:          return GL_COPY_READ_BUFFER_BINDING;
    case GL_COPY_WRITE_BUFFER:         return GL_COPY_WRITE_BUFFER_BINDING;
    case GL_DRAW_INDIRECT_BUFFER:      return GL_DRAW_INDIRECT_BUFFER_BINDING;
    case GL_SHADER_STORAGE_BUFFER:     return GL_SHADER_STORAGE_BUFFER_BINDING;
    case GL_DISPATCH_INDIRECT_BUFFER:  return GL_DISPATCH_INDIRECT_BUFFER_BINDING;
    case GL_ATOMIC_COUNTER_BUFFER:     return GL_ATOMIC_COUNTER_BUFFER_BINDING;
    default:                           return 0;
    }
}

// ============================================================================
// Vertex Array Map Helpers
// ============================================================================

GLuint gen_array() {
    if (!g_free_array_ids.empty()) {
        GLuint id = g_free_array_ids.back();
        g_free_array_ids.pop_back();
        ensure_array_capacity(id);
        g_gen_arrays[id] = 0;
        g_gen_array_exists[id] = 1;
        g_element_array_buffer_per_vao[id] = 0;
        if (id > (GLuint)maxArrayId) maxArrayId = id;
        return id;
    }
    maxArrayId++;
    ensure_array_capacity((GLuint)maxArrayId);
    g_gen_arrays[maxArrayId] = 0;
    g_gen_array_exists[maxArrayId] = 1;
    g_element_array_buffer_per_vao[maxArrayId] = 0;
    return (GLuint)maxArrayId;
}

GLboolean has_array(GLuint key) {
    return key < g_gen_array_exists.size() ? (g_gen_array_exists[key] != 0) : 0;
}

void modify_array(GLuint key, GLuint value) {
    if (key >= g_gen_arrays.size()) ensure_array_capacity(key);
    g_gen_arrays[key] = value;
    g_gen_array_exists[key] = 1;
}

void remove_array(GLuint key) {
    if (key < g_gen_array_exists.size() && g_gen_array_exists[key]) {
        g_gen_array_exists[key] = 0;
        g_gen_arrays[key] = 0;
        if (key < g_element_array_buffer_per_vao.size()) g_element_array_buffer_per_vao[key] = 0;
        g_free_array_ids.push_back(key);
    }
}

GLuint find_real_array(GLuint key) {
    if (key < g_gen_arrays.size() && g_gen_array_exists[key]) return g_gen_arrays[key];
    return 0;
}

// ============================================================================
// Map Initialization
// ============================================================================

void InitBufferMap(size_t expectedSize) {
    g_gen_buffers.reserve(expectedSize + 2);
    g_gen_buffer_exists.reserve(expectedSize + 2);
    g_buffer_datasize.reserve(expectedSize + 2);
    g_gen_buffers.resize(1, 0);
    g_gen_buffer_exists.resize(1, 0);
    g_buffer_datasize.resize(1, 0);
}

void InitVertexArrayMap(size_t expectedSize) {
    g_gen_arrays.reserve(expectedSize + 2);
    g_gen_array_exists.reserve(expectedSize + 2);
    g_element_array_buffer_per_vao.reserve(expectedSize + 2);
    g_gen_arrays.resize(1, 0);
    g_gen_array_exists.resize(1, 0);
    g_element_array_buffer_per_vao.resize(1, 0);
}

// ============================================================================
// Buffer Object Lifecycle (ES 3.2 native)
// ============================================================================

void glGenBuffers(GLsizei n, GLuint* buffers) {
    LOG()
    LOG_D("glGenBuffers(%i, %p)", n, buffers)
    for (int i = 0; i < n; ++i) {
        buffers[i] = gen_buffer();
    }
}

void glDeleteBuffers(GLsizei n, const GLuint* buffers) {
    LOG()
    LOG_D("glDeleteBuffers(%i, %p)", n, buffers)
    for (int i = 0; i < n; ++i) {
        auto [real_buff, exists] = find_real_buffer_with_exists(buffers[i]);
        if (exists) {
            GLES.glDeleteBuffers(1, &real_buff);
            CHECK_GL_ERROR
        }
        // Clean up any PBO shadow data for this buffer.
        pbo_shadow_delete(buffers[i]);
        remove_buffer(buffers[i]);
    }
}

GLboolean glIsBuffer(GLuint buffer) {
    LOG()
    LOG_D("glIsBuffer, buffer = %d", buffer)
    return has_buffer(buffer);
}

// ============================================================================
// Buffer Binding (ES 3.2 native, with buffer map)
// ============================================================================

void glBindBuffer(GLenum target, GLuint buffer) {
    LOG()
    LOG_D("glBindBuffer, target = %s, buffer = %d", glEnumToString(target), buffer)
    set_bound_buffer_by_target(target, buffer);
    // save ibo binding to vao
    if (target == GL_ELEMENT_ARRAY_BUFFER) {
        update_vao_ibo_binding(find_bound_array(), buffer);
    }

    if (buffer == 0) [[unlikely]] {
        GLES.glBindBuffer(target, buffer);
        CHECK_GL_ERROR
        return;
    }
    auto [real_buffer, exists] = find_real_buffer_with_exists(buffer);
    if (!exists) [[unlikely]] {
        GLES.glBindBuffer(target, buffer);
        CHECK_GL_ERROR
        return;
    }
    if (!real_buffer) {
        GLES.glGenBuffers(1, &real_buffer);
        modify_buffer_direct(buffer, real_buffer);
        CHECK_GL_ERROR
    }
    LOG_D("glBindBuffer: %d -> %d", buffer, real_buffer)
    GLES.glBindBuffer(target, real_buffer);
    CHECK_GL_ERROR
}

// ============================================================================
// Buffer Data Upload (ES 3.2 native, with buffer map)
// ============================================================================

void glBufferData(GLenum target, GLsizeiptr size, const void* data, GLenum usage) {
    LOG()
    LOG_D("glBufferData, target = %s, size = %d, data = 0x%x, usage = %s", glEnumToString(target), size, data,
          glEnumToString(usage))
    GLES.glBufferData(target, size, data, usage);
    int idx = binding_target_to_index(target);
    if (idx >= 0) {
        set_buffer_data_size(g_bound_buffers_arr[idx], size);
        // Sync PBO shadow for GL_PIXEL_UNPACK_BUFFER. Use idx (already known)
        // instead of re-checking target.
        if (idx == BI_PIXEL_UNPACK) {
            pbo_shadow_alloc(g_bound_buffers_arr[idx], size, data);
        }
    }
    CHECK_GL_ERROR
}

void glBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, const void* data) {
    LOG()
    LOG_D("glBufferSubData, target = %s, offset = %d, size = %d, data = %p", glEnumToString(target), offset, size, data)
    GLES.glBufferSubData(target, offset, size, data);
    // Sync PBO shadow for GL_PIXEL_UNPACK_BUFFER. target is known at this
    // point so index g_bound_buffers_arr directly (skip the switch).
    if (target == GL_PIXEL_UNPACK_BUFFER) {
        pbo_shadow_subdata(g_bound_buffers_arr[BI_PIXEL_UNPACK], offset, size, data);
    }
    CHECK_GL_ERROR
}

// ============================================================================
// Buffer Mapping (ES 3.2 native)
// ============================================================================

void* glMapBuffer(GLenum target, GLenum access) {
    LOG()
    LOG_D("glMapBuffer, target = %s, access = %s", glEnumToString(target), glEnumToString(access))
    if (g_gles_caps.GL_OES_mapbuffer) {
        return GLES.glMapBufferOES(target, access);
    }
    GLint buffer_size;
    GLES.glGetBufferParameteriv(target, GL_BUFFER_SIZE, &buffer_size);
    if (buffer_size <= 0) {
        return nullptr;
    }
    GLbitfield flags = 0;
    switch (access) {
    case GL_READ_ONLY:  flags = GL_MAP_READ_BIT;                              break;
    case GL_WRITE_ONLY: flags = GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT; break;
    case GL_READ_WRITE: flags = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT;            break;
    default:            return nullptr;
    }
    void* ptr = glMapBufferRange(target, 0, buffer_size, flags);
    return ptr;
}

void* glMapBufferRange(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access) {
    LOG()
    if (global_settings.buffer_coherent_as_flush) access &= ~GL_MAP_FLUSH_EXPLICIT_BIT;
    // For write mappings of GL_PIXEL_UNPACK_BUFFER, return a pointer into the
    // CPU shadow buffer so that subsequent glTexSubImage2D can swizzle the
    // data directly from CPU memory. The actual GLES buffer is still mapped
    // and the shadow will be synced to GLES on glUnmapBuffer.
    //
    // Hot path: a single combined locked lookup (pbo_shadow_ensure_and_map_write)
    // replaces the previous 3-lock sequence (pbo_shadow_get + pbo_shadow_alloc
    // + pbo_shadow_map_write). Under high CPU load the lock-acquisition jitter
    // from 3 separate acquisitions was a measurable source of frame-time
    // variance; collapsing them into one critical section stabilises the
    // upload path.
    if (target == GL_PIXEL_UNPACK_BUFFER && (access & GL_MAP_WRITE_BIT)) {
        // target == GL_PIXEL_UNPACK_BUFFER implies binding_target_to_index()
        // returns BI_PIXEL_UNPACK (never -1), so skip the switch and index the
        // binding table directly on this hot path.
        GLuint pbo = g_bound_buffers_arr[BI_PIXEL_UNPACK];
        if (pbo != 0) [[likely]] {
            void* shadowPtr = pbo_shadow_ensure_and_map_write(pbo, offset, length);
            if (shadowPtr) [[likely]] {
                // Still map GLES buffer for write so the data lands in
                // the real buffer too (for non-texture reads of PBO).
                // Use GL_MAP_WRITE_BIT only (no INVALIDATE since we
                // preserve shadow). Actually, we don't need GLES mapping
                // at all if we sync via glBufferSubData on unmap.
                return shadowPtr;
            }
        }
    }
    return GLES.glMapBufferRange(target, offset, length, access);
}

GLboolean glUnmapBuffer(GLenum target) {
    LOG()
    LOG_D("%s(%s)", __func__, glEnumToString(target));
    // For PBO write mappings, we returned a pointer into the CPU shadow.
    // Now sync only the dirty mapped region to the GLES buffer via
    // glBufferSubData (which is always supported, unlike
    // glMapBufferRange(GL_MAP_READ_BIT)). Syncing only the mapped range
    // avoids uploading the whole shadow when the app mapped a small slice.
    //
    // Hot path: pbo_shadow_unmap_and_get_range collapses the previous
    // 2-lock sequence (pbo_shadow_get_mapped_range + pbo_shadow_unmap) into
    // a single locked lookup. Under high CPU load the lock-acquisition
    // jitter from 2 separate acquisitions was a measurable source of
    // frame-time variance; collapsing them into one critical section
    // stabilises the upload path.
    if (target == GL_PIXEL_UNPACK_BUFFER) {
        // target == GL_PIXEL_UNPACK_BUFFER implies binding_target_to_index()
        // returns BI_PIXEL_UNPACK (never -1), so skip the switch and index the
        // binding table directly on this hot path.
        GLuint pbo = g_bound_buffers_arr[BI_PIXEL_UNPACK];
        if (pbo != 0) [[likely]] {
            const unsigned char* shadowBase = nullptr;
            GLintptr mapOffset = 0;
            GLsizeiptr mapLength = 0;
            if (pbo_shadow_unmap_and_get_range(pbo, &shadowBase, &mapOffset, &mapLength)) [[likely]] {
                if (shadowBase && mapLength > 0) {
                    GLES.glBufferSubData(target, mapOffset, mapLength,
                                          shadowBase + mapOffset);
                }
            }
            CHECK_GL_ERROR
            return GL_TRUE;
        }
    }
    if (g_gles_caps.GL_OES_mapbuffer) return GLES.glUnmapBuffer(target);

    GLboolean result = GLES.glUnmapBuffer(target);
    CHECK_GL_ERROR
    return result;
}

void glFlushMappedBufferRange(GLenum target, GLintptr offset, GLsizeiptr length) {
    LOG()
    if (!global_settings.buffer_coherent_as_flush) GLES.glFlushMappedBufferRange(target, offset, length);
}

// ============================================================================
// Buffer Range / Base Binding (ES 3.2 native, with buffer map)
// ============================================================================

struct atomic_buffer {
    GLuint id;
    GLsizeiptr size;
    GLintptr offset;
};

static std::vector<atomic_buffer> g_buffer_map_atomic_buffer_info;
static std::vector<GLuint> g_buffer_map_ssbo_id;

// Track indexed SSBO binding (stores the real GL buffer ID for CPU-side lookup,
// avoiding glGetIntegeri_v GPU round-trips in the multidraw compute path).
static inline void track_ssbo_indexed(GLuint index, GLuint real_id) {
    if (g_buffer_map_ssbo_id.empty()) {
        g_buffer_map_ssbo_id.resize(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, 0);
    }
    if (index < g_buffer_map_ssbo_id.size()) g_buffer_map_ssbo_id[index] = real_id;
}

GLuint find_bound_ssbo_indexed(GLuint index) {
    if (index < g_buffer_map_ssbo_id.size()) return g_buffer_map_ssbo_id[index];
    return 0;
}

void bindAllAtomicCounterAsSSBO() {
    const size_t count = g_buffer_map_atomic_buffer_info.size();
    for (size_t i = 0; i < count; ++i) {
        const auto& buf = g_buffer_map_atomic_buffer_info[i];
        if (buf.id != 0) {
            GLuint realID = find_real_buffer(buf.id);
            GLES.glBindBufferRange(GL_SHADER_STORAGE_BUFFER, i, realID, buf.offset, buf.size);
            LOG_D("Bound atomic counter buffer %u(real: %u) as SSBO at index %zu", buf.id, realID, i);
        }
    }
}

void glBindBufferRange(GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size) {
    LOG()
    LOG_D("glBindBufferRange, target = %s, index = %d, buffer = %d, offset = %p, size = %zi", glEnumToString(target),
          index, buffer, (void*)offset, size)

    if (buffer == 0) {
        GLES.glBindBufferRange(target, index, buffer, offset, size);
        if (target == GL_SHADER_STORAGE_BUFFER) track_ssbo_indexed(index, 0);
        CHECK_GL_ERROR
        return;
    }
    auto [real_buffer, exists] = find_real_buffer_with_exists(buffer);
    if (!exists) {
        GLES.glBindBufferRange(target, index, buffer, offset, size);
        if (target == GL_SHADER_STORAGE_BUFFER) track_ssbo_indexed(index, buffer);
        CHECK_GL_ERROR
        return;
    }
    if (!real_buffer) {
        GLES.glGenBuffers(1, &real_buffer);
        modify_buffer_direct(buffer, real_buffer);
        CHECK_GL_ERROR
    }
    GLES.glBindBufferRange(target, index, real_buffer, offset, size);
    if (target == GL_ATOMIC_COUNTER_BUFFER) {
        if (g_buffer_map_atomic_buffer_info.empty()) {
            g_buffer_map_atomic_buffer_info.resize(GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS, {});
        }
        g_buffer_map_atomic_buffer_info[index] = {buffer, size, offset};
        // Atomic counter emulation state is now potentially active; refresh
        // the cached fast-path flag consulted by every draw/dispatch.
        GLState.buffer.atomicCounterBufferBinding = buffer;
        GLState.buffer.atomicCounterBufferSize = size;
        GLState.buffer.atomicCounterBufferOffset = offset;
        if (buffer != 0 && !GLState.buffer.atomicCounterData.empty()) {
            g_atomicCountersActive = true;
        } else {
            g_atomicCountersActive = false;
        }
    } else if (target == GL_SHADER_STORAGE_BUFFER) {
        track_ssbo_indexed(index, real_buffer);
    }
    CHECK_GL_ERROR
}

void glBindBufferBase(GLenum target, GLuint index, GLuint buffer) {
    LOG()
    LOG_D("glBindBufferBase, target = %s, index = %d, buffer = %d", glEnumToString(target), index, buffer)

    if (buffer == 0) {
        GLES.glBindBufferBase(target, index, buffer);
        if (target == GL_SHADER_STORAGE_BUFFER) track_ssbo_indexed(index, 0);
        CHECK_GL_ERROR
        return;
    }
    auto [real_buffer, exists] = find_real_buffer_with_exists(buffer);
    if (!exists) {
        GLES.glBindBufferBase(target, index, buffer);
        if (target == GL_SHADER_STORAGE_BUFFER) track_ssbo_indexed(index, buffer);
        CHECK_GL_ERROR
        return;
    }
    if (!real_buffer) {
        GLES.glGenBuffers(1, &real_buffer);
        modify_buffer_direct(buffer, real_buffer);
        CHECK_GL_ERROR
    }
    GLES.glBindBufferBase(target, index, real_buffer);
    if (target == GL_SHADER_STORAGE_BUFFER) {
        track_ssbo_indexed(index, real_buffer);
    }
    CHECK_GL_ERROR
}

// ============================================================================
// Vertex Buffer Binding (ES 3.2 native, with buffer map)
// ============================================================================

void glBindVertexBuffer(GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride) {
    LOG()
    LOG_D("glBindVertexBuffer, bindingindex = %d, buffer = %d, offset = %p, stride = %i", bindingindex, buffer, offset,
          stride)
    if (buffer == 0) [[unlikely]] {
        GLES.glBindVertexBuffer(bindingindex, buffer, offset, stride);
        CHECK_GL_ERROR
        return;
    }
    auto [real_buffer, exists] = find_real_buffer_with_exists(buffer);
    if (!exists) [[unlikely]] {
        GLES.glBindVertexBuffer(bindingindex, buffer, offset, stride);
        CHECK_GL_ERROR
        return;
    }
    if (!real_buffer) {
        GLES.glGenBuffers(1, &real_buffer);
        modify_buffer_direct(buffer, real_buffer);
        CHECK_GL_ERROR
    }
    GLES.glBindVertexBuffer(bindingindex, real_buffer, offset, stride);
    CHECK_GL_ERROR
}

// ============================================================================
// Buffer Copy (ES 3.2 native)
// ============================================================================

void glCopyBufferSubData(GLenum readTarget, GLenum writeTarget, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size) {
    LOG()
    LOG_D("glCopyBufferSubData, readTarget = %s, writeTarget = %s, readOffset = %d, writeOffset = %d, size = %d",
          glEnumToString(readTarget), glEnumToString(writeTarget), readOffset, writeOffset, size)
    GLES.glCopyBufferSubData(readTarget, writeTarget, readOffset, writeOffset, size);
    CHECK_GL_ERROR
}

// ============================================================================
// Buffer Parameter Queries (ES 3.2 native)
// ============================================================================

void glGetBufferParameteriv(GLenum target, GLenum pname, GLint* params) {
    LOG()
    LOG_D("glGetBufferParameteriv, target = %s, pname = %s", glEnumToString(target), glEnumToString(pname))
    GLES.glGetBufferParameteriv(target, pname, params);
    CHECK_GL_ERROR
}

void glGetBufferParameteri64v(GLenum target, GLenum pname, GLint64* params) {
    LOG()
    LOG_D("glGetBufferParameteri64v, target = %s, pname = %s", glEnumToString(target), glEnumToString(pname))
    GLES.glGetBufferParameteri64v(target, pname, params);
    CHECK_GL_ERROR
}

void glGetBufferPointerv(GLenum target, GLenum pname, void** params) {
    LOG()
    LOG_D("glGetBufferPointerv, target = %s, pname = %s", glEnumToString(target), glEnumToString(pname))
    GLES.glGetBufferPointerv(target, pname, params);
    CHECK_GL_ERROR
}

// ============================================================================
// Buffer Storage (ES 3.2 native via EXT)
// ============================================================================

void glBufferStorage(GLenum target, GLsizeiptr size, const void* data, GLbitfield flags) {
    LOG()
    if (GLES.glBufferStorageEXT) {
        if (global_settings.buffer_coherent_as_flush &&
            ((flags & GL_MAP_PERSISTENT_BIT) != 0 || (flags & GL_DYNAMIC_STORAGE_BIT) != 0))
            flags |= (GL_MAP_WRITE_BIT | GL_MAP_COHERENT_BIT | GL_MAP_PERSISTENT_BIT);
        GLES.glBufferStorageEXT(target, size, data, flags);
        // Mirror glBufferData: keep the PBO CPU shadow in sync so the BGRA
        // swizzle in texture.cpp can read from CPU memory instead of mapping
        // the (immutable, possibly non-DYNAMIC_STORAGE) GLES buffer for read.
        // Without this, Xaero-style glBufferStorage PBO uploads would miss
        // the shadow and fall back to glCopyBufferSubData, which may fail on
        // some GLES drivers, leaving the texture un-swizzled (blue).
        if (target == GL_PIXEL_UNPACK_BUFFER) {
            int idx = binding_target_to_index(target);
            if (idx >= 0) {
                set_buffer_data_size(g_bound_buffers_arr[idx], size);
                pbo_shadow_alloc(g_bound_buffers_arr[idx], size, data);
            }
        }
    }
    CHECK_GL_ERROR
}

// ============================================================================
// Texture Buffer Objects (ES 3.2 native)
// ============================================================================
//
// TBO attaches a buffer object's data store to a texture buffer target.
// Virtual buffer IDs are resolved to real GPU buffer IDs via the buffer map.
// On ES 3.2, glTexBuffer/glTexBufferRange are natively supported.
//
// Safety: runtime function-pointer check — if the driver does not expose
// glTexBuffer (e.g. incomplete ES 3.2 implementation), we log an error
// and return gracefully rather than crashing.
// ============================================================================

// Checked once at first call; static const avoids per-call branch overhead.
static bool tbo_available() {
    static const bool avail = [] {
        if (GLES.glTexBuffer == nullptr) {
            LOG_E("glTexBuffer is not available on this driver — TBO unsupported");
            return false;
        }
        return true;
    }();
    return avail;
}

// Resolve virtual buffer → real GPU buffer in a single lookup (avoids separate
// has_buffer + find_real_buffer calls). Returns the real buffer, allocating one
// on-demand if the virtual buffer is tracked but not yet backed by a GPU object.
static inline GLuint resolve_tbo_buffer(GLuint buffer) {
    if (buffer == 0) return 0;
    auto [rb, exists] = find_real_buffer_with_exists(buffer);
    if (!exists) return buffer; // not tracked, pass through
    if (rb != 0) return rb;     // already resolved
    // Tracked but not yet allocated → create GPU backing
    GLuint new_rb = 0;
    GLES.glGenBuffers(1, &new_rb);
    modify_buffer_direct(buffer, new_rb);
    return new_rb;
}

void glTexBuffer(GLenum target, GLenum internalformat, GLuint buffer) {
    LOG()
    LOG_D("glTexBuffer, target = %s, internalformat = %s, buffer = %d", glEnumToString(target),
          glEnumToString(internalformat), buffer)
    if (target != GL_TEXTURE_BUFFER) return;
    if (!tbo_available()) [[unlikely]] return;

    GLuint real_buffer = resolve_tbo_buffer(buffer);
    GLES.glTexBuffer(target, internalformat, real_buffer);
    CHECK_GL_ERROR
}

void glTexBufferRange(GLenum target, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size) {
    LOG()
    LOG_D("glTexBufferRange, target = %s, internalformat = %s, buffer = %d, offset = %p, size = %zi",
          glEnumToString(target), glEnumToString(internalformat), buffer, (void*)offset, size)
    if (target != GL_TEXTURE_BUFFER) return;
    if (!tbo_available()) [[unlikely]] return;

    GLuint real_buffer = resolve_tbo_buffer(buffer);
    GLES.glTexBufferRange(target, internalformat, real_buffer, offset, size);
    CHECK_GL_ERROR
}

// ============================================================================
// Vertex Array Object Lifecycle (ES 3.2 native, with VAO map)
// ============================================================================

void glGenVertexArrays(GLsizei n, GLuint* arrays) {
    LOG()
    LOG_D("glGenVertexArrays(%i, %p)", n, arrays)
    for (int i = 0; i < n; ++i) {
        arrays[i] = gen_array();
    }
}

void glDeleteVertexArrays(GLsizei n, const GLuint* arrays) {
    LOG()
    LOG_D("glDeleteVertexArrays(%i, %p)", n, arrays)
    for (int i = 0; i < n; ++i) {
        if (has_array(arrays[i]) && arrays[i] != 0) {
            GLuint real_array = find_real_array(arrays[i]);
            if (real_array) {
                GLES.glDeleteVertexArrays(1, &real_array);
                CHECK_GL_ERROR
            }
        }
        remove_array(arrays[i]);
    }
}

GLboolean glIsVertexArray(GLuint array) {
    LOG()
    LOG_D("glIsVertexArray(%d)", array)
    return has_array(array);
}

void glBindVertexArray(GLuint array) {
    LOG()
    LOG_D("glBindVertexArray(%d)", array)

    bound_array = array;

    // update bound ibo
    set_bound_buffer_by_target(GL_ELEMENT_ARRAY_BUFFER, get_ibo_by_vao(array));

    if (!has_array(array) || array == 0) [[unlikely]] {
        LOG_D("Does not have va=%d found!", array)
        GLES.glBindVertexArray(array);
        CHECK_GL_ERROR
        return;
    }

    GLuint real_array = find_real_array(array);
    if (!real_array) {
        LOG_D("va=%d not initialized, initializing...", array)
        GLES.glGenVertexArrays(1, &real_array);
        modify_array(array, real_array);
        CHECK_GL_ERROR
    }
    LOG_D("glBindVertexArray: %d -> %d", array, real_array)
    GLES.glBindVertexArray(real_array);
    CHECK_GL_ERROR
}

// ============================================================================
// ARB Aliases
// ============================================================================

#if GLOBAL_DEBUG || DEBUG
#include <fstream>
#define BIN_FILE_PREFIX "/sdcard/MG/buf/"
#endif

extern "C"
{
    GLAPI GLAPIENTRY void* glMapBufferARB(GLenum target, GLenum access) __attribute__((alias("glMapBuffer")));
    GLAPI GLAPIENTRY void glBufferDataARB(GLenum target, GLsizeiptr size, const void* data, GLenum usage)
        __attribute__((alias("glBufferData")));
    GLAPI GLAPIENTRY GLboolean glUnmapBufferARB(GLenum target) __attribute__((alias("glUnmapBuffer")));
    GLAPI GLAPIENTRY void glBufferStorageARB(GLenum target, GLsizeiptr size, const void* data, GLbitfield flags)
        __attribute__((alias("glBufferStorage")));
    GLAPI GLAPIENTRY void glBindBufferARB(GLenum target, GLuint buffer) __attribute__((alias("glBindBuffer")));
}