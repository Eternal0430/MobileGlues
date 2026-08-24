// MobileGlues - gl/mg.h
// State manager header: backward-compatible wrappers around GLStateManager.
// All old extern declarations and macros are mapped to GLStateManager methods.
//
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

#pragma once

#include "state.h"
#include <EGL/egl.h>
#include "../gles/gles.h"
#include "glext.h"

// Include the loader handle extern (defined in gles/loader.cpp)
extern void* g_loader_handle;

// ============================================================================
// Per-context scalar GL state (ported from upstream mg.h for egl/context.h)
//
// Upstream moved this state INTO the per-context MGContext record
// (`gl_state_s gl;` member). This fork still drives its global GLStateManager,
// so the struct is defined here -- BEFORE the legacy #define aliases below,
// whose bodies would otherwise rewrite these snake_case member names.
//
// NOTE: when the port of egl/context.cpp resumes it also needs the
// thread_local `gl_state` POINTER from upstream mg.h; that name currently
// collides with the `#define gl_state (&GLState)` alias below and must be
// reconciled as part of that port.
// ============================================================================
struct gl_state_s {
    GLsizei proxy_width;
    GLsizei proxy_height;
    GLenum proxy_intformat;

    GLuint current_program;
    GLuint current_tex_unit;
    GLuint current_draw_fbo;

    // The six pixel-store parameters desktop GL has and GLES does not (see
    // upstream mg.h). GLES answers GL_INVALID_ENUM for all six in both
    // directions; they are tracked here so they can be set and read back.
    GLint unpack_swap_bytes;
    GLint unpack_lsb_first;
    GLint pack_swap_bytes;
    GLint pack_lsb_first;
    GLint pack_image_height;
    GLint pack_skip_images;

    // Default constructor matching the zero-initialised fallback state.
    gl_state_s()
        : proxy_width(0), proxy_height(0), proxy_intformat(0),
          current_program(0), current_tex_unit(0), current_draw_fbo(0),
          unpack_swap_bytes(0), unpack_lsb_first(0),
          pack_swap_bytes(0), pack_lsb_first(0),
          pack_image_height(0), pack_skip_images(0) {}
};
typedef struct gl_state_s* gl_state_t;

// ============================================================================
// Backward-compatible #define macros for old code
// ============================================================================
// NOTE: Removed legacy aliases that had no remaining callers in the codebase:
//   current_program, current_tex_unit, is_draw_call, buffer_map*,
//   vao_map*, texture_map*, fbo_map*, shader_map*, program_map*,
//   buffer_info, getReal*/getVirtual* inline helpers, FUNC_GL_STATE_* /
//   GET_GL_STATE_* setter/getter macros. The retained aliases below are
//   still used by texture.cpp / getter.cpp / ExtWrappers / framebuffer.h.

// Old state variable aliases (pointer-style for backward compat)
#define gl_state         (&GLState)
#define hardware         (&GLState)
#define proxy_width      proxyWidth
#define proxy_height     proxyHeight
#define proxy_intformat  proxyInternalFormat
#define current_draw_fbo GLState.currentDrawFBO
#define current_read_fbo GLState.framebuffer.readFBO

// Old format conversion aliases (inline functions to avoid macro collisions)
inline GLenum CheckTextureTarget(GLenum target) { return GLStateManager::ConvertTextureTarget(target); }
inline GLenum CheckInternalFormat(GLenum format) { return GLStateManager::ConvertInternalFormat(format); }
inline GLenum CheckFormat(GLenum format) { return GLStateManager::ConvertFormat(format); }
inline GLenum CheckType(GLenum type) { return GLStateManager::ConvertType(type); }
inline bool mglIsDepthStencil(GLenum format) { return GLStateManager::IsDepthStencilFormat(format); }
inline bool mglIsCompressed(GLenum format) { return GLStateManager::IsCompressedFormat(format); }
inline GLenum TextureBindingTarget(GLenum target) { return GLStateManager::GetTextureBindingTarget(target); }
inline GLenum targetToBinding(GLenum target) { return GLStateManager::TargetToBindingTarget(target); }

// Old state setter functions (called from texture.cpp, buffer.cpp, etc.)
inline void set_gl_state_proxy_width(GLsizei value) { GLState.proxyWidth = value; }
inline void set_gl_state_proxy_height(GLsizei value) { GLState.proxyHeight = value; }
inline void set_gl_state_proxy_intformat(GLenum value) { GLState.proxyInternalFormat = value; }
inline void set_gl_state_current_program(GLuint value) { GLState.currentProgram = value; }
inline void set_gl_state_current_tex_unit(GLuint value) { GLState.currentTexUnit = value; }
inline void set_gl_state_current_draw_fbo(GLuint value) { GLState.currentDrawFBO = value; }

// Legacy helpers from original mg.cpp (called from texture.cpp)
inline GLenum pname_convert(GLenum pname) {
    switch (pname) {
    case GL_TEXTURE_LOD_BIAS:
        return GL_TEXTURE_LOD_BIAS_QCOM;
    }
    return pname;
}

inline GLenum map_tex_target(GLenum target) {
    switch (target) {
    case GL_TEXTURE_1D:
    case GL_TEXTURE_RECTANGLE_ARB:
        return GL_TEXTURE_2D;
    case GL_PROXY_TEXTURE_1D:
    case GL_PROXY_TEXTURE_RECTANGLE_ARB:
        return GL_PROXY_TEXTURE_2D;
    default:
        return target;
    }
}

// Old state setter/getter macros (FUNC_GL_STATE_*, GET_GL_STATE_*) removed:
// no remaining callers; direct field access is used everywhere instead.

// ============================================================================
// New: RenderState version accessor (for backend dirty-tracking)
// ============================================================================

inline uint32_t GetRenderStateVersion() { return GLState.renderState.GetVersion(); }
inline const RenderStateParameters& GetRenderStateParameters() { return GLState.renderState.GetAllParameters(); }

// ============================================================================
// New: ErrorState convenience accessors
// ============================================================================

inline void RecordGLError(ErrorCode code, const char* message) {
    GLState.errorState.RecordError(code, std::make_unique<GenericErrorInfo>(message));
}

inline bool HasGLError() { return GLState.errorState.HasGLError(); }
inline void ClearGLErrors() { GLState.errorState.Clear(); }

// ============================================================================
// Global texture unit count (queried from GLES)
// ============================================================================

extern int MG_MAX_COMBINED_TEXTURE_IMAGE_UNITS;