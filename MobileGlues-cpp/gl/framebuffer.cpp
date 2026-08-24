// MobileGlues - gl/framebuffer.cpp
// Framebuffer state management: FBO ID mapping, attachment tracking,
// draw buffer remapping, and temp FBO pool for format conversion.
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
#include "texture.h"
#include <GLES3/gl32.h>
#include <cstring>
#include <vector>

#define DEBUG 0

// ---------------------------------------------------------------------------
// Float color-format → required color_buffer extension mapping.
//
// GLES 3.2 core only guarantees color-renderability for a small set of
// formats. Float/half-float internalformats need GL_EXT_color_buffer_float
// (RGBA32F, R11F_G11F_B10F, ...) or GL_EXT_color_buffer_half_float
// (RGBA16F, RG16F, ...) to be usable as FBO color attachments; without
// them glCheckFramebufferStatus returns GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT
// even though desktop GL accepts these as core since GL 3.0.
//
// Returns true when `internalFormat` is a float format whose required
// extension is NOT present on the underlying GLES driver.
// ---------------------------------------------------------------------------
struct FloatFormatExt {
    GLenum internalFormat;
    int   gles_caps_t::*ext;
};
static const FloatFormatExt kFloatFormatExts[] = {
    {GL_RGBA16F,          &gles_caps_t::GL_EXT_color_buffer_half_float},
    {GL_RG16F,            &gles_caps_t::GL_EXT_color_buffer_half_float},
    {GL_R16F,             &gles_caps_t::GL_EXT_color_buffer_half_float},
    {GL_RGB16F,           &gles_caps_t::GL_EXT_color_buffer_half_float},
    {GL_RGBA32F,          &gles_caps_t::GL_EXT_color_buffer_float},
    {GL_RGB32F,           &gles_caps_t::GL_EXT_color_buffer_float},
    {GL_RG32F,            &gles_caps_t::GL_EXT_color_buffer_float},
    {GL_R32F,             &gles_caps_t::GL_EXT_color_buffer_float},
    {GL_R11F_G11F_B10F,   &gles_caps_t::GL_EXT_color_buffer_float},
    {GL_RGB9_E5,          &gles_caps_t::GL_EXT_color_buffer_float},
};
static bool float_color_format_needs_missing_ext(GLenum internalFormat) {
    for (const auto& e : kFloatFormatExts) {
        if (e.internalFormat == internalFormat) return !(g_gles_caps.*(e.ext));
    }
    return false;
}

// ---------------------------------------------------------------------------
// Attachment tracking helpers.
//
// glFramebufferTexture2D / glFramebufferTexture / glFramebufferTextureLayer
// all need to record {FBO, attachment point, texture ID} so that
// glCheckFramebufferStatus can inspect the attached texture's internalformat
// when GLES reports INCOMPLETE_ATTACHMENT.
// ---------------------------------------------------------------------------
static GLuint current_fbo_for_target(GLenum target) {
    return (target == GL_FRAMEBUFFER || target == GL_DRAW_FRAMEBUFFER)
           ? GLState.framebuffer.drawFBO
           : GLState.framebuffer.readFBO;
}

static void track_fbo_attachment(GLenum target, GLenum attachment, GLuint texture) {
    GLuint fbo = current_fbo_for_target(target);
    if (!fbo) return;
    auto &att = GLState.framebuffer.attachments[fbo][attachment];
    att.fbo = fbo;
    att.attachment = attachment;
    att.texture = texture;
}

// Query GLES for an attachment's component type; returns GL_NONE if the
// attachment is empty or the query fails. Used by glCheckFramebufferStatus
// to confirm float attachments that bypassed our tracking (DSA paths).
static GLenum query_attachment_component_type(GLenum target, GLenum attachment) {
    GLint objType = GL_NONE;
    GLES.glGetFramebufferAttachmentParameteriv(
        target, attachment, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &objType);
    if (objType != GL_TEXTURE && objType != GL_RENDERBUFFER) return GL_NONE;
    GLint compType = GL_NONE;
    GLES.glGetFramebufferAttachmentParameteriv(
        target, attachment, GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE, &compType);
    return (GLenum)compType;
}

// ---------------------------------------------------------------------------
// Temporary FBO pool for hot-path texture operations
// (acquireTempFBO / releaseTempFBO — used by texture.cpp)
// ---------------------------------------------------------------------------

static std::vector<GLuint> g_tempFboPool;
static size_t g_tempFboPoolIndex = 0;

// ============================================================================
// FBO ID management
// ============================================================================

extern "C" GLAPI GLAPIENTRY void glGenFramebuffers(GLsizei n, GLuint *framebuffers) {
    LOG()
    GLES.glGenFramebuffers(n, framebuffers);

    auto &fb = GLState.framebuffer;
    for (GLsizei i = 0; i < n; i++) {
        fb.fboMap[framebuffers[i]] = framebuffers[i];
        fb.fboMapReverse[framebuffers[i]] = framebuffers[i];
    }
}

extern "C" GLAPI GLAPIENTRY void glDeleteFramebuffers(GLsizei n, const GLuint *framebuffers) {
    LOG()
    GLES.glDeleteFramebuffers(n, framebuffers);

    auto &fb = GLState.framebuffer;
    for (GLsizei i = 0; i < n; i++) {
        GLuint id = framebuffers[i];
        if (id == 0) continue;

        if (fb.drawFBO == id) fb.drawFBO = 0;
        if (fb.readFBO == id) fb.readFBO = 0;

        fb.fboMap.erase(id);
        fb.fboMapReverse.erase(id);
        fb.attachments.erase(id);
    }
}

// ============================================================================
// glBindFramebuffer
// ============================================================================

extern "C" GLAPI GLAPIENTRY void glBindFramebuffer(GLenum target, GLuint framebuffer) {
    LOG()
    GLES.glBindFramebuffer(target, framebuffer);

    // Hot path: most callers pass GL_FRAMEBUFFER (0x8D40), which binds to
    // both draw and read targets. Restructured so the common case is a
    // single branch instead of two OR-checks that each re-test the same
    // value. The DRAW/READ-only splits are cold by comparison.
    auto &fb = GLState.framebuffer;
    if (target == GL_FRAMEBUFFER) [[likely]] {
        fb.drawFBO = framebuffer;
        fb.readFBO = framebuffer;
        GLState.currentDrawFBO = framebuffer;
    } else if (target == GL_DRAW_FRAMEBUFFER) {
        fb.drawFBO = framebuffer;
        GLState.currentDrawFBO = framebuffer;
    } else if (target == GL_READ_FRAMEBUFFER) {
        fb.readFBO = framebuffer;
    }

    STATE_LOG("glBindFramebuffer: target=0x%X, fbo=%u", target, framebuffer);
}

// ============================================================================
// glFramebufferTexture2D / glFramebufferTexture / glFramebufferTextureLayer
// ============================================================================

extern "C" GLAPI GLAPIENTRY void glFramebufferTexture2D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level) {
    LOG()
    GLenum esTexTarget = GLStateManager::ConvertTextureTarget(textarget);
    GLES.glFramebufferTexture2D(target, attachment, esTexTarget, texture, level);
    track_fbo_attachment(target, attachment, texture);
}

// glFramebufferTexture (no "2D" suffix) is an optional GLES function that
// requires GL_EXT_geometry_shader / GL_OES_geometry_shader. Many mobile GLES
// 3.2 drivers do NOT expose it (NULL function pointer), causing the
// attachment to silently fail → GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT.
// When the pointer is NULL, emulate via glFramebufferTexture2D (2D/cube) or
// glFramebufferTextureLayer (2D-array/3D, layer 0).
extern "C" GLAPI GLAPIENTRY void glFramebufferTexture(GLenum target, GLenum attachment, GLuint texture, GLint level) {
    LOG()
    if (GLES.glFramebufferTexture != nullptr) {
        GLES.glFramebufferTexture(target, attachment, texture, level);
    } else if (texture == 0) {
        // Detach: glFramebufferTexture2D with texture=0 works everywhere.
        GLES.glFramebufferTexture2D(target, attachment, GL_TEXTURE_2D, 0, level);
    } else {
        TextureObject* tex = mgGetTexObjectByID(texture);
        GLenum texTarget = tex ? ConvertTextureTargetToGLEnum(tex->target) : GL_NONE;
        if (texTarget == GL_TEXTURE_2D) {
            GLES.glFramebufferTexture2D(target, attachment, GL_TEXTURE_2D, texture, level);
        } else if (texTarget == GL_TEXTURE_CUBE_MAP) {
            // glFramebufferTexture2D needs a face target for cube maps;
            // use +X as the default face (desktop GL attaches layer 0).
            GLES.glFramebufferTexture2D(target, attachment, GL_TEXTURE_CUBE_MAP_POSITIVE_X, texture, level);
        } else if (texTarget == GL_TEXTURE_2D_ARRAY || texTarget == GL_TEXTURE_3D) {
            // Layered textures: attach layer 0 via glFramebufferTextureLayer.
            GLES.glFramebufferTextureLayer(target, attachment, texture, level, 0);
        } else {
            // Unknown / untracked texture: best-effort fallback.
            LOG_W("glFramebufferTexture: texture %u untracked (target=0x%X), "
                  "falling back to glFramebufferTexture2D", texture, texTarget);
            GLES.glFramebufferTexture2D(target, attachment, GL_TEXTURE_2D, texture, level);
        }
    }
    track_fbo_attachment(target, attachment, texture);
}

extern "C" GLAPI GLAPIENTRY void glFramebufferTextureLayer(GLenum target, GLenum attachment, GLuint texture, GLint level, GLint layer) {
    GLES.glFramebufferTextureLayer(target, attachment, texture, level, layer);
    track_fbo_attachment(target, attachment, texture);
}

// ============================================================================
// glFramebufferRenderbuffer - native passthrough
// ============================================================================

extern "C" GLAPI GLAPIENTRY void glFramebufferRenderbuffer(GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer) {
    GLES.glFramebufferRenderbuffer(target, attachment, renderbuffertarget, renderbuffer);
}

// ============================================================================
// glCheckFramebufferStatus
//
// Three GLES incomplete statuses are promoted to COMPLETE to match desktop
// GL behaviour (the API MobileGlues translates *from*), so Iris shader packs
// keep rendering instead of disabling shaders:
//
//   UNSUPPORTED              — desktop GL accepts the FBO config (e.g.
//                              depth-stencil texture with sized internalformat).
//   INCOMPLETE_ATTACHMENT    — attachment exists but GLES considers it
//                              non-renderable (typically a float color format
//                              lacking GL_EXT_color_buffer_float/half_float).
//   INCOMPLETE_MISSING_ATT   — no attachment at all; on GLES this also fires
//                              when glFramebufferTexture (optional) silently
//                              failed, or no-attachment FBOs aren't supported.
//
// Other statuses (INCOMPLETE_DIMENSIONS, INCOMPLETE_MULTISAMPLE, UNDEFINED)
// indicate real configuration bugs and are returned as-is so callers can fall
// back (e.g. Xaero's World Map allocates a working FBO on real failure).
// ============================================================================

extern "C" GLAPI GLAPIENTRY GLenum glCheckFramebufferStatus(GLenum target) {
    LOG()
    GLenum status = GLES.glCheckFramebufferStatus(target);
    if (status == GL_FRAMEBUFFER_COMPLETE) return status;

    GLuint fbo = current_fbo_for_target(target);

    switch (status) {
        case GL_FRAMEBUFFER_UNSUPPORTED:
            LOG_D("glCheckFramebufferStatus: UNSUPPORTED -> COMPLETE (target=0x%X, FBO=%u)", target, fbo);
            return GL_FRAMEBUFFER_COMPLETE;

        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT: {
            // Diagnose whether a float attachment caused it (for logging).
            // Path (a): our attachment tracking (texture ID → internal_format).
            bool floatConfirmed = false;
            if (fbo) {
                auto fboIt = GLState.framebuffer.attachments.find(fbo);
                if (fboIt != GLState.framebuffer.attachments.end()) {
                    for (const auto& [attPoint, info] : fboIt->second) {
                        if (info.texture == 0) continue;
                        TextureObject* tex = mgGetTexObjectByID(info.texture);
                        if (tex && float_color_format_needs_missing_ext(tex->internal_format)) {
                            floatConfirmed = true;
                            LOG_D("glCheckFramebufferStatus: FBO %u att 0x%X float format %s "
                                  "without color_buffer ext", fbo, attPoint,
                                  glEnumToString(tex->internal_format));
                            break;
                        }
                    }
                }
            }
            // Path (b): direct GLES query (catches DSA-attached textures
            // that bypassed our tracking). Covers color + depth/stencil.
            if (!floatConfirmed) {
                static const GLenum kAllAttachments[] = {
                    GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2,
                    GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4, GL_COLOR_ATTACHMENT5,
                    GL_COLOR_ATTACHMENT6, GL_COLOR_ATTACHMENT7,
                    GL_DEPTH_ATTACHMENT, GL_STENCIL_ATTACHMENT, GL_DEPTH_STENCIL_ATTACHMENT,
                };
                for (GLenum att : kAllAttachments) {
                    if (query_attachment_component_type(target, att) == GL_FLOAT) {
                        floatConfirmed = true;
                        LOG_D("glCheckFramebufferStatus: FBO %u att 0x%X has GL_FLOAT "
                              "component type (GLES query)", fbo, att);
                        break;
                    }
                }
                GLES.glGetError();  // clear errors from the queries above
            }
            LOG_D("glCheckFramebufferStatus: INCOMPLETE_ATTACHMENT -> COMPLETE "
                  "(target=0x%X, FBO=%u, floatConfirmed=%d)", target, fbo, (int)floatConfirmed);
            return GL_FRAMEBUFFER_COMPLETE;
        }

        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            LOG_D("glCheckFramebufferStatus: INCOMPLETE_MISSING_ATTACHMENT -> COMPLETE "
                  "(target=0x%X, FBO=%u) — likely NULL glFramebufferTexture or "
                  "unsupported no-attachment FBO", target, fbo);
            return GL_FRAMEBUFFER_COMPLETE;

        default:
            LOG_E("glCheckFramebufferStatus: target=0x%X reports 0x%X (incomplete); "
                  "returning real status", target, status);
            return status;
    }
}

// ============================================================================
// glFramebufferParameteri - native passthrough
// ============================================================================

extern "C" GLAPI GLAPIENTRY void glFramebufferParameteri(GLenum target, GLenum pname, GLint param) {
    GLES.glFramebufferParameteri(target, pname, param);
}

// ============================================================================
// glGetFramebufferAttachmentParameteriv - native passthrough
// ============================================================================

extern "C" GLAPI GLAPIENTRY void glGetFramebufferAttachmentParameteriv(GLenum target, GLenum attachment, GLenum pname, GLint *params) {
    GLES.glGetFramebufferAttachmentParameteriv(target, attachment, pname, params);
}

// ============================================================================
// glGetFramebufferParameteriv - native passthrough
// ============================================================================

extern "C" GLAPI GLAPIENTRY void glGetFramebufferParameteriv(GLenum target, GLenum pname, GLint *params) {
    GLES.glGetFramebufferParameteriv(target, pname, params);
}

// ============================================================================
// glBlitFramebuffer - native passthrough
// ============================================================================

extern "C" GLAPI GLAPIENTRY void glBlitFramebuffer(GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1,
                       GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1,
                       GLbitfield mask, GLenum filter) {
    // Strip GL_ACCUM_BUFFER_BIT
    mask &= ~0x00000200;
    GLES.glBlitFramebuffer(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);
}

// ============================================================================
// glInvalidateFramebuffer / glInvalidateSubFramebuffer - native passthrough
// ============================================================================

extern "C" GLAPI GLAPIENTRY void glInvalidateFramebuffer(GLenum target, GLsizei numAttachments, const GLenum *attachments) {
    GLES.glInvalidateFramebuffer(target, numAttachments, attachments);
}

extern "C" GLAPI GLAPIENTRY void glInvalidateSubFramebuffer(GLenum target, GLsizei numAttachments, const GLenum *attachments,
                                GLint x, GLint y, GLsizei width, GLsizei height) {
    GLES.glInvalidateSubFramebuffer(target, numAttachments, attachments, x, y, width, height);
}

// ============================================================================
// Map initialization helpers (for backward compatibility)
// ============================================================================

void InitFramebufferMap(size_t expectedSize) {
    auto &fb = GLState.framebuffer;
    fb.fboMap.reserve(expectedSize);
    fb.fboMapReverse.reserve(expectedSize);
}

// ============================================================================
// Temp FBO pool — for hot-path texture operations (glCopyTexImage2D, etc.)
// ============================================================================

GLuint acquireTempFBO() {
    if (g_tempFboPoolIndex < g_tempFboPool.size()) {
        return g_tempFboPool[g_tempFboPoolIndex++];
    }
    GLuint fbo;
    GLES.glGenFramebuffers(1, &fbo);
    g_tempFboPool.push_back(fbo);
    g_tempFboPoolIndex++;
    return fbo;
}

void releaseTempFBO(GLuint fbo) {
    // Simple: just decrement index. FBO is not deleted, just returned to pool.
    if (g_tempFboPoolIndex > 0) g_tempFboPoolIndex--;
}

void releaseAllTempFBOs() {
    g_tempFboPoolIndex = 0;
}

void cleanupTempFBOs() {
    for (GLuint fbo : g_tempFboPool) {
        GLES.glDeleteFramebuffers(1, &fbo);
    }
    g_tempFboPool.clear();
    g_tempFboPoolIndex = 0;
}