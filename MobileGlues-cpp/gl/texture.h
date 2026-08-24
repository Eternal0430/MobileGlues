// MobileGlues - gl/texture.h
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

// ============================================================================
// Texture Module Header (OpenGL ES 3.2)
//
// Architecture rule: "ES 3.2 native → native, ES 3.2 not native → CPU simulation"
//
// Native (ES 3.2 directly supports):
//   glTexImage2D, glTexImage3D, glTexSubImage2D, glTexSubImage3D
//   glTexStorage2D, glTexStorage3D
//   glCompressedTexImage2D, glCompressedTexImage3D
//   glCompressedTexSubImage2D, glCompressedTexSubImage3D
//   glCopyTexImage2D, glCopyTexSubImage2D, glCopyTexSubImage3D
//   glGenerateMipmap, glGenerateTextureMipmap
//   glTexParameterf, glTexParameteri, glTexParameteriv, glTexParameterfv
//   glGetTexParameteriv, glGetTexParameterfv
//   glGetTexLevelParameteriv, glGetTexLevelParameterfv
//   glTexBuffer, glTexBufferRange
//   glBindTexture, glDeleteTextures, glActiveTexture
// ============================================================================

#ifndef MOBILEGLUES_TEXTURE_H
#define MOBILEGLUES_TEXTURE_H

#include <memory>
#include <vector>

#ifdef __cplusplus
extern "C"
{
#endif

#include <GL/gl.h>

    // ============================================================================
    // Texture binding / management
    // ============================================================================

    GLAPI GLAPIENTRY void glBindTexture(GLenum target, GLuint texture);
    GLAPI GLAPIENTRY void glDeleteTextures(GLsizei n, const GLuint* textures);
    GLAPI GLAPIENTRY void glActiveTexture(GLenum texture);

    // ============================================================================
    // Native texture image specification (ES 3.2)
    // ============================================================================

    GLAPI GLAPIENTRY void glTexImage2D(GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height,
                                       GLint border, GLenum format, GLenum type, const GLvoid* pixels);
    GLAPI GLAPIENTRY void glTexImage3D(GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height,
                                       GLsizei depth, GLint border, GLenum format, GLenum type, const GLvoid* pixels);
    GLAPI GLAPIENTRY void glTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width,
                                          GLsizei height, GLenum format, GLenum type, const void* pixels);
    GLAPI GLAPIENTRY void glTexSubImage3D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset,
                                          GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type,
                                          const void* pixels);
    GLAPI GLAPIENTRY void glTexStorage2D(GLenum target, GLsizei levels, GLenum internalFormat, GLsizei width,
                                         GLsizei height);
    GLAPI GLAPIENTRY void glTexStorage3D(GLenum target, GLsizei levels, GLenum internalFormat, GLsizei width,
                                         GLsizei height, GLsizei depth);

    // ============================================================================
    // Native compressed texture functions (ES 3.2)
    // ============================================================================

    GLAPI GLAPIENTRY void glCompressedTexImage2D(GLenum target, GLint level, GLenum internalformat, GLsizei width,
                                                 GLsizei height, GLint border, GLsizei imageSize, const void* data);
    GLAPI GLAPIENTRY void glCompressedTexImage3D(GLenum target, GLint level, GLenum internalformat, GLsizei width,
                                                 GLsizei height, GLsizei depth, GLint border, GLsizei imageSize,
                                                 const void* data);
    GLAPI GLAPIENTRY void glCompressedTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset,
                                                    GLsizei width, GLsizei height, GLenum format, GLsizei imageSize,
                                                    const void* data);
    GLAPI GLAPIENTRY void glCompressedTexSubImage3D(GLenum target, GLint level, GLint xoffset, GLint yoffset,
                                                    GLint zoffset, GLsizei width, GLsizei height, GLsizei depth,
                                                    GLenum format, GLsizei imageSize, const void* data);

    // ============================================================================
    // Native copy texture functions (ES 3.2)
    // ============================================================================

    GLAPI GLAPIENTRY void glCopyTexImage2D(GLenum target, GLint level, GLenum internalFormat, GLint x, GLint y,
                                           GLsizei width, GLsizei height, GLint border);
    GLAPI GLAPIENTRY void glCopyTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x,
                                              GLint y, GLsizei width, GLsizei height);
    GLAPI GLAPIENTRY void glCopyTexSubImage3D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset,
                                              GLint x, GLint y, GLsizei width, GLsizei height);

    // ============================================================================
    // Native mipmap generation (ES 3.2)
    // ============================================================================

    GLAPI GLAPIENTRY void glGenerateMipmap(GLenum target);
    GLAPI GLAPIENTRY void glGenerateTextureMipmap(GLuint texture);

    // ============================================================================
    // Native texture parameter functions (ES 3.2)
    // ============================================================================

    GLAPI GLAPIENTRY void glTexParameterf(GLenum target, GLenum pname, GLfloat param);
    GLAPI GLAPIENTRY void glTexParameteri(GLenum target, GLenum pname, GLint param);
    GLAPI GLAPIENTRY void glTexParameteriv(GLenum target, GLenum pname, const GLint* params);
    GLAPI GLAPIENTRY void glTexParameterfv(GLenum target, GLenum pname, const GLfloat* params);

    // ============================================================================
    // Native texture query functions (ES 3.2)
    // ============================================================================

    GLAPI GLAPIENTRY void glGetTexParameteriv(GLenum target, GLenum pname, GLint* params);
    GLAPI GLAPIENTRY void glGetTexParameterfv(GLenum target, GLenum pname, GLfloat* params);
    GLAPI GLAPIENTRY void glGetTexLevelParameterfv(GLenum target, GLint level, GLenum pname, GLfloat* params);
    GLAPI GLAPIENTRY void glGetTexLevelParameteriv(GLenum target, GLint level, GLenum pname, GLint* params);

    // ============================================================================
    // Native texture buffer functions (ES 3.2)
    // ============================================================================

    GLAPI GLAPIENTRY void glTexBuffer(GLenum target, GLenum internalformat, GLuint buffer);
    GLAPI GLAPIENTRY void glTexBufferRange(GLenum target, GLenum internalformat, GLuint buffer, GLintptr offset,
                                           GLsizeiptr size);

    // ============================================================================
    // Renderbuffer functions (native)
    // ============================================================================

    GLAPI GLAPIENTRY void glRenderbufferStorage(GLenum target, GLenum internalFormat, GLsizei width, GLsizei height);
    GLAPI GLAPIENTRY void glRenderbufferStorageMultisample(GLenum target, GLsizei samples, GLenum internalFormat,
                                                           GLsizei width, GLsizei height);

    // ============================================================================
    // Other functions (keep existing logic)
    // ============================================================================

    GLAPI GLAPIENTRY void glGetTexImage(GLenum target, GLint level, GLenum format, GLenum type, void* pixels);
    GLAPI GLAPIENTRY void glReadPixels(GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type,
                                       void* pixels);
    GLAPI GLAPIENTRY void glClearTexImage(GLuint texture, GLint level, GLenum format, GLenum type, const void* data);
    GLAPI GLAPIENTRY void glPixelStorei(GLenum pname, GLint param);

#ifdef __cplusplus
}
#endif

enum class TextureTarget : unsigned int {
    TEXTURE_1D = 0,
    PROXY_TEXTURE_1D,
    TEXTURE_1D_ARRAY,
    PROXY_TEXTURE_1D_ARRAY,
    TEXTURE_2D,
    PROXY_TEXTURE_2D,
    TEXTURE_2D_ARRAY,
    PROXY_TEXTURE_2D_ARRAY,
    TEXTURE_2D_MULTISAMPLE,
    PROXY_TEXTURE_2D_MULTISAMPLE,
    TEXTURE_2D_MULTISAMPLE_ARRAY,
    PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY,
    TEXTURE_3D,
    PROXY_TEXTURE_3D,
    TEXTURE_RECTANGLE,
    PROXY_TEXTURE_RECTANGLE,
    TEXTURE_CUBE_MAP,
    PROXY_TEXTURE_CUBE_MAP,
    // TEXTURE_CUBE_MAP_POSITIVE_X,
    // TEXTURE_CUBE_MAP_NEGATIVE_X,
    // TEXTURE_CUBE_MAP_POSITIVE_Y,
    // TEXTURE_CUBE_MAP_NEGATIVE_Y,
    // TEXTURE_CUBE_MAP_POSITIVE_Z,
    // TEXTURE_CUBE_MAP_NEGATIVE_Z,
    TEXTURE_CUBE_MAP_ARRAY,
    PROXY_TEXTURE_CUBE_MAP_ARRAY,
    TEXTURE_BUFFER,
    TEXTURES_COUNT,
    UNKNWON
};

GLenum ConvertTextureTargetToGLEnum(TextureTarget target);
TextureTarget ConvertGLEnumToTextureTarget(GLenum target);

class TextureObject { // TODO: Make this a more standard class
public:
    TextureTarget target;
    GLuint texture;
    GLenum internal_format;
    GLenum format;
    GLint swizzle_param[4];
    GLsizei width;
    GLsizei height;
    GLsizei depth;
    // Tracks whether glGenerateMipmap has been called for this texture since
    // the last glTexImage*(level=0) reset. When false, mipmap min-filters
    // (GL_*_MIPMAP_*) are downgraded to their non-mipmap counterparts in
    // glTexParameteri, because GLES returns (0,0,0,1) when sampling an
    // incomplete mipmap chain whereas desktop GL gracefully falls back to
    // the base level.
    bool hasMipmaps = false;
    // Tracks whether this texture has had a CPU-side BGRA->RGBA swizzle
    // applied during pixel upload. When true, the GLES-side texture stores
    // correct RGBA data, so any GL_TEXTURE_SWIZZLE_R/B swap requested by the
    // application (e.g. Xaero setting R=BLUE, B=RED on desktop GL to match
    // its BGRA uploads) must NOT be forwarded to GLES - doing so would
    // re-swap the already-correct channels and produce a blue image.
    bool bgraCpuSwizzled = false;
    // Reverse mapping: set of TextureBindingSlot pointers (as uintptr_t) that
    // reference this object. Used by MarkTextureObjectForDeletion to avoid
    // iterating all 32 units × 24 targets (768 slots).
    // Small N (typically 1-3): vector with linear scan beats unordered_set.
    std::vector<uintptr_t> binding_slots;
};

TextureObject* mgGetTexObjectByTarget(GLenum target);
TextureObject* mgGetTexObjectByID(unsigned texture);
int GetCurrentTextureUnitIndex();
void InitTextureMap(size_t expectedSize);

#endif