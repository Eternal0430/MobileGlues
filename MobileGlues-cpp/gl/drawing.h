// MobileGlues - gl/drawing.h
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

// ============================================================================
// Drawing Module Header (OpenGL ES 3.2)
//
// Architecture rule: "ES 3.2 native → native, ES 3.2 not native → CPU simulation"
//
// Native draw calls (ES 3.2 directly supports):
//   glDrawElements, glDrawElementsInstanced, glDrawElementsBaseVertex
//   glDrawArrays, glDrawArraysInstanced, glDrawRangeElements
//   glDrawArraysIndirect, glDrawElementsIndirect, glClear
// ============================================================================

#ifndef MOBILEGLUES_DRAWING_H
#define MOBILEGLUES_DRAWING_H

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <GLES3/gl32.h>
#include "../includes.h"
#include <GL/gl.h>
#include "glcorearb.h"
#include "log.h"
#include "../gles/loader.h"
#include "mg.h"
#include "state.h"  // for MAX_TEXTURE_UNITS

struct SamplerInfo {
    GLint locWidth;
    GLint locHeight;
    std::vector<GLint> samplers;
};

// Texture binding tracking per unit to avoid glGetIntegerv GPU queries.
// Updated by glBindTexture (texture.cpp), read by setupBufferTextureUniforms (drawing.cpp).
// Use MAX_TEXTURE_UNITS from state.h for consistency.
extern GLuint g_tracked_tex2d_binding[MAX_TEXTURE_UNITS];
extern GLuint g_tracked_tex_cube_binding[MAX_TEXTURE_UNITS];
extern GLuint g_tracked_tex_2d_array_binding[MAX_TEXTURE_UNITS];
extern GLuint g_tracked_tex_3d_binding[MAX_TEXTURE_UNITS];
extern GLuint g_tracked_tex_2d_ms_binding[MAX_TEXTURE_UNITS];
extern GLuint g_tracked_tex_2d_ms_array_binding[MAX_TEXTURE_UNITS];
extern GLuint g_tracked_tex_cube_array_binding[MAX_TEXTURE_UNITS];
extern GLuint g_tracked_tex_rect_binding[MAX_TEXTURE_UNITS];

#ifdef __cplusplus
extern "C"
{
#endif

    // ============================================================================
    // Native draw calls (ES 3.2)
    // ============================================================================

    GLAPI GLAPIENTRY void glDrawElements(GLenum mode, GLsizei count, GLenum type, const void* indices);
    GLAPI GLAPIENTRY void glDrawElementsInstanced(GLenum mode, GLsizei count, GLenum type, const void* indices,
                                                  GLsizei primcount);
    GLAPI GLAPIENTRY void glDrawElementsBaseVertex(GLenum mode, GLsizei count, GLenum type, const void* indices,
                                                   GLint basevertex);
    GLAPI GLAPIENTRY void glDrawArrays(GLenum mode, GLint first, GLsizei count);
    GLAPI GLAPIENTRY void glDrawArraysInstanced(GLenum mode, GLint first, GLsizei count, GLsizei instancecount);
    GLAPI GLAPIENTRY void glDrawRangeElements(GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type,
                                              const void* indices);
    GLAPI GLAPIENTRY void glDrawArraysIndirect(GLenum mode, const void* indirect);
    GLAPI GLAPIENTRY void glDrawElementsIndirect(GLenum mode, GLenum type, const void* indirect);
    GLAPI GLAPIENTRY void glClear(GLbitfield mask);

    // ============================================================================
    // Other functions (keep existing logic)
    // ============================================================================

    GLAPI GLAPIENTRY void glBindImageTexture(GLuint unit, GLuint texture, GLint level, GLboolean layered, GLint layer,
                                             GLenum access, GLenum format);
    GLAPI GLAPIENTRY void glDispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z);
    GLAPI GLAPIENTRY void glMemoryBarrier(GLbitfield barriers);
    GLAPI GLAPIENTRY void glUniform1i(GLint location, GLint v0);

#ifdef __cplusplus
}
#endif

#endif // MOBILEGLUES_DRAWING_H