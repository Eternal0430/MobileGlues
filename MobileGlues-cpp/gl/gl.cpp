// MobileGlues - gl/gl.cpp
// Core GL state wrappers: glClearDepth, glClear, glHint, glDrawBuffer, glReadBuffer.
// NOTE: glDepthRange, glPolygonMode, glPointSize are in gl_stub.cpp
//       glViewport is in FSR1/FSR1.cpp
//       glGenTextures is in gl_native.cpp
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
#include "framebuffer.h"
#include "../gles/loader.h"
#include "mg.h"
#include <GLES3/gl32.h>

#define DEBUG 0

// ============================================================================
// glClearDepth - desktop uses double, ES uses float
// ============================================================================

extern "C" GLAPI GLAPIENTRY void glClearDepth(GLclampd depth) {
    GLES.glClearDepthf((float)depth);
    GLState.legacy.clearDepth = (GLfloat)depth;
}

// ============================================================================
// glClear - handles legacy clear mask conversion
// ============================================================================

extern "C" GLAPI GLAPIENTRY void glClear(GLbitfield mask) {
    // GL_ACCUM_BUFFER_BIT is not supported in ES, strip it
    mask &= ~0x00000200; // GL_ACCUM_BUFFER_BIT
    GLES.glClear(mask);
}

// ============================================================================
// glHint - pass through (ES supports basic hints)
// ============================================================================

extern "C" GLAPI GLAPIENTRY void glHint(GLenum target, GLenum mode) {
    GLES.glHint(target, mode);
}

// ============================================================================
// glDrawBuffer - maps to glDrawBuffers in ES
// ============================================================================

extern "C" GLAPI GLAPIENTRY void glDrawBuffer(GLenum mode) {
    if (mode == GL_NONE) {
        GLenum none = GL_NONE;
        GLES.glDrawBuffers(1, &none);
        GLState.framebuffer.drawBuffers[0] = GL_NONE;
        GLState.framebuffer.drawBufferCount = 1;
    } else if (mode >= GL_COLOR_ATTACHMENT0 && mode <= GL_COLOR_ATTACHMENT0 + 31) {
        GLES.glDrawBuffers(1, &mode);
        GLState.framebuffer.drawBuffers[0] = mode;
        GLState.framebuffer.drawBufferCount = 1;
    } else if (mode == GL_BACK || mode == GL_FRONT || mode == GL_FRONT_AND_BACK ||
               mode == GL_LEFT || mode == GL_RIGHT || mode == GL_BACK_LEFT ||
               mode == GL_FRONT_LEFT || mode == GL_BACK_RIGHT || mode == GL_FRONT_RIGHT ||
               mode == GL_AUX0 || mode == GL_AUX1 || mode == GL_AUX2 || mode == GL_AUX3) {
        // Map desktop GL default-framebuffer tokens to the GLES-legal value.
        // On the default framebuffer (drawFBO == 0) GLES only accepts GL_BACK
        // (or GL_NONE). On a user FBO (drawFBO != 0) GLES rejects GL_BACK /
        // GL_FRONT entirely and only accepts GL_COLOR_ATTACHMENTi - passing
        // GL_BACK silently no-ops the call and leaves the previous draw
        // buffer (possibly GL_NONE from a depth-only pre-pass) active, which
        // manifests as a black screen because no color attachment is written.
        GLenum mapped;
        if (GLState.framebuffer.drawFBO != 0) {
            mapped = GL_COLOR_ATTACHMENT0;
        } else {
            mapped = GL_BACK;
        }
        GLES.glDrawBuffers(1, &mapped);
        GLState.framebuffer.drawBuffers[0] = mapped;
        GLState.framebuffer.drawBufferCount = 1;
    } else {
        LOG_D("glDrawBuffer: unmapped mode 0x%X (drawFBO=%u) - ignoring", mode, GLState.framebuffer.drawFBO);
    }
}

// ============================================================================
// glReadBuffer - track read buffer state
// ============================================================================

extern "C" GLAPI GLAPIENTRY void glReadBuffer(GLenum mode) {
    // GLES 3.2 restricts the legal values: GL_NONE / GL_BACK on the default
    // framebuffer, GL_NONE / GL_COLOR_ATTACHMENTi on a user FBO. Desktop GL
    // additionally allows GL_FRONT, GL_LEFT, GL_RIGHT, GL_AUX0..3, etc.
    // Passing desktop-only tokens straight to GLES triggers GL_INVALID_ENUM,
    // the driver keeps the previous read buffer (which may be GL_NONE) and
    // the wrapper cache (GLState.framebuffer.readBuffer) gets out of sync
    // with the real GL state. The mismatch then breaks glBlitFramebuffer
    // reads (source FBO has no readable color buffer) -> black screen.
    GLenum mapped;
    if (mode == GL_NONE) {
        mapped = GL_NONE;
    } else if (mode >= GL_COLOR_ATTACHMENT0 && mode <= GL_COLOR_ATTACHMENT0 + 31) {
        // Color attachment is only valid on a user FBO. If the caller passed
        // it while on the default framebuffer (which can happen with legacy
        // desktop GL code paths that assume a single color attachment),
        // remap to GL_BACK.
        mapped = (GLState.framebuffer.readFBO != 0) ? mode : GL_BACK;
    } else if (mode == GL_BACK) {
        // GL_BACK is only valid on the default framebuffer in GLES.
        mapped = (GLState.framebuffer.readFBO != 0) ? GL_COLOR_ATTACHMENT0 : GL_BACK;
    } else {
        // Desktop-only tokens: GL_FRONT, GL_LEFT, GL_RIGHT, GL_BACK_LEFT,
        // GL_FRONT_LEFT, GL_BACK_RIGHT, GL_FRONT_RIGHT, GL_AUX0..3.
        // Remap to the appropriate GLES-legal value based on the bound FBO.
        mapped = (GLState.framebuffer.readFBO != 0) ? GL_COLOR_ATTACHMENT0 : GL_BACK;
    }
    GLES.glReadBuffer(mapped);
    GLState.framebuffer.readBuffer = mapped;
}