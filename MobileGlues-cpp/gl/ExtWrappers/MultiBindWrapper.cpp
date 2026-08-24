// MobileGlues - gl/ExtWrappers/MultiBindWrapper.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

#include "MultiBindWrapper.h"
#include <cassert>
#include "../texture.h"

#define DEBUG 0

void glBindTextures(GLuint first, GLsizei count, const GLuint* textures) {
    GLuint prevUnit = GL_TEXTURE0 + GetCurrentTextureUnitIndex();
    for (GLsizei i = 0; i < count; ++i) {
        if (textures[i] == 0) {
            // Unbind: use GL_TEXTURE_2D as default target for unbind
            glActiveTexture(GL_TEXTURE0 + first + i);
            glBindTexture(GL_TEXTURE_2D, 0);
        } else {
            auto* texObj = mgGetTexObjectByID(textures[i]);
            if (!texObj) {
                LOG_E("glBindTextures: texture %u not found in tracking table", textures[i]);
                continue;
            }
            GLenum target = ConvertTextureTargetToGLEnum(texObj->target);
            glActiveTexture(GL_TEXTURE0 + first + i);
            glBindTexture(target, textures[i]);
        }
    }
    glActiveTexture(prevUnit);
}

void glBindSamplers(GLuint first, GLsizei count, const GLuint* samplers) {
    for (GLsizei i = 0; i < count; ++i) {
        glBindSampler(first + i, samplers[i]);
    }
}

void glBindImageTextures(GLuint first, GLsizei count, const GLuint* textures) {
    for (int i = 0; i < count; i++) {
        if (textures == nullptr || textures[i] == 0) {
            glBindImageTexture(first + i, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8);
        } else {
            auto* texObj = mgGetTexObjectByID(textures[i]);
            GLenum fmt = (texObj != nullptr) ? texObj->internal_format : GL_R8;
            glBindImageTexture(first + i, textures[i], 0, GL_TRUE, 0, GL_READ_WRITE, fmt);
        }
    }
}

void glBindVertexBuffers(GLuint first, GLsizei count, const GLuint* buffers, const GLintptr* offsets,
                         const GLsizei* strides) {
    for (GLsizei i = 0; i < count; ++i) {
        glBindVertexBuffer(first + i, buffers[i], offsets[i], strides[i]);
    }
}
