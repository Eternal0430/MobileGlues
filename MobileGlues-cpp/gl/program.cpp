// MobileGlues - gl/program.cpp
// Program state management: ID mapping, linking, usage tracking.
// Shader compilation and source management is handled by shader.cpp.
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
#include <cstring>

#define DEBUG 0

// ============================================================================
// Program Object Lifecycle (ES 3.2 native)
// ============================================================================

GLuint glCreateProgram() {
    LOG()
    GLuint program = GLES.glCreateProgram();
    // Track in state manager (virtual == real for programs)
    auto &ss = GLState.shader;
    ss.programMap[program] = program;
    ss.programMapReverse[program] = program;
    ss.programInfo[program] = GLStateManager::ShaderState::ProgramInfo{};
    return program;
}

void glDeleteProgram(GLuint program) {
    LOG()
    auto &ss = GLState.shader;
    if (ss.currentProgram == program) {
        ss.currentProgram = 0;
        GLState.currentProgram = 0;
        // Unbind from GLES so the cache (now 0) matches the actual bound program.
        // This keeps the glUseProgram short-circuit valid after deletion.
        GLES.glUseProgram(0);
    }
    ss.programMap.erase(program);
    ss.programMapReverse.erase(program);
    ss.programInfo.erase(program);
    GLES.glDeleteProgram(program);
    CHECK_GL_ERROR
}

// ============================================================================
// Shader Object Lifecycle
// ============================================================================

// glCreateShader is handled in shader.cpp (GLSL conversion pipeline)
// glShaderSource is handled in shader.cpp
// glCompileShader is handled in shader.cpp

void glDeleteShader(GLuint shader) {
    LOG()
    auto &ss = GLState.shader;
    ss.shaderMap.erase(shader);
    ss.shaderMapReverse.erase(shader);
    ss.shaderInfo.erase(shader);
    GLES.glDeleteShader(shader);
    CHECK_GL_ERROR
}

// ============================================================================
// Program Linking & Validation (ES 3.2 native)
// ============================================================================

void glAttachShader(GLuint program, GLuint shader) {
    LOG()
    GLES.glAttachShader(program, shader);
    // Track in state manager
    auto &ss = GLState.shader;
    auto it = ss.programInfo.find(program);
    if (it != ss.programInfo.end()) {
        it->second.attachedShaders.push_back(shader);
    } else {
        ss.programInfo[program] = GLStateManager::ShaderState::ProgramInfo{};
        ss.programInfo[program].attachedShaders.push_back(shader);
    }
}

void glDetachShader(GLuint program, GLuint shader) {
    LOG()
    GLES.glDetachShader(program, shader);
    auto &ss = GLState.shader;
    auto it = ss.programInfo.find(program);
    if (it != ss.programInfo.end()) {
        auto &vec = it->second.attachedShaders;
        vec.erase(std::remove(vec.begin(), vec.end(), shader), vec.end());
    }
    CHECK_GL_ERROR
}

void glLinkProgram(GLuint program) {
    LOG()
    GLES.glLinkProgram(program);

    GLint linkStatus = 0;
    GLES.glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);

    auto &ss = GLState.shader;
    auto it = ss.programInfo.find(program);
    if (it != ss.programInfo.end()) {
        it->second.linked = (linkStatus == GL_TRUE);
    }

    if (linkStatus == GL_FALSE) {
        GLint infoLogLength = 0;
        GLES.glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);
        if (infoLogLength > 1) {
            char* infoLog = new char[infoLogLength];
            GLES.glGetProgramInfoLog(program, infoLogLength, nullptr, infoLog);
            MG_LOG_ERROR("glLinkProgram FAILED: %s", infoLog);
            delete[] infoLog;
        }
    }
}

void glValidateProgram(GLuint program) {
    LOG()
    GLES.glValidateProgram(program);

    GLint validateStatus = 0;
    GLES.glGetProgramiv(program, GL_VALIDATE_STATUS, &validateStatus);

    auto &ss = GLState.shader;
    auto it = ss.programInfo.find(program);
    if (it != ss.programInfo.end()) {
        it->second.validated = (validateStatus == GL_TRUE);
    }

    if (validateStatus == GL_FALSE) {
        GLint infoLogLength = 0;
        GLES.glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);
        if (infoLogLength > 1) {
            char* infoLog = new char[infoLogLength];
            GLES.glGetProgramInfoLog(program, infoLogLength, nullptr, infoLog);
            MG_LOG_ERROR("glValidateProgram FAILED: %s", infoLog);
            delete[] infoLog;
        }
    }
}

// ============================================================================
// Program Usage (ES 3.2 native)
// ============================================================================

void glUseProgram(GLuint program) {
    LOG()
    // Short-circuit: same program already active, skip the GLES driver call.
    // Safe because GLState.shader.currentProgram is kept in sync with the actual
    // GLES bound program: glDeleteProgram unbinds (→0), FSR1's GLStateGuard
    // syncs the cache on scope exit, and multidraw restores prev_program before
    // returning.
    if (GLState.shader.currentProgram == program) [[likely]] return;
    GLState.shader.currentProgram = program;
    GLState.currentProgram = program;
    GLES.glUseProgram(program);
}

// ============================================================================
// Program/Shader Queries (ES 3.2 native)
// ============================================================================

void glGetProgramiv(GLuint program, GLenum pname, GLint* params) {
    GLES.glGetProgramiv(program, pname, params);
}

void glGetShaderiv(GLuint shader, GLenum pname, GLint* params) {
    GLES.glGetShaderiv(shader, pname, params);
}

void glGetProgramInfoLog(GLuint program, GLsizei bufSize, GLsizei* length, GLchar* infoLog) {
    LOG()
    GLES.glGetProgramInfoLog(program, bufSize, length, infoLog);
    CHECK_GL_ERROR
}

void glGetShaderInfoLog(GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* infoLog) {
    LOG()
    GLES.glGetShaderInfoLog(shader, bufSize, length, infoLog);
    CHECK_GL_ERROR
}

void glGetAttachedShaders(GLuint program, GLsizei maxCount, GLsizei* count, GLuint* shaders) {
    LOG()
    GLES.glGetAttachedShaders(program, maxCount, count, shaders);
    CHECK_GL_ERROR
}

// ============================================================================
// Attribute Location (ES 3.2 native)
// ============================================================================

void glBindAttribLocation(GLuint program, GLuint index, const GLchar* name) {
    LOG()
    GLES.glBindAttribLocation(program, index, name);
    CHECK_GL_ERROR
}

GLint glGetAttribLocation(GLuint program, const GLchar* name) {
    LOG()
    GLint result = GLES.glGetAttribLocation(program, name);
    CHECK_GL_ERROR
    return result;
}

// ============================================================================
// Uniform Location (ES 3.2 native)
// ============================================================================

GLint glGetUniformLocation(GLuint program, const GLchar* name) {
    LOG()
    GLint result = GLES.glGetUniformLocation(program, name);
    CHECK_GL_ERROR
    return result;
}

// ============================================================================
// Active Attribute/Uniform Queries (ES 3.2 native)
// ============================================================================

void glGetActiveAttrib(GLuint program, GLuint index, GLsizei bufSize, GLsizei* length, GLint* size,
                       GLenum* type, GLchar* name) {
    LOG()
    GLES.glGetActiveAttrib(program, index, bufSize, length, size, type, name);
    CHECK_GL_ERROR
}

void glGetActiveUniform(GLuint program, GLuint index, GLsizei bufSize, GLsizei* length, GLint* size,
                        GLenum* type, GLchar* name) {
    LOG()
    GLES.glGetActiveUniform(program, index, bufSize, length, size, type, name);
    CHECK_GL_ERROR
}



