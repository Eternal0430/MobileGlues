// MobileGlues - gl/glsl/glsl_for_es.h
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header
#ifndef GLSL_FOR_ES
#define GLSL_FOR_ES
#include "../../gles/loader.h"
#include "../../includes.h"
#include "../glcorearb.h"
#include "../log.h"
#include <GL/gl.h>
#include <stdio.h>
#include <string>

#include "../mg.h"

// ============================================================================
// GLSL → SPIR-V → GLSL ES Conversion Interface
// ============================================================================
// Pipeline: Desktop GLSL → glslang(EShClientOpenGL) → SPIR-V 1.5 → spirv-cross → GLSL ES 3.2
// Target:   OpenGL ES 3.2 (ESSL 320)
// ============================================================================

// Main entry point: convert desktop GLSL to GLSL ES with caching
//   glsl_code   - input desktop GLSL source
//   glsl_type   - shader type (GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, etc.)
//   essl_version - target ESSL version (320 for ES 3.2)
//   glsl_version - desktop GLSL version hint (0 = auto-detect)
//   return_code  - output: 0=success, 1=success+atomicCounterEmulated, <0=error
// Returns converted GLSL ES source, or empty string on failure
std::string GLSLtoGLSLES(const char* glsl_code, GLenum glsl_type, uint essl_version, uint glsl_version,
                         int& return_code);

// Direct conversion without caching (used internally by GLSLtoGLSLES through _2)
std::string GLSLtoGLSLES_2(const char* glsl_code, GLenum glsl_type, uint essl_version, int& return_code);

// Extract #version from GLSL source (e.g. "#version 330 core" → 330, not found → -1)
int getGLSLVersion(const char* glsl_code);

// Pre-initialize glslang (builds its built-in symbol tables). Idempotent and
// thread-safe via std::call_once. Calling this at startup moves the one-time
// glslang init cost out of the first glShaderSource path, avoiding the
// combined "init + full compile" CPU spike on the first rendered frame.
void init_glslang_once();

#endif