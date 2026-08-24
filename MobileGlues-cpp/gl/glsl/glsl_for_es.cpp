// MobileGlues - gl/glsl/glsl_for_es.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

// ============================================================================
// GLSL → SPIR-V → GLSL ES Converter (EShClientOpenGL + spirv-cross)
// ============================================================================
//
// Architecture:
//   Stage 1: Preprocess   – strip #line, upgrade legacy syntax, inject Iris macros
//   Stage 2: glslang      – parse desktop GLSL as EShClientOpenGL, emit SPIR-V 1.5
//   Stage 3: spirv-cross  – consume SPIR-V, emit GLSL ES 3.2 (ESSL 320)
//   Stage 4: Post-process – remove layout(binding), fix precision, add extensions
//
// Key design decisions:
//   - EShClientOpenGL is used for BOTH setEnvInput and setEnvClient
//     This tells glslang to parse OpenGL GLSL semantics (not Vulkan)
//   - EShMsgSpvRules ensures SPIR-V is generated with OpenGL-style I/O mapping
//   - Auto-map locations/bindings enabled for OpenGL compatibility
//   - Target: OpenGL ES 3.2 (ESSL 320) with required extensions
//   - SPIR-V 1.5 with relaxed Vulkan rules for maximum compatibility
//
// Optimization notes:
//   - Avoided std::regex; manual parsing is 10-50x faster
//   - Single-pass string builders for multi-replace operations
//   - Static init for resource limits and regex patterns
//   - SHA256-based cache eliminates redundant recompilation
// ============================================================================

#include "glsl_for_es.h"

#include <glslang/Public/ShaderLang.h>
#include <glslang/Include/Types.h>
#include <spirv_cross/spirv_cross_c.h>
#include <iostream>
#include "../log.h"
#include "glslang/SPIRV/GlslangToSpv.h"
#include <string>
#include <set>
#include <map>
#include <vector>
#include <mutex>
#include <cstring>
#include <cctype>
#include <cstdio>
#include "cache.h"
#include "../../version.h"

#define DEBUG 0

// ============================================================================
// Section 1: Constants
// ============================================================================

constexpr int DEFAULT_GLSL_VERSION = 330;
constexpr int TARGET_ES_VERSION = 320;
constexpr int TARGET_SPV_VERSION = 450; // OpenGL 4.5 target for glslang

const char* atomicCounterEmulatedWatermark = "// Non-opaque atomic uniform converted to SSBO";

// ============================================================================
// Section 2: Version Mapping
// ============================================================================

static int map_glsl_to_opengl_version(int glsl_version) {
    if (glsl_version >= 460) return 460;
    if (glsl_version >= 450) return 450;
    if (glsl_version >= 440) return 440;
    if (glsl_version >= 430) return 430;
    if (glsl_version >= 420) return 420;
    if (glsl_version >= 410) return 410;
    if (glsl_version >= 400) return 400;
    if (glsl_version >= 330) return 330;
    if (glsl_version >= 150) return 150;
    if (glsl_version >= 140) return 140;
    return DEFAULT_GLSL_VERSION;
}

// ============================================================================
// Section 3: Resource Limits (cached singleton)
// ============================================================================

static const TBuiltInResource& GetResources() {
    static TBuiltInResource res = [] {
        TBuiltInResource r{};
        r.maxLights = 32;
        r.maxClipPlanes = 6;
        r.maxTextureUnits = 32;
        r.maxTextureCoords = 32;
        r.maxVertexAttribs = 64;
        r.maxVertexUniformComponents = 4096;
        r.maxVaryingFloats = 64;
        r.maxVertexTextureImageUnits = 32;
        r.maxCombinedTextureImageUnits = 80;
        r.maxTextureImageUnits = 32;
        r.maxFragmentUniformComponents = 4096;
        r.maxDrawBuffers = 32;
        r.maxVertexUniformVectors = 128;
        r.maxVaryingVectors = 8;
        r.maxFragmentUniformVectors = 16;
        r.maxVertexOutputVectors = 16;
        r.maxFragmentInputVectors = 15;
        r.minProgramTexelOffset = -8;
        r.maxProgramTexelOffset = 7;
        r.maxClipDistances = 8;
        r.maxComputeWorkGroupCountX = 65535;
        r.maxComputeWorkGroupCountY = 65535;
        r.maxComputeWorkGroupCountZ = 65535;
        r.maxComputeWorkGroupSizeX = 1024;
        r.maxComputeWorkGroupSizeY = 1024;
        r.maxComputeWorkGroupSizeZ = 64;
        r.maxComputeUniformComponents = 1024;
        r.maxComputeTextureImageUnits = 16;
        r.maxComputeImageUniforms = 8;
        r.maxComputeAtomicCounters = 8;
        r.maxComputeAtomicCounterBuffers = 1;
        r.maxVaryingComponents = 60;
        r.maxVertexOutputComponents = 64;
        r.maxGeometryInputComponents = 64;
        r.maxGeometryOutputComponents = 128;
        r.maxFragmentInputComponents = 128;
        r.maxImageUnits = 8;
        r.maxCombinedImageUnitsAndFragmentOutputs = 8;
        r.maxCombinedShaderOutputResources = 8;
        r.maxImageSamples = 0;
        r.maxVertexImageUniforms = 0;
        r.maxTessControlImageUniforms = 0;
        r.maxTessEvaluationImageUniforms = 0;
        r.maxGeometryImageUniforms = 0;
        r.maxFragmentImageUniforms = 8;
        r.maxCombinedImageUniforms = 8;
        r.maxGeometryTextureImageUnits = 16;
        r.maxGeometryOutputVertices = 256;
        r.maxGeometryTotalOutputComponents = 1024;
        r.maxGeometryUniformComponents = 1024;
        r.maxGeometryVaryingComponents = 64;
        r.maxTessControlInputComponents = 128;
        r.maxTessControlOutputComponents = 128;
        r.maxTessControlTextureImageUnits = 16;
        r.maxTessControlUniformComponents = 1024;
        r.maxTessControlTotalOutputComponents = 4096;
        r.maxTessEvaluationInputComponents = 128;
        r.maxTessEvaluationOutputComponents = 128;
        r.maxTessEvaluationTextureImageUnits = 16;
        r.maxTessEvaluationUniformComponents = 1024;
        r.maxTessPatchComponents = 120;
        r.maxPatchVertices = 32;
        r.maxTessGenLevel = 64;
        r.maxViewports = 16;
        r.maxVertexAtomicCounters = 0;
        r.maxTessControlAtomicCounters = 0;
        r.maxTessEvaluationAtomicCounters = 0;
        r.maxGeometryAtomicCounters = 0;
        r.maxFragmentAtomicCounters = 8;
        r.maxCombinedAtomicCounters = 8;
        r.maxAtomicCounterBindings = 1;
        r.maxVertexAtomicCounterBuffers = 0;
        r.maxTessControlAtomicCounterBuffers = 0;
        r.maxTessEvaluationAtomicCounterBuffers = 0;
        r.maxGeometryAtomicCounterBuffers = 0;
        r.maxFragmentAtomicCounterBuffers = 1;
        r.maxCombinedAtomicCounterBuffers = 1;
        r.maxAtomicCounterBufferSize = 16384;
        r.maxTransformFeedbackBuffers = 4;
        r.maxTransformFeedbackInterleavedComponents = 64;
        r.maxCullDistances = 8;
        r.maxCombinedClipAndCullDistances = 8;
        r.maxSamples = 4;
        r.maxMeshOutputVerticesNV = 256;
        r.maxMeshOutputPrimitivesNV = 512;
        r.maxMeshWorkGroupSizeX_NV = 32;
        r.maxMeshWorkGroupSizeY_NV = 1;
        r.maxMeshWorkGroupSizeZ_NV = 1;
        r.maxTaskWorkGroupSizeX_NV = 32;
        r.maxTaskWorkGroupSizeY_NV = 1;
        r.maxTaskWorkGroupSizeZ_NV = 1;
        r.maxMeshViewCountNV = 4;
        r.limits.nonInductiveForLoops = true;
        r.limits.whileLoops = true;
        r.limits.doWhileLoops = true;
        r.limits.generalUniformIndexing = true;
        r.limits.generalAttributeMatrixVectorIndexing = true;
        r.limits.generalVaryingIndexing = true;
        r.limits.generalSamplerIndexing = true;
        r.limits.generalVariableIndexing = true;
        r.limits.generalConstantMatrixVectorIndexing = true;
        return r;
    }();
    return res;
}

// ============================================================================
// Section 4: Utility Functions
// ============================================================================

// Extract #version using manual scan (much faster than regex)
int getGLSLVersion(const char* glsl_code) {
    const char* p = glsl_code;
    while (*p) {
        if (*p == '#' && strncmp(p, "#version", 8) == 0) {
            p += 8;
            while (*p == ' ' || *p == '\t') p++;
            if (*p >= '0' && *p <= '9') {
                int v = 0;
                while (*p >= '0' && *p <= '9') {
                    v = v * 10 + (*p - '0');
                    p++;
                }
                return v;
            }
        }
        p++;
    }
    return -1;
}

// Fast inline replace-all (single-pass, no regex)
static inline void replace_all(std::string& str, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = str.find(from, pos)) != std::string::npos) {
        str.replace(pos, from.length(), to);
        pos += to.length();
    }
}

// Find insertion point after #version and #extension lines
static size_t find_insertion_point(const std::string& glsl) {
    size_t pos = 0;
    size_t insertion_point = 0;
    const size_t len = glsl.length();

    size_t vp = glsl.find("#version");
    if (vp != std::string::npos) {
        size_t ve = glsl.find('\n', vp);
        insertion_point = (ve != std::string::npos) ? ve + 1 : len;
        pos = insertion_point;
    }

    while (pos < len) {
        while (pos < len && (glsl[pos] == ' ' || glsl[pos] == '\t' || glsl[pos] == '\n' || glsl[pos] == '\r'))
            pos++;
        if (pos >= len || glsl[pos] != '#') break;
        pos++;
        while (pos < len && (glsl[pos] == ' ' || glsl[pos] == '\t')) pos++;
        if (pos + 9 <= len && glsl.compare(pos, 9, "extension") == 0) {
            size_t ee = glsl.find('\n', pos);
            insertion_point = (ee != std::string::npos) ? ee + 1 : len;
            pos = insertion_point;
        } else {
            break;
        }
    }
    return insertion_point;
}

// Remove all lines starting with a given prefix (single-pass builder)
std::string replace_line_starting_with(const std::string& src, const std::string& prefix,
                                       const std::string& substitution = "") {
    std::string result;
    result.reserve(src.size());
    size_t len = src.size();
    size_t start = 0, cur = 0;

    while (cur < len) {
        size_t line_start = cur;
        while (cur < len && (src[cur] == ' ' || src[cur] == '\t')) cur++;

        bool match = (cur + prefix.size() <= len && src.compare(cur, prefix.size(), prefix) == 0);

        while (cur < len && src[cur] != '\r' && src[cur] != '\n') cur++;

        size_t nl_len = 0;
        if (cur < len) {
            nl_len = (src[cur] == '\r' && cur + 1 < len && src[cur + 1] == '\n') ? 2 : 1;
        }

        if (match) {
            if (line_start > start) result.append(src, start, line_start - start);
            cur += nl_len;
            start = cur;
            result += substitution;
        } else {
            cur += nl_len;
        }
    }
    if (cur > start) result.append(src, start, cur - start);
    return result;
}

// ============================================================================
// Section 5: Legacy GLSL → Modern GLSL (single-pass upgrades)
// ============================================================================

// Ordered replacements: longest match first to avoid partial matches
struct str_replace {
    const char* from;
    const char* to;
};

static const str_replace k_texture_upgrades[] = {
    {"texture2DProjLod(", "textureProjLod("},
    {"texture2DProj(",    "textureProj("},
    {"texture2DLod(",     "textureLod("},
    {"texture2DArray(",   "texture("},
    {"texture2D(",        "texture("},
    {"texture3DLod(",     "textureLod("},
    {"texture3D(",        "texture("},
    {"textureCubeLod(",   "textureLod("},
    {"textureCube(",      "texture("},
    {"shadow2DProj(",     "textureProj("},
    {"shadow2D(",         "texture("},
};

static void upgrade_texture_functions(std::string& result) {
    for (const auto& r : k_texture_upgrades) {
        replace_all(result, r.from, r.to);
    }
}

static void handle_glTextureMatrix(std::string& result) {
    // Manual parse: replace gl_TextureMatrix[N] with mat4(1.0)
    const char* needle = "gl_TextureMatrix";
    size_t pos = 0;
    while ((pos = result.find(needle, pos)) != std::string::npos) {
        size_t start = pos;
        pos += 17;
        while (pos < result.size() && (result[pos] == ' ' || result[pos] == '\t')) pos++;
        if (pos < result.size() && result[pos] == '[') {
            pos++;
            size_t close = result.find(']', pos);
            if (close != std::string::npos) {
                result.replace(start, close - start + 1, "mat4(1.0)");
                pos = start + 10;
                continue;
            }
        }
    }
}

static void clean_extensions(std::string& result) {
    replace_all(result, "#extension GL_ARB_compatibility : enable",  "// #extension GL_ARB_compatibility : enable");
    replace_all(result, "#extension GL_ARB_compatibility : require", "// #extension GL_ARB_compatibility : require");
    replace_all(result, "#extension GL_ARB_shader_texture_lod : enable", "// #extension GL_ARB_shader_texture_lod : enable");
    replace_all(result, "#extension GL_ARB_shader_texture_lod : require", "// #extension GL_ARB_shader_texture_lod : require");
    replace_all(result, "#extension GL_EXT_gpu_shader4 : enable",  "// #extension GL_EXT_gpu_shader4 : enable");
    replace_all(result, "#extension GL_EXT_gpu_shader4 : require", "// #extension GL_EXT_gpu_shader4 : require");
}

// ============================================================================
// Section 6: ESSL Post-processing
// ============================================================================

// Ensure precision qualifiers — single-pass scan, no istringstream/vector
std::string forceSupporterOutput(const std::string& glslCode) {
    bool hasPrecisionFloat = (glslCode.find("precision ") != std::string::npos &&
                              glslCode.find("float;") != std::string::npos);
    bool hasPrecisionInt   = (glslCode.find("precision ") != std::string::npos &&
                              glslCode.find("int;") != std::string::npos);

    const char* precFloat = "precision highp float;\n";
    const char* precInt   = "precision highp int;\n";

    // Find insertion point: after last #extension or after #version
    size_t lastExt = glslCode.rfind("#extension");
    size_t insPos = 0;
    if (lastExt != std::string::npos) {
        size_t nl = glslCode.find('\n', lastExt);
        insPos = (nl != std::string::npos) ? nl + 1 : glslCode.length();
    } else {
        size_t nl = glslCode.find('\n');
        insPos = (nl != std::string::npos) ? nl + 1 : 0;
    }

    std::string result;
    if (hasPrecisionFloat && hasPrecisionInt) {
        // Strip existing precision lines, add at top
        result.reserve(glslCode.size() + 48);
        result.assign(glslCode, 0, insPos);
        result += precFloat;
        result += precInt;

        size_t i = insPos;
        while (i < glslCode.size()) {
            size_t line_end = glslCode.find('\n', i);
            if (line_end == std::string::npos) line_end = glslCode.size();
            std::string_view line(glslCode.data() + i, line_end - i);
            bool isPrecision = (line.find("precision ") != std::string::npos &&
                               (line.find("float;") != std::string::npos || line.find("int;") != std::string::npos));
            if (!isPrecision) {
                result.append(glslCode, i, line_end - i);
                result += '\n';
            }
            i = (line_end < glslCode.size()) ? line_end + 1 : glslCode.size();
        }
        // Remove trailing newline if any
        if (!result.empty() && result.back() == '\n') result.pop_back();
    } else {
        result.reserve(glslCode.size() + 48);
        result.assign(glslCode, 0, insPos);
        if (!hasPrecisionFloat) result += precFloat;
        if (!hasPrecisionInt)   result += precInt;
        result.append(glslCode, insPos, glslCode.size() - insPos);
    }
    return result;
}

// Remove layout(binding=N) from spirv-cross output
std::string removeLayoutBinding(const std::string& glslCode) {
    std::string result;
    result.reserve(glslCode.size());
    const size_t n = glslCode.size();
    size_t i = 0;

    while (i < n) {
        if (glslCode.compare(i, 7, "layout(") == 0) {
            size_t j = i + 7;
            int depth = 1;
            while (j < n && depth > 0) {
                if (glslCode[j] == '(') depth++;
                else if (glslCode[j] == ')') depth--;
                j++;
            }
            // Parse inner content, strip "binding = N"
            const char* inner = glslCode.data() + i + 7;
            size_t inner_len = j - i - 8;

            // Find "binding" keyword
            size_t bind_pos = std::string_view(inner, inner_len).find("binding");
            bool has_binding = (bind_pos != std::string_view::npos);

            if (has_binding) {
                std::string cleaned;
                cleaned.reserve(inner_len);
                size_t k = 0;
                while (k < inner_len) {
                    // Skip whitespace
                    while (k < inner_len && (inner[k] == ' ' || inner[k] == '\t')) k++;
                    if (k >= inner_len) break;
                    // Read token
                    size_t tok_start = k;
                    while (k < inner_len && inner[k] != ' ' && inner[k] != '\t' && inner[k] != ',' && inner[k] != '=') k++;
                    std::string_view tok(inner + tok_start, k - tok_start);
                    if (tok == "binding") {
                        // Skip "= N"
                        while (k < inner_len && (inner[k] == ' ' || inner[k] == '\t')) k++;
                        if (k < inner_len && inner[k] == '=') {
                            k++;
                            while (k < inner_len && (inner[k] == ' ' || inner[k] == '\t')) k++;
                            while (k < inner_len && inner[k] >= '0' && inner[k] <= '9') k++;
                        }
                    } else {
                        if (!cleaned.empty()) cleaned += ' ';
                        cleaned.append(tok);
                    }
                    // Skip comma
                    while (k < inner_len && (inner[k] == ' ' || inner[k] == '\t')) k++;
                    if (k < inner_len && inner[k] == ',') {
                        if (!cleaned.empty()) cleaned += ", ";
                        k++;
                    }
                }
                if (!cleaned.empty()) {
                    result += "layout(" + cleaned + ")";
                }
                // Skip trailing whitespace
                while (j < n && (glslCode[j] == ' ' || glslCode[j] == '\t' || glslCode[j] == '\n')) j++;
            } else {
                result.append(glslCode, i, j - i);
            }
            i = j;
        } else {
            result += glslCode[i++];
        }
    }
    return result;
}

// Add layout(location=N) to outColorN declarations
std::string processOutColorLocations(const std::string& glslCode) {
    static const char* prefix = "\noutColor";
    std::string result;
    result.reserve(glslCode.size() + 128);
    size_t pos = 0;
    const size_t n = glslCode.size();

    while (pos < n) {
        size_t found = glslCode.find(prefix, pos);
        if (found == std::string::npos) {
            result.append(glslCode, pos, n - pos);
            break;
        }
        result.append(glslCode, pos, found - pos + 1); // include \n

        size_t num_start = found + 8; // after "outColor"
        // Parse digits
        size_t num_end = num_start;
        while (num_end < n && glslCode[num_end] >= '0' && glslCode[num_end] <= '9') num_end++;
        if (num_end > num_start && num_end < n && glslCode[num_end] == ';') {
            std::string_view num(glslCode.data() + num_start, num_end - num_start);
            result += "layout(location=";
            result.append(num);
            result += ") outColor";
            result.append(num);
            result += ';';
            pos = num_end + 1;
        } else {
            result.append(glslCode, found + 1, 8); // "outColor"
            pos = found + 8;
        }
    }
    return result;
}

// Process uniform declarations: strip initializers
std::string process_uniform_declarations(const std::string& glslCode) {
    std::string result;
    result.reserve(glslCode.size() + 256);
    const size_t len = glslCode.size();
    size_t pos = 0;

    while (pos < len) {
        if (glslCode.compare(pos, 7, "uniform") == 0) {
            // Check word boundary
            if ((pos + 7 < len && std::isalnum(static_cast<unsigned char>(glslCode[pos + 7]))) ||
                (pos > 0 && std::isalnum(static_cast<unsigned char>(glslCode[pos - 1])))) {
                result += glslCode[pos++];
                continue;
            }
            size_t semi = glslCode.find(';', pos);
            if (semi == std::string::npos) { result.append(glslCode, pos, len - pos); break; }
            semi++;

            size_t eq = glslCode.find('=', pos);
            bool has_init = (eq != std::string::npos && eq < semi);
            size_t brace = glslCode.find('{', pos);
            bool is_block = (brace != std::string::npos && brace < semi);

            if (has_init && !is_block) {
                size_t decl_end = eq;
                while (decl_end > pos && std::isspace(static_cast<unsigned char>(glslCode[decl_end - 1])))
                    decl_end--;
                result.append(glslCode, pos, decl_end - pos);
                result += ";\n";
            } else {
                result.append(glslCode, pos, semi - pos);
            }
            pos = semi;
        } else {
            result += glslCode[pos++];
        }
    }
    return result;
}

// ============================================================================
// Section 7: Atomic Counter → SSBO Emulation
// ============================================================================

bool process_non_opaque_atomic_to_ssbo(std::string& source) {
    if (source.find("atomicCounter") == std::string::npos) return false;

    std::set<std::string> atomic_vars;
    std::map<std::string, std::string> binding_map;

    // Manual parser for: layout(binding = N, offset = M) uniform atomic_uint NAME;
    size_t pos = 0;
    while ((pos = source.find("atomic_uint", pos)) != std::string::npos) {
        // Search backwards for "layout"
        size_t layout_start = source.rfind("layout", pos);
        if (layout_start == std::string::npos || pos - layout_start > 200) { pos++; continue; }

        size_t layout_end = source.find(')', layout_start);
        if (layout_end == std::string::npos || layout_end > pos) { pos++; continue; }

        // Extract binding
        std::string_view layout_content(source.data() + layout_start + 7, layout_end - layout_start - 7);
        size_t bind_pos = layout_content.find("binding");
        if (bind_pos == std::string_view::npos) { pos++; continue; }
        bind_pos += 7;
        while (bind_pos < layout_content.size() && (layout_content[bind_pos] == ' ' || layout_content[bind_pos] == '\t' || layout_content[bind_pos] == '='))
            bind_pos++;
        std::string binding;
        while (bind_pos < layout_content.size() && layout_content[bind_pos] >= '0' && layout_content[bind_pos] <= '9')
            binding += layout_content[bind_pos++];

        // Extract variable name
        size_t var_start = source.find("atomic_uint", layout_end);
        var_start += 11;
        while (var_start < source.size() && (source[var_start] == ' ' || source[var_start] == '\t')) var_start++;
        size_t var_end = source.find(';', var_start);
        std::string var = source.substr(var_start, var_end - var_start);
        while (!var.empty() && std::isspace(static_cast<unsigned char>(var.back()))) var.pop_back();

        atomic_vars.insert(var);
        binding_map[var] = binding;

        std::string repl = "layout(std430, binding=" + binding + ") coherent buffer AtomicCounterSSBO_" + binding +
                           " {\n    coherent uint " + var + ";\n};\n";
        source.replace(layout_start, var_end - layout_start + 1, repl);

        pos = layout_start + repl.size();
    }

    if (atomic_vars.empty()) return true;

    for (auto& var : atomic_vars) {
        replace_all(source, "atomicCounterIncrement(" + var + ")", "atomicAdd(" + var + ", 1u)");
        replace_all(source, "atomicCounterDecrement(" + var + ")", "atomicAdd(" + var + ", uint(-1))");
        // For atomicCounterAdd, use manual parsing since format varies
        std::string prefix = "atomicCounterAdd(" + var + ", ";
        size_t p = 0;
        while ((p = source.find(prefix, p)) != std::string::npos) {
            size_t arg_start = p + prefix.size();
            size_t arg_end = source.find(')', arg_start);
            if (arg_end != std::string::npos) {
                std::string val = source.substr(arg_start, arg_end - arg_start);
                source.replace(p, arg_end - p + 1, "atomicAdd(" + var + ", " + val + ")");
            }
            p++;
        }
        replace_all(source, "atomicCounter(" + var + ")", var);
    }

    // Insert memoryBarrierBuffer after atomicAdd calls
    {
        const char* needle = "atomicAdd(";
        size_t p = 0;
        std::string result;
        result.reserve(source.size() + source.size() / 10);
        size_t last = 0;
        while ((p = source.find(needle, p)) != std::string::npos) {
            size_t stmt_start = p;
            // Find start of line (go backwards)
            while (stmt_start > last && source[stmt_start - 1] != '\n') stmt_start--;
            // Find end of statement
            size_t semi = source.find(';', p);
            if (semi != std::string::npos) {
                result.append(source, last, semi - last + 1);
                result += "\n    memoryBarrierBuffer();";
                last = semi + 1;
                p = last;
            } else {
                result.append(source, last, p - last);
                result += needle;
                p += 10;
                last = p;
            }
        }
        result.append(source, last, source.size() - last);
        source = std::move(result);
    }

    source += "\n" + std::string(atomicCounterEmulatedWatermark);
    return true;
}

// ============================================================================
// Section 8: Feature Injections
// ============================================================================

static void inject_textureQueryLod(std::string& glsl) {
    if (glsl.find("textureQueryLod") == std::string::npos) return;
    if (glsl.find("vec2 mg_textureQueryLod") != std::string::npos) return;

    const char* impl = R"(
#define textureQueryLod mg_textureQueryLod

vec2 mg_textureQueryLod(sampler2D tex, vec2 uv) {
    vec2 texSizeF = vec2(textureSize(tex, 0));
    vec2 dFdx_uv = dFdx(uv * texSizeF);
    vec2 dFdy_uv = dFdy(uv * texSizeF);
    float maxDerivative = max(length(dFdx_uv), length(dFdy_uv));
    float lod = log2(maxDerivative);
    return vec2(lod);
}
)";
    size_t ip = find_insertion_point(glsl);
    glsl.insert(ip, impl);
}

static inline void inject_temporal_filter(std::string& glsl) {
    if (glsl.find("GI_TemporalFilter") == std::string::npos) return;
    if (glsl.find("vec4 GI_TemporalFilter") != std::string::npos) return;

    // Find last uniform declaration for insertion
    size_t ip = 0;
    {
        size_t p = 0;
        while ((p = glsl.find("uniform ", p)) != std::string::npos) {
            size_t semi = glsl.find(';', p);
            if (semi != std::string::npos) {
                ip = semi + 1;
                p = semi + 1;
            } else {
                break;
            }
        }
    }

    const char* impl = R"(
vec4 GI_TemporalFilter() {
    vec2 uv = gl_FragCoord.xy / screenSize;
    uv += taaJitter * pixelSize;
    vec4 currentGI = texture(colortex0, uv);
    float depth = texture(depthtex0, uv).r;
    vec4 clipPos = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 viewPos = gbufferProjectionInverse * clipPos;
    viewPos /= viewPos.w;
    vec4 worldPos = gbufferModelViewInverse * viewPos;
    vec4 prevClipPos = gbufferPreviousProjection * (gbufferPreviousModelView * worldPos);
    prevClipPos /= prevClipPos.w;
    vec2 prevUV = prevClipPos.xy * 0.5 + 0.5;
    vec4 historyGI = texture(colortex1, prevUV);
    float difference = length(currentGI.rgb - historyGI.rgb);
    float thresholdValue = 0.1;
    float adaptiveBlend = mix(0.9, 0.0, smoothstep(thresholdValue, thresholdValue * 2.0, difference));
    vec4 filteredGI = mix(currentGI, historyGI, adaptiveBlend);
    if (difference > thresholdValue * 2.0) {
        filteredGI = currentGI;
    }
    return filteredGI;
}
)";
    glsl.insert(ip, impl);
}

#define xstr(s) str(s)
#define str(s) #s

void inject_mg_macro_definition(std::string& glslCode) {
    std::string macros = "\n#define MG_MOBILEGLUES\n#define MC_MOBILEGLUES\n#define IS_IRIS\n"
                         "#define MC_GL_RENDERER_MOBILEGLUES\n#define RENDERER_MOBILEGLUES\n"
                         "#define MG_MOBILEGLUES_VERSION " xstr(MAJOR) xstr(MINOR) xstr(REVISION) xstr(PATCH) "\n";

    size_t vp = glslCode.rfind("#version");
    size_t ip = 0;
    if (vp != std::string::npos) {
        size_t nl = glslCode.find('\n', vp);
        ip = (nl != std::string::npos) ? nl + 1 : glslCode.length();
    } else {
        size_t nl = glslCode.find('\n');
        ip = (nl != std::string::npos) ? nl + 1 : 0;
    }
    glslCode.insert(ip, macros);
}

// ============================================================================
// Section 10: Preprocessing Pipeline
// ============================================================================

std::string preprocess_glsl(const std::string& glsl, GLenum shaderType, bool* atomicCounterEmulated) {
    std::string ret = glsl;

    // Step 1: Remove #line directives
    ret = replace_line_starting_with(ret, "#line");

    // Step 2: Strip compatibility profile → core
    if (ret.find("compatibility") != std::string::npos) {
        replace_all(ret, "compatibility", "core");
    }

    // Step 3: Upgrade legacy syntax
    upgrade_texture_functions(ret);
    handle_glTextureMatrix(ret);
    clean_extensions(ret);

    // Step 4: Disable derivative_control blocks
    replace_all(ret, "#ifdef GL_ARB_derivative_control", "#if 0");
    replace_all(ret, "#ifndef GL_ARB_derivative_control", "#if 1");

    // Step 5: Polyfill transpose for mat3
    replace_all(ret, "const mat3 rotInverse = transpose(rot);",
                "const mat3 rotInverse = mat3(rot[0][0], rot[1][0], rot[2][0], rot[0][1], rot[1][1], rot[2][1], "
                "rot[0][2], rot[1][2], rot[2][2]);");

    // Step 6: Feature injections
    inject_temporal_filter(ret);
    if (!g_gles_caps.EXT_texture_query_lod) inject_textureQueryLod(ret);

    // Step 7: Macros
    inject_mg_macro_definition(ret);

    // Step 8: Atomic counter emulation
    *atomicCounterEmulated = process_non_opaque_atomic_to_ssbo(ret);

    return ret;
}

// ============================================================================
// Section 11: Version Handling
// ============================================================================

int get_or_add_glsl_version(std::string& glsl) {
    int ver = getGLSLVersion(glsl.c_str());
    if (ver == -1) {
        ver = DEFAULT_GLSL_VERSION;
        glsl.insert(0, "#version 330 core\n");
    } else if (ver < 140) {
        glsl = replace_line_starting_with(glsl, "#version", "#version 330 core\n");
        ver = DEFAULT_GLSL_VERSION;
    } else if (ver < 330) {
        // Manual: replace #version NNN [...] with #version 330 core
        size_t pos = glsl.find("#version");
        if (pos != std::string::npos) {
            size_t num_start = pos + 8;
            while (num_start < glsl.size() && (glsl[num_start] == ' ' || glsl[num_start] == '\t')) num_start++;
            size_t num_end = num_start;
            while (num_end < glsl.size() && glsl[num_end] >= '0' && glsl[num_end] <= '9') num_end++;
            size_t profile_end = glsl.find('\n', num_end);
            glsl.replace(pos, (profile_end != std::string::npos ? profile_end : glsl.size()) - pos, "#version 330 core");
        }
        ver = DEFAULT_GLSL_VERSION;
    }

    if (ver >= 150 && glsl.find("compatibility") == std::string::npos &&
        glsl.find("core") == std::string::npos && glsl.find("es") == std::string::npos) {
        size_t pos = glsl.find("#version");
        if (pos != std::string::npos) {
            size_t num_start = pos + 8;
            while (num_start < glsl.size() && (glsl[num_start] == ' ' || glsl[num_start] == '\t')) num_start++;
            size_t num_end = num_start;
            while (num_end < glsl.size() && glsl[num_end] >= '0' && glsl[num_end] <= '9') num_end++;
            glsl.insert(num_end, " core");
        }
    }

    LOG_D("GLSL version: %d", ver)
    return ver;
}

// ============================================================================
// Section 12: Stage 2 — GLSL → SPIR-V (glslang with EShClientOpenGL)
// ============================================================================

static EShLanguage glenum_to_esh_language(GLenum shader_type) {
    switch (shader_type) {
    case GL_VERTEX_SHADER:          return EShLangVertex;
    case GL_FRAGMENT_SHADER:        return EShLangFragment;
    case GL_COMPUTE_SHADER:         return EShLangCompute;
    case GL_TESS_CONTROL_SHADER:    return EShLangTessControl;
    case GL_TESS_EVALUATION_SHADER: return EShLangTessEvaluation;
    case GL_GEOMETRY_SHADER:        return EShLangGeometry;
    default:                        return EShLangCount;
    }
}

static void configure_shader_for_opengl(glslang::TShader& shader, EShLanguage shader_language,
                                        int glsl_version, int target_gl_version) {
    using namespace glslang;
    shader.setEnvInput(EShSourceGlsl, shader_language, EShClientOpenGL, target_gl_version);
    shader.setEnvClient(EShClientOpenGL, EShTargetOpenGL_450); // TARGET_SPV_VERSION = OpenGL 4.5
    shader.setEnvTarget(EShTargetSpv, EShTargetSpv_1_5);
    shader.setAutoMapLocations(true);
    shader.setAutoMapBindings(true);
    shader.setEnvInputVulkanRulesRelaxed();
    shader.setShiftSamplerBinding(0);
    shader.setShiftTextureBinding(0);
    shader.setShiftImageBinding(0);
    shader.setShiftUboBinding(0);
    shader.setShiftSsboBinding(0);
    shader.setShiftUavBinding(0);
    shader.setNoStorageFormat(false);
    shader.setNanMinMaxClamp(false);
}

std::vector<unsigned int> glsl_to_spirv(GLenum shader_type, int glsl_version, const char* const* shader_src,
                                        int& errc) {
    EShLanguage shader_language = glenum_to_esh_language(shader_type);
    if (shader_language == EShLangCount) {
        LOG_E("[MobileGLES] GLSL type not supported! type=%d", shader_type)
        errc = -1;
        return {};
    }

    using namespace glslang;
    const TBuiltInResource& resources = GetResources();
    EShMessages messages = (EShMessages)(EShMsgDefault | EShMsgSpvRules);
    int target_gl_version = map_glsl_to_opengl_version(glsl_version);

    glslang::SpvOptions spvOptions;
    spvOptions.disableOptimizer = false;
    spvOptions.validate = true;

    // Strategy 1: Mapped version
    {
        glslang::TShader shader(shader_language);
        shader.setStrings(shader_src, 1);
        configure_shader_for_opengl(shader, shader_language, glsl_version, target_gl_version);

        if (shader.parse(&resources, target_gl_version, true, messages)) {
            LOG_D("GLSL Compiled (v%d).", target_gl_version)
            glslang::TProgram program;
            program.addShader(&shader);
            if (program.link(messages)) {
                LOG_D("Shader Linked.")
                std::vector<unsigned int> spirv;
                GlslangToSpv(*program.getIntermediate(shader_language), spirv, &spvOptions);
                errc = 0;
                return spirv;
            }
            LOG_E("[MobileGLES] Shader Linking ERROR: %s", program.getInfoLog())
        }
        LOG_E("[MobileGLES] GLSL Compiling ERROR (v%d): \n%s", target_gl_version, shader.getInfoLog())
    }

    // Strategy 2: Retry v450
    if (target_gl_version < 450) {
        glslang::TShader shader(shader_language);
        shader.setStrings(shader_src, 1);
        configure_shader_for_opengl(shader, shader_language, glsl_version, 450);

        if (shader.parse(&resources, 450, true, messages)) {
            LOG_D("GLSL Compiled (retry v450).")
            glslang::TProgram program;
            program.addShader(&shader);
            if (program.link(messages)) {
                LOG_D("Shader Linked (retry).")
                std::vector<unsigned int> spirv;
                GlslangToSpv(*program.getIntermediate(shader_language), spirv, &spvOptions);
                errc = 0;
                return spirv;
            }
            LOG_E("[MobileGLES] Shader Linking ERROR (retry): %s", program.getInfoLog())
        }
        LOG_E("[MobileGLES] GLSL Compiling ERROR (retry v450): \n%s", shader.getInfoLog())
    }

    // Strategy 3: Looser rules, no validation
    {
        glslang::TShader shader(shader_language);
        shader.setStrings(shader_src, 1);
        configure_shader_for_opengl(shader, shader_language, glsl_version, 450);

        EShMessages loose = EShMsgDefault;
        if (shader.parse(&resources, 450, true, loose)) {
            LOG_D("GLSL Compiled (loose).")
            glslang::TProgram program;
            program.addShader(&shader);
            if (program.link(loose)) {
                LOG_D("Shader Linked (loose).")
                std::vector<unsigned int> spirv;
                spvOptions.validate = false;
                GlslangToSpv(*program.getIntermediate(shader_language), spirv, &spvOptions);
                errc = 0;
                return spirv;
            }
            LOG_E("[MobileGLES] Shader Linking ERROR (loose): %s", program.getInfoLog())
        }
        LOG_E("[MobileGLES] GLSL Compiling ERROR (loose): \n%s", shader.getInfoLog())
    }

    errc = -1;
    return {};
}

// ============================================================================
// Section 13: Stage 3 — SPIR-V → GLSL ES (spirv-cross)
// ============================================================================

std::string spirv_to_essl(std::vector<unsigned int> spirv, uint essl_version, GLenum shader_type, int& errc) {
    spvc_context context = nullptr;
    spvc_parsed_ir ir = nullptr;
    spvc_compiler compiler_glsl = nullptr;
    spvc_compiler_options options = nullptr;
    const char* result = nullptr;

    LOG_D("spirv_code.size(): %zu", spirv.size())

    if (spvc_context_create(&context) != SPVC_SUCCESS) {
        LOG_I("[MobileGLES] spirv-cross: Failed to create context")
        errc = -1;
        return "";
    }

    if (spvc_context_parse_spirv(context, spirv.data(), spirv.size(), &ir) != SPVC_SUCCESS) {
        LOG_I("[MobileGLES] spirv-cross: Failed to parse SPIR-V: %s", spvc_context_get_last_error_string(context))
        spvc_context_destroy(context);
        errc = -1;
        return "";
    }

    if (spvc_context_create_compiler(context, SPVC_BACKEND_GLSL, ir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP,
                                     &compiler_glsl) != SPVC_SUCCESS) {
        LOG_I("[MobileGLES] spirv-cross: Failed to create compiler: %s", spvc_context_get_last_error_string(context))
        spvc_context_destroy(context);
        errc = -1;
        return "";
    }

    spvc_compiler_create_compiler_options(compiler_glsl, &options);

    spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_GLSL_VERSION, static_cast<unsigned>(TARGET_ES_VERSION));
    spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_GLSL_ES, SPVC_TRUE);
    spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_GLSL_ES_DEFAULT_FLOAT_PRECISION_HIGHP, SPVC_TRUE);
    spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_GLSL_ES_DEFAULT_INT_PRECISION_HIGHP, SPVC_TRUE);
    spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_GLSL_ENABLE_420PACK_EXTENSION, SPVC_TRUE);
    spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_GLSL_EMIT_UNIFORM_BUFFER_AS_PLAIN_UNIFORMS, SPVC_FALSE);
    spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_GLSL_EMIT_PUSH_CONSTANT_AS_UNIFORM_BUFFER, SPVC_FALSE);
    spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_GLSL_SUPPORT_NONZERO_BASE_INSTANCE, SPVC_TRUE);
    spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_GLSL_SEPARATE_SHADER_OBJECTS, SPVC_TRUE);

    spvc_compiler_install_compiler_options(compiler_glsl, options);

    if (spvc_compiler_compile(compiler_glsl, &result) != SPVC_SUCCESS) {
        LOG_I("[MobileGLES] spirv-cross: Compilation failed: %s", spvc_context_get_last_error_string(context));
        spvc_context_destroy(context);
        errc = -1;
        return "";
    }

    if (!result) {
        LOG_I("[MobileGLES] Error: unexpected error in spirv-cross.")
        spvc_context_destroy(context);
        errc = -1;
        return "";
    }

    std::string essl(result);
    spvc_context_destroy(context);
    errc = 0;
    return essl;
}

// ============================================================================
// Section 14: Stage 4 — ESSL Post-processing
// ============================================================================

static void add_required_extensions(std::string& essl, GLenum shader_type, uint essl_version) {
    const char* ext = nullptr;
    if (shader_type == GL_GEOMETRY_SHADER) {
        ext = "#extension GL_EXT_geometry_shader : require\n";
    } else if (shader_type == GL_TESS_CONTROL_SHADER || shader_type == GL_TESS_EVALUATION_SHADER) {
        ext = "#extension GL_EXT_tessellation_shader : require\n";
    }
    if (ext) {
        size_t nl = essl.find('\n');
        if (nl != std::string::npos) essl.insert(nl + 1, ext);
    }
}

static std::string postprocess_essl(std::string essl, GLenum shader_type, uint essl_version) {
    if (shader_type != GL_COMPUTE_SHADER) {
        essl = removeLayoutBinding(essl);
    }
    essl = processOutColorLocations(essl);
    essl = forceSupporterOutput(essl);
    add_required_extensions(essl, shader_type, essl_version);
    return essl;
}

// ============================================================================
// Section 15: Main Conversion Entry Points
// ============================================================================

static std::once_flag glslang_init_flag;

void init_glslang_once() {
    std::call_once(glslang_init_flag, [] { glslang::InitializeProcess(); });
}

std::string GLSLtoGLSLES_2(const char* glsl_code, GLenum glsl_type, uint essl_version, int& return_code) {
    bool atomicCounterEmulated = false;

    std::string corrected = preprocess_glsl(glsl_code, glsl_type, &atomicCounterEmulated);
    LOG_D("Preprocessed GLSL:\n%s", corrected.c_str())

    int glsl_version = get_or_add_glsl_version(corrected);

    init_glslang_once();

    const char* src[] = {corrected.c_str()};
    int errc = 0;
    std::vector<unsigned int> spirv = glsl_to_spirv(glsl_type, glsl_version, src, errc);
    if (errc != 0) { return_code = -1; return ""; }

    errc = 0;
    std::string essl = spirv_to_essl(std::move(spirv), essl_version, glsl_type, errc);
    if (errc != 0) { return_code = -2; return ""; }

    essl = postprocess_essl(std::move(essl), glsl_type, essl_version);

    LOG_D("GLSL to GLSL ES Complete: \n%s", essl.c_str())

    return_code = atomicCounterEmulated ? 1 : 0;
    return essl;
}

std::string GLSLtoGLSLES(const char* glsl_code, GLenum glsl_type, uint essl_version, uint glsl_version,
                         int& return_code) {
    // Build cache suffix on stack (no heap allocation).
    // Suffix format: \n//<glsl_type>|<MAJOR>.<MINOR>.<REVISION>|<essl_version>
    char suffix[64];
    snprintf(suffix, sizeof(suffix), "\n//%u|%d.%d.%d|%u",
             (unsigned)glsl_type, MAJOR, MINOR, REVISION, essl_version);

    // Compute hash incrementally over glsl_code + suffix.
    // Avoids copying the full GLSL source into a std::string key on every call.
    auto hash = Cache::computeSHA256Parts(glsl_code, suffix);

    int cached_rc = -1;
    const char* cached = Cache::get_instance().getByHash(hash, &cached_rc);
    if (cached) {
        LOG_D("GLSL Hit Cache:\n%s\n-->\n%s", glsl_code, cached)
        return_code = cached_rc;
        return (char*)cached;
    }

    return_code = -1;
    std::string converted = GLSLtoGLSLES_2(glsl_code, glsl_type, essl_version, return_code);
    if (return_code >= 0 && !converted.empty()) {
        converted = process_uniform_declarations(converted);
        Cache::get_instance().putByHash(hash, converted.c_str(), return_code);
    }

    return (return_code >= 0) ? converted : glsl_code;
}