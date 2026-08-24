// MobileGlues - gl/pixel.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

// ============================================================================
// Pixel Format Conversion Architecture
//
// Principle: "ES 3.2 native → native, ES 3.2 not native → CPU simulation"
//
// This file implements CPU-side pixel format conversion utilities needed
// because OpenGL supports more pixel formats than OpenGL ES 3.2.
//
// When a desktop OpenGL pixel format is not natively supported by ES 3.2,
// the pixel data is converted on the CPU before being passed to the native
// GLES implementation. This is CPU simulation, not GPU simulation.
//
// Supported conversions:
//   - Format size queries (gl_sizeof, pixel_sizeof)
//   - Packed type detection (is_type_packed)
//   - BGRA <-> RGBA swizzle helpers for texture upload / readback
// ============================================================================

#include "pixel.h"
#include "log.h"
#include "mg.h"

#define DEBUG 0

// ============================================================================
// Section: Type Size Query
// ============================================================================

// Sorted: GL type → size in bytes
struct TypeSizeEntry {
    GLenum key;
    GLsizei value;
};

static const TypeSizeEntry kTypeSizeTable[] = {
    {GL_BYTE,                          1},
    {GL_UNSIGNED_BYTE,                 1},
    {GL_SHORT,                         2},
    {GL_UNSIGNED_SHORT,                2},
    {GL_INT,                           4},
    {GL_UNSIGNED_INT,                  4},
    {GL_FLOAT,                         4},
    {GL_2_BYTES,                       2},
    {GL_3_BYTES,                       3},
    {GL_4_BYTES,                       4},
    {GL_DOUBLE,                        8},
    {GL_HALF_FLOAT,                    2},
    {GL_COLOR_INDEX,                   1},
    {GL_DEPTH_COMPONENT,               1},
    {GL_ALPHA,                         1},
    {GL_LUMINANCE,                     1},
    {GL_LUMINANCE_ALPHA,               2},
    {GL_UNSIGNED_BYTE_3_3_2,           1},
    {GL_UNSIGNED_SHORT_4_4_4_4,        2},
    {GL_UNSIGNED_SHORT_5_5_5_1,        2},
    {GL_UNSIGNED_INT_8_8_8_8,          4},
    {GL_UNSIGNED_BYTE_2_3_3_REV,       1},
    {GL_UNSIGNED_SHORT_5_6_5,          2},
    {GL_UNSIGNED_SHORT_5_6_5_REV,      2},
    {GL_UNSIGNED_SHORT_4_4_4_4_REV,    2},
    {GL_UNSIGNED_SHORT_1_5_5_5_REV,    2},
    {GL_UNSIGNED_INT_8_8_8_8_REV,      4},
    {GL_UNSIGNED_INT_10_10_10_2,       4},
    {GL_UNSIGNED_INT_2_10_10_10_REV,   4},
    {GL_UNSIGNED_INT_24_8,             4},
};
static constexpr size_t kTypeSizeTableCount = sizeof(kTypeSizeTable) / sizeof(kTypeSizeTable[0]);

GLsizei gl_sizeof(GLenum type) {
    size_t lo = 0, hi = kTypeSizeTableCount;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (kTypeSizeTable[mid].key < type) lo = mid + 1;
        else hi = mid;
    }
    if (lo < kTypeSizeTableCount && kTypeSizeTable[lo].key == type) [[likely]]
        return kTypeSizeTable[lo].value;
    LOG_D("Unsupported pixel data type: %s\n", glEnumToString(type))
    return 0;
}

// ============================================================================
// Section: Packed Type Detection
// ============================================================================

// Sorted: packed type entries for binary search
static const GLenum kPackedTypes[] = {
    GL_2_BYTES,
    GL_3_BYTES,
    GL_4_BYTES,
    GL_UNSIGNED_BYTE_3_3_2,
    GL_UNSIGNED_SHORT_4_4_4_4,
    GL_UNSIGNED_SHORT_5_5_5_1,
    GL_UNSIGNED_INT_8_8_8_8,
    GL_UNSIGNED_BYTE_2_3_3_REV,
    GL_UNSIGNED_SHORT_5_6_5,
    GL_UNSIGNED_SHORT_5_6_5_REV,
    GL_UNSIGNED_SHORT_4_4_4_4_REV,
    GL_UNSIGNED_SHORT_1_5_5_5_REV,
    GL_UNSIGNED_INT_8_8_8_8_REV,
    GL_UNSIGNED_INT_10_10_10_2,
    GL_UNSIGNED_INT_2_10_10_10_REV,
    GL_DEPTH_STENCIL,
};
static constexpr size_t kPackedTypeCount = sizeof(kPackedTypes) / sizeof(kPackedTypes[0]);

GLboolean is_type_packed(GLenum type) {
    size_t lo = 0, hi = kPackedTypeCount;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (kPackedTypes[mid] < type) lo = mid + 1;
        else hi = mid;
    }
    return (lo < kPackedTypeCount && kPackedTypes[lo] == type);
}

// ============================================================================
// Section: Pixel Size Calculation
// ============================================================================

// Sorted: format → component count
static const TypeSizeEntry kFormatWidthTable[] = {
    {GL_COLOR_INDEX,     1},
    {GL_DEPTH_COMPONENT, 1},
    {GL_RED,             1},
    {GL_ALPHA,           1},
    {GL_RGB,             3},
    {GL_RGBA,            4},
    {GL_LUMINANCE,       1},
    {GL_LUMINANCE_ALPHA, 2},
    {GL_R,               1},
    {GL_RGB8,            3},
    {GL_RGBA8,           4},
    {GL_BGR,             3},
    {GL_BGRA,            4},
    {GL_RG,              2},
    {GL_R32F,            4},
    {GL_DEPTH_STENCIL,   1},
    {GL_R11F_G11F_B10F,  4},
};
static constexpr size_t kFormatWidthTableCount = sizeof(kFormatWidthTable) / sizeof(kFormatWidthTable[0]);

GLsizei pixel_sizeof(GLenum format, GLenum type) {
    GLsizei width = 0;
    size_t lo = 0, hi = kFormatWidthTableCount;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (kFormatWidthTable[mid].key < format) lo = mid + 1;
        else hi = mid;
    }
    if (lo < kFormatWidthTableCount && kFormatWidthTable[lo].key == format) [[likely]] {
        width = kFormatWidthTable[lo].value;
    } else {
        LOG_D("unsupported pixel format %s\n", glEnumToString(format))
        return 0;
    }

    if (is_type_packed(type)) width = 1;

    return width * gl_sizeof(type);
}

// ============================================================================
// Section: Color Channel Layout Mapping (removed)
// ----------------------------------------------------------------------------
// The legacy pixel_convert() and its get_color_map() helper have been removed
// (dead code, no callers in the codebase). BGRA <-> RGBA channel conversion
// is now handled by the swizzle helpers below (CopyAndSwizzleRGBA8 +
// get_rgba8_unpack_swizzle / get_rgba_pack_swizzle), which are tighter and
// used on the actual texture upload / readback hot paths.
// ============================================================================


// ============================================================================
// Section: BGRA <=> RGBA Swizzle Helpers
//
// Implements the in-place 8-bit-per-channel swizzle and the lookup helpers
// that decide whether a swizzle is needed for an upload (unpack) or readback
// (pack) operation. The semantics mirror the MobileGL-DirectGLES branch's
// MG_Util/Texture/PixelStoreProcessor: the swizzle table describes, for each
// output byte position, which source byte to copy.
//
// Memory layout reference (little-endian, the only target we ship to):
//   GL_BGRA + GL_UNSIGNED_BYTE                => [B, G, R, A]
//   GL_BGRA + GL_UNSIGNED_INT_8_8_8_8         => [A, R, G, B]   (msb=B)
//   GL_BGRA + GL_UNSIGNED_INT_8_8_8_8_REV     => [B, G, R, A]   (lsb=B)
//   GL_RGBA + GL_UNSIGNED_BYTE                => [R, G, B, A]
//   GL_RGBA + GL_UNSIGNED_INT_8_8_8_8         => [A, B, G, R]   (msb=R)
//   GL_RGBA + GL_UNSIGNED_INT_8_8_8_8_REV     => [R, G, B, A]   (lsb=R)
//
// GLES only accepts GL_RGBA + GL_UNSIGNED_BYTE, so we always feed it that and
// do the channel rearrangement on the CPU.
// ============================================================================

void ProcessColorSwizzle(void* data, GLuint pixelCount, const unsigned char* swizzle, int channels) {
    if (!data || !swizzle || pixelCount == 0 || channels <= 0 || channels > 4) return;
    unsigned char* bytes = static_cast<unsigned char*>(data);
    unsigned char scratch[4];
    for (GLuint i = 0; i < pixelCount; ++i) {
        unsigned char* pixel = bytes + i * channels;
        for (int ch = 0; ch < channels; ++ch) {
            unsigned char s = swizzle[ch];
            switch (s) {
                case SWIZZLE_RED:   scratch[ch] = pixel[0]; break;
                case SWIZZLE_GREEN: scratch[ch] = pixel[1]; break;
                case SWIZZLE_BLUE:  scratch[ch] = pixel[2]; break;
                case SWIZZLE_ALPHA: scratch[ch] = pixel[3]; break;
                case SWIZZLE_ZERO:  scratch[ch] = 0x00;     break;
                case SWIZZLE_ONE:   scratch[ch] = 0xFF;     break;
                default:            scratch[ch] = 0xBD;     break;
            }
        }
        for (int ch = 0; ch < channels; ++ch) pixel[ch] = scratch[ch];
    }
}

// Combined copy + 4-channel byte swizzle in a single pass.
//
// Detects the two most common swizzle patterns emitted by
// get_rgba8_unpack_swizzle and dispatches to a tight specialised loop:
//   - BGRA->RGBA (R/B swap): {Blue, Green, Red, Alpha}
//   - GL_UNSIGNED_INT_8_8_8_8 BGRA->RGBA: {Green, Blue, Alpha, Red}
// Both avoid the per-channel switch of the generic in-place swizzle and
// let the compiler vectorise the inner loop. Falls back to the generic
// in-place algorithm (copy + ProcessColorSwizzle) for arbitrary swizzles.
//
// `dstStride`/`srcStride` are row strides in bytes. `srcImageStride` is the
// 3D image stride in bytes; for 2D (depth==1) it is ignored.
void CopyAndSwizzleRGBA8(void* dst, GLsizei dstStride,
                          const void* src, GLsizei srcStride,
                          GLsizei width, GLsizei height, GLsizei depth,
                          GLsizei srcImageStride,
                          const unsigned char swizzle[4]) {
    if (!dst || !src || width <= 0 || height <= 0 || depth <= 0) return;
    const unsigned char* s = static_cast<const unsigned char*>(src);
    unsigned char* d = static_cast<unsigned char*>(dst);
    const GLsizei rowBytes = width * 4;

    // Fast path: BGRA -> RGBA = swap bytes 0 and 2, keep G and A.
    // swizzle == {SWIZZLE_BLUE, SWIZZLE_GREEN, SWIZZLE_RED, SWIZZLE_ALPHA}
    if (swizzle[0] == SWIZZLE_BLUE  && swizzle[1] == SWIZZLE_GREEN &&
        swizzle[2] == SWIZZLE_RED   && swizzle[3] == SWIZZLE_ALPHA) {
        // Hint: pointers are 4-byte aligned (each pixel is 4 bytes), which
        // helps the compiler vectorise the inner loop.
        const unsigned char* sp = static_cast<const unsigned char*>(__builtin_assume_aligned(s, 4));
        unsigned char* dp = static_cast<unsigned char*>(__builtin_assume_aligned(d, 4));
        for (GLsizei z = 0; z < depth; ++z) {
            const unsigned char* srcRow = sp;
            unsigned char* dstRow = dp;
            for (GLsizei y = 0; y < height; ++y) {
                for (GLsizei i = 0; i < width; ++i) {
                    const unsigned char* p = srcRow + i * 4;
                    unsigned char* o = dstRow + i * 4;
                    unsigned char b = p[0];   // B
                    o[1] = p[1];              // G
                    o[0] = p[2];              // R <- swap
                    o[2] = b;                 // B <- swap
                    o[3] = p[3];              // A
                }
                srcRow += srcStride;
                dstRow += dstStride;
            }
            sp += srcImageStride;
        }
        return;
    }

    // Fast path: GL_UNSIGNED_INT_8_8_8_8 BGRA -> RGBA.
    // swizzle == {SWIZZLE_GREEN, SWIZZLE_BLUE, SWIZZLE_ALPHA, SWIZZLE_RED}
    // Memory layout [A,R,G,B] -> output [R,G,B,A]
    if (swizzle[0] == SWIZZLE_GREEN && swizzle[1] == SWIZZLE_BLUE &&
        swizzle[2] == SWIZZLE_ALPHA && swizzle[3] == SWIZZLE_RED) {
        const unsigned char* sp = static_cast<const unsigned char*>(__builtin_assume_aligned(s, 4));
        unsigned char* dp = static_cast<unsigned char*>(__builtin_assume_aligned(d, 4));
        for (GLsizei z = 0; z < depth; ++z) {
            const unsigned char* srcRow = sp;
            unsigned char* dstRow = dp;
            for (GLsizei y = 0; y < height; ++y) {
                for (GLsizei i = 0; i < width; ++i) {
                    const unsigned char* p = srcRow + i * 4;
                    unsigned char* o = dstRow + i * 4;
                    o[0] = p[1];   // R
                    o[1] = p[2];   // G
                    o[2] = p[3];   // B
                    o[3] = p[0];   // A
                }
                srcRow += srcStride;
                dstRow += dstStride;
            }
            sp += srcImageStride;
        }
        return;
    }

    // Generic fallback: copy row-by-row then in-place swizzle.
    for (GLsizei z = 0; z < depth; ++z) {
        const unsigned char* srcRow = s;
        unsigned char* dstRow = d;
        for (GLsizei y = 0; y < height; ++y) {
            memcpy(dstRow, srcRow, rowBytes);
            ProcessColorSwizzle(dstRow, static_cast<GLuint>(width), swizzle, 4);
            srcRow += srcStride;
            dstRow += dstStride;
        }
        s += srcImageStride;
    }
}

bool get_rgba8_unpack_swizzle(GLenum srcFormat, GLenum srcType, unsigned char out[4]) {
    // GL_BGRA sources: the in-memory byte order differs from GL_RGBA, so a
    // channel rearrangement is required before GLES accepts the data as
    // GL_RGBA + GL_UNSIGNED_BYTE.
    if (srcFormat == GL_BGRA) {
        if (srcType == GL_UNSIGNED_INT_8_8_8_8) {
            // [A, R, G, B] -> [R, G, B, A]
            out[0] = SWIZZLE_GREEN;  // R <- src.R   (src byte 1)
            out[1] = SWIZZLE_BLUE;   // G <- src.G   (src byte 2)
            out[2] = SWIZZLE_ALPHA;  // B <- src.B   (src byte 3)
            out[3] = SWIZZLE_RED;    // A <- src.A   (src byte 0)
            return true;
        }
        // GL_UNSIGNED_BYTE: [B, G, R, A] -> [R, G, B, A]  (swap R/B)
        // GL_UNSIGNED_INT_8_8_8_8_REV: [B, G, R, A] -> [R, G, B, A]  (swap R/B)
        out[0] = SWIZZLE_BLUE;
        out[1] = SWIZZLE_GREEN;
        out[2] = SWIZZLE_RED;
        out[3] = SWIZZLE_ALPHA;
        return true;
    }

    if (srcFormat == GL_RGBA) {
        if (srcType == GL_UNSIGNED_INT_8_8_8_8) {
            // [A, B, G, R] -> [R, G, B, A]
            out[0] = SWIZZLE_ALPHA;
            out[1] = SWIZZLE_BLUE;
            out[2] = SWIZZLE_GREEN;
            out[3] = SWIZZLE_RED;
            return true;
        }
        // GL_UNSIGNED_BYTE: already [R, G, B, A] - no work needed.
        // GL_UNSIGNED_INT_8_8_8_8_REV: already [R, G, B, A] - no work needed.
    }

    return false;
}

bool get_rgba_pack_swizzle(GLenum dstFormat, GLenum dstType, unsigned char out[4]) {
    // Start from the identity mapping; the readback always produces
    // [R, G, B, A] in memory.
    out[0] = SWIZZLE_RED;
    out[1] = SWIZZLE_GREEN;
    out[2] = SWIZZLE_BLUE;
    out[3] = SWIZZLE_ALPHA;

    bool needSwizzle = false;

    if (dstFormat == GL_BGRA) {
        // Want [B, G, R, A] from [R, G, B, A]: swap R/B.
        out[0] = SWIZZLE_BLUE;
        out[1] = SWIZZLE_GREEN;
        out[2] = SWIZZLE_RED;
        out[3] = SWIZZLE_ALPHA;
        needSwizzle = true;
    }

    if (dstType == GL_UNSIGNED_INT_8_8_8_8) {
        // _REV variants are byte-for-byte identical to GL_UNSIGNED_BYTE on
        // little-endian, but the non-REV packed types reverse the byte order.
        // Reverse the swizzle to write [component4, component3, component2, component1]
        // (i.e. msb-first) in memory.
        unsigned char tmp;
        tmp = out[0]; out[0] = out[3]; out[3] = tmp;
        tmp = out[1]; out[1] = out[2]; out[2] = tmp;
        needSwizzle = true;
    }

    return needSwizzle;
}