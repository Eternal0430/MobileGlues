// MobileGlues - gl/pixel.h
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

#ifndef MOBILEGLUES_PIXEL_H
#define MOBILEGLUES_PIXEL_H

#include <GL/gl.h>
#include "../gles/gles.h"
#include "log.h"

#ifdef __BIG_ENDIAN__
#define GL_INT8_REV GL_UNSIGNED_INT_8_8_8_8
#define GL_INT8 GL_UNSIGNED_INT_8_8_8_8_REV
#else
#define GL_INT8_REV GL_UNSIGNED_INT_8_8_8_8_REV
#define GL_INT8 GL_UNSIGNED_INT_8_8_8_8
#endif

#define widthalign(width, align) ((((uintptr_t)(width)) + ((uintptr_t)(align) - 1)) & (~((uintptr_t)(align) - 1)))

GLsizei gl_sizeof(GLenum type);

GLsizei pixel_sizeof(GLenum format, GLenum type);

GLboolean is_type_packed(GLenum type);

// ============================================================================
// BGRA <=> RGBA Swizzle Helpers
//
// Design adapted from MobileGL-DirectGLES branch (MG_Util/Texture/PixelStoreProcessor).
// GLES 3.2 does not support GL_BGRA / GL_UNSIGNED_INT_8_8_8_8 / GL_UNSIGNED_INT_8_8_8_8_REV
// as pixel transfer formats. Instead of silently rewriting the format enum and
// uploading garbled bytes, we CPU-swizzle the pixel data so that GLES can be
// called with GL_RGBA + GL_UNSIGNED_BYTE (for uploads) or so that user-requested
// BGRA layouts can be produced from a GL_RGBA + GL_UNSIGNED_BYTE readback.
//
// The swizzle is expressed as a 4-entry table of source-channel selectors that
// describes, for each output byte position, which input byte to copy.
// ============================================================================

// Source-channel selectors used by ProcessColorSwizzle.
// Values are intentionally 0..3 = R/G/B/A byte offsets within a pixel, plus
// 4/5 for the constant ZERO/ONE values.
enum SwizzleChannel : unsigned char {
    SWIZZLE_RED   = 0,
    SWIZZLE_GREEN = 1,
    SWIZZLE_BLUE  = 2,
    SWIZZLE_ALPHA = 3,
    SWIZZLE_ZERO  = 4,
    SWIZZLE_ONE   = 5,
};

// In-place 8-bit-per-channel color swizzle.
//   data       : pointer to the first pixel
//   pixelCount : number of pixels to process
//   swizzle    : array of `channels` SwizzleChannel selectors, one per output
//                byte position; swizzle[ch] describes which source channel
//                is written to output byte ch
//   channels   : number of channels per pixel (1..4)
// Assumes each channel is exactly 1 byte (RGBA8/BGRA8/etc.).
void ProcessColorSwizzle(void* data, GLuint pixelCount, const unsigned char* swizzle, int channels);

// Combined copy+swizzle: copies `pixelCount` pixels from `src` to `dst` while
// applying a 4-channel byte swizzle. Equivalent to memcpy then
// ProcessColorSwizzle, but a single pass - faster for the common BGRA->RGBA
// case on the hot texture-upload path.
// `srcStride` and `dstStride` are in bytes per row; pass equal strides for
// tightly packed 4-byte-per-pixel data.
void CopyAndSwizzleRGBA8(void* dst, GLsizei dstStride,
                          const void* src, GLsizei srcStride,
                          GLsizei width, GLsizei height, GLsizei depth,
                          GLsizei srcImageStride,
                          const unsigned char swizzle[4]);

// Determines whether a CPU swizzle is required when uploading pixel data of
// (srcFormat, srcType) into an RGBA8 texture. If so, fills `out` with the
// 4-channel swizzle that converts the in-memory bytes to GL_RGBA +
// GL_UNSIGNED_BYTE layout, and returns true. Returns false when the source
// layout already matches RGBA + UNSIGNED_BYTE (no copy/swizzle needed).
//
// Returned GLES transfer format/type after applying the swizzle is always
// (GL_RGBA, GL_UNSIGNED_BYTE).
bool get_rgba8_unpack_swizzle(GLenum srcFormat, GLenum srcType, unsigned char out[4]);

// Determines whether a CPU swizzle is required when reading back a GL_RGBA +
// GL_UNSIGNED_BYTE image into the user-requested (dstFormat, dstType) layout.
// If so, fills `out` with the 4-channel swizzle and returns true.
// Returns false when the requested layout is already RGBA + UNSIGNED_BYTE
// (no swizzle needed, can write directly to user buffer).
bool get_rgba_pack_swizzle(GLenum dstFormat, GLenum dstType, unsigned char out[4]);

#endif // MOBILEGLUES_PIXEL_H
