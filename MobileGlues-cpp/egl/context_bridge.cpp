// MobileGlues - egl/context_bridge.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

// Bridge definition for the per-context pointer the ported upstream files
// reference (gl/enable.cpp, gl/multidraw.cpp, gl/pixel.cpp).
//
// The full context tracker lives in egl/context.cpp, which cannot join the
// build yet: it drives upstream's thread_local `gl_state` pointer model, and
// this fork's gl/mg.h still `#define`s that name as a GLStateManager alias.
// Until that collision is resolved and context.cpp is added to CMakeLists,
// this translation unit supplies the one symbol the linker needs so the
// library links with --no-undefined (the Android NDK toolchain default).
//
// Every reader of g_current_ctx in the build tree handles nullptr by falling
// back to process-global state, which keeps behaviour identical to the
// pre-port fork until real context tracking lands.

#include "context.h"

thread_local MGContext* g_current_ctx = nullptr;
