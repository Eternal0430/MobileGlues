// MobileGlues - gl/vertexattrib.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

// ============================================================================
// Vertex Attribute Architecture
//
// Principle: "ES 3.2 native → native, ES 3.2 not native → CPU simulation"
//
// ES 3.2 natively supports:
//   - glVertexAttribI4ui, glVertexAttribI4i (integer vertex attributes with 4 components)
// ============================================================================

#include "vertexattrib.h"

#define DEBUG 0