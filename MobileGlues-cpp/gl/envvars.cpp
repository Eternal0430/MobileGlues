// MobileGlues - gl/envvars.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header
#include "envvars.h"
#include "glext.h"
#include "../includes.h"
#include <cstdio>
#include <cstdlib>

const char* GetEnvVar(const char* name) {
    return getenv(name);
}

int GetEnvVarInt(const char* name, int* i, int def) {
    const char* s = GetEnvVar(name);
    *i = s ? atoi(s) : def;
    return s != NULL;
}
