// MobileGlues - gl/envvars.h
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header
#ifndef _MOBILEGLUES_ENVVARS_H_
#define _MOBILEGLUES_ENVVARS_H_

#ifdef __cplusplus
extern "C"
{
#endif

    const char* GetEnvVar(const char* name);

    int GetEnvVarInt(const char* name, int* i, int def);

#ifdef __cplusplus
}
#endif

#endif
