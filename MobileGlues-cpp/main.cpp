// MobileGlues - main.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

#include "config/config.h"
#include "config/settings.h"
#include "egl/egl.h"
#include "egl/loader.h"
#include "gl/envvars.h"
#include "gl/gl.h"
#include "gl/glsl/glsl_for_es.h"
#include "gl/log.h"
#include "gl/mg.h"
#include "gles/loader.h"
#include "includes.h"
#include <cerrno>
#include <cstring>
#include <sys/stat.h>

#define DEBUG 0

__attribute__((used))
const char* license = "GNU LGPL-2.1 License";

void init_config() {
    if (check_path()) config_refresh();
}

void show_license() {
    LOG_V("The Open Source License of MobileGlues: ");
    LOG_V("  %s", license);
}

#if PROFILING

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

void init_perfetto() {
    perfetto::TracingInitArgs args;

    args.backends |= perfetto::kSystemBackend;

    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();
}
#endif

void proc_init() {
    init_config();

    clear_log();
    start_log();

    LOG_V("Initializing %s ...", RENDERERNAME);
    show_license();

    init_settings();

    load_libs();
    init_target_egl();
    init_target_gles();
    set_multidraw_setting();

    init_settings_post();

    // Pre-initialize glslang so the one-time symbol-table build does not land
    // on the first glShaderSource call (which would stack with the first
    // shader compile and cause a noticeable CPU spike on the first frame).
    init_glslang_once();

#if PROFILING
    init_perfetto();
#endif

    // Cleanup
    destroy_temp_egl_ctx();
    g_initialized = 1;
}
