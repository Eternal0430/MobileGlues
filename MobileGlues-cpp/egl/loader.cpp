// MobileGlues - egl/loader.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

#include "loader.h"
#include "../gl/envvars.h"
#include "../gl/log.h"
#include "../gl/mg.h"
#include "../gles/loader.h"
#include "../includes.h"
#include <EGL/egl.h>
#include <string.h>

#define DEBUG 0

static EGLDisplay eglDisplay = EGL_NO_DISPLAY;
static EGLSurface eglSurface = EGL_NO_SURFACE;
static EGLContext eglContext = EGL_NO_CONTEXT;

// Kept so that per-thread fallback contexts can be created later with the same
// configuration. The primary context above is created during init and is kept
// bound to whichever thread ran that init; other threads need their own.
static EGLConfig g_pbuf_config = nullptr;
static bool g_pbuf_config_valid = false;

// Set once init_target_egl() succeeds. The context is intentionally kept alive
// (see the note in loader.h) so that host GLES queries can be answered on
// threads that have no current context of their own.
static bool g_fallback_context_ready = false;

void init_target_egl() {
    LOAD_EGL(eglGetProcAddress);
    LOAD_EGL(eglBindAPI);
    LOAD_EGL(eglInitialize);
    LOAD_EGL(eglGetDisplay);
    LOAD_EGL(eglCreatePbufferSurface);
    LOAD_EGL(eglDestroySurface);
    LOAD_EGL(eglDestroyContext);
    LOAD_EGL(eglMakeCurrent);
    LOAD_EGL(eglChooseConfig);
    LOAD_EGL(eglCreateContext);
    LOAD_EGL(eglQueryString);
    LOAD_EGL(eglTerminate);
    LOAD_EGL(eglGetError);

    EGLint configAttribs[] = {EGL_RED_SIZE,
                              8,
                              EGL_GREEN_SIZE,
                              8,
                              EGL_BLUE_SIZE,
                              8,
                              EGL_ALPHA_SIZE,
                              8,
                              EGL_SURFACE_TYPE,
                              EGL_PBUFFER_BIT,
                              EGL_RENDERABLE_TYPE,
                              EGL_OPENGL_ES2_BIT,
                              EGL_NONE};

    EGLint ctxAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};

    EGLint pbAttribs[] = {EGL_WIDTH, 32, EGL_HEIGHT, 32, EGL_NONE};

    EGLConfig pbufConfig;
    EGLint configsFound = 0;

    eglDisplay = egl_eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (eglDisplay == EGL_NO_DISPLAY) {
        LOG_E("eglGetDisplay failed (0x%x)", egl_eglGetError());
        goto cleanup;
    }

    if (egl_eglInitialize(eglDisplay, NULL, NULL) != EGL_TRUE) {
        LOG_E("eglInitialize failed (0x%x)", egl_eglGetError());
        goto cleanup;
    }

    if (egl_eglBindAPI(EGL_OPENGL_ES_API) != EGL_TRUE) {
        LOG_E("eglBindAPI failed (0x%x)", egl_eglGetError());
        goto cleanup;
    }

    if (egl_eglChooseConfig(eglDisplay, configAttribs, &pbufConfig, 1, &configsFound) != EGL_TRUE) {
        LOG_E("eglChooseConfig failed (0x%x)", egl_eglGetError());
        goto cleanup;
    }

    if (configsFound == 0) {
        configAttribs[6] = 0;
        if (egl_eglChooseConfig(eglDisplay, configAttribs, &pbufConfig, 1, &configsFound) != EGL_TRUE) {
            LOG_E("Retry eglChooseConfig failed (0x%x)", egl_eglGetError());
            goto cleanup;
        }
        if (configsFound) {
            LOG_D("Using config without alpha channel");
        } else {
            LOG_E("No valid EGL config found");
            goto cleanup;
        }
    }
    g_pbuf_config = pbufConfig;
    g_pbuf_config_valid = true;

    eglContext = egl_eglCreateContext(eglDisplay, pbufConfig, EGL_NO_CONTEXT, ctxAttribs);
    if (eglContext == EGL_NO_CONTEXT) {
        LOG_E("eglCreateContext failed (0x%x)", egl_eglGetError());
        goto cleanup;
    }

    eglSurface = egl_eglCreatePbufferSurface(eglDisplay, pbufConfig, pbAttribs);
    if (eglSurface == EGL_NO_SURFACE) {
        LOG_E("eglCreatePbufferSurface failed (0x%x)", egl_eglGetError());
        goto cleanup;
    }

    if (egl_eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext) != EGL_TRUE) {
        LOG_E("eglMakeCurrent failed (0x%x)", egl_eglGetError());
        goto cleanup;
    }

    LOG_V("EGL initialized successfully");
    g_fallback_context_ready = true;
    return;

cleanup:
    if (eglSurface != EGL_NO_SURFACE) {
        egl_eglDestroySurface(eglDisplay, eglSurface);
    }
    if (eglContext != EGL_NO_CONTEXT) {
        egl_eglDestroyContext(eglDisplay, eglContext);
    }
    if (eglDisplay != EGL_NO_DISPLAY) {
        egl_eglTerminate(eglDisplay);
    }
    LOG_E("EGL initialization failed");
}

void destroy_temp_egl_ctx() {
    LOAD_EGL(eglMakeCurrent);

    // Only unbind the context from this thread. The display/context/surface
    // objects themselves are kept alive: they serve as the fallback context for
    // host GLES string queries issued from threads without a current context
    // (see BindFallbackEGLContextIfNeeded below).
    egl_eglMakeCurrent(eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
}

// ---------------------------------------------------------------------------
// Per-thread fallback contexts
//
// Every thread that issues host GL calls without a current context gets its OWN
// pbuffer context, created on first need and then left current for the life of
// the thread. Two earlier designs both failed on real hardware, and the shape of
// each failure is what motivates this one:
//
//   1. One shared context, bound and released around every call ("borrowed").
//      Two eglMakeCurrent round-trips per GL entry point is not just slow — on
//      Adreno/turnip it corrupts state across calls. A session with this design
//      logged 15 glMapBufferRange failures (GL_INVALID_OPERATION, buffers that
//      demonstrably had storage), 1 glBufferStorage rejection and 14 texture
//      uploads with no texture bound, and the gui pipeline's vertex shader
//      failed to compile with an empty info log (the signature of a driver that
//      discarded the call outright).
//
//   2. One shared context, bound and kept ("sticky"). GL state survives, but an
//      EGLContext may be current on only one thread, so when the render thread
//      held it across CompletableFuture.join() the background executor's shader
//      compile could never acquire it. That is the hang, not a slow path.
//
// One context per thread has neither problem: nothing is shared, so nothing can
// contend, and the context stays current, so state is never torn down. The
// contexts share a share group so that buffers, textures, shaders and programs
// created on any of them remain visible to the others.
//
// It also makes the guards cheap. After the first call on a thread, every later
// guard is a single eglGetCurrentContext() that finds a context already current
// and does nothing — and it means the GL entry points that were never wrapped
// (glBindBuffer, glBindBufferBase, glGetBufferParameteriv, ...) still work,
// because the context is current whether or not they asked for it. That gap is
// precisely what broke design 1.
// ---------------------------------------------------------------------------
namespace {

struct ThreadFallback {
    EGLContext ctx = EGL_NO_CONTEXT;
    EGLSurface surf = EGL_NO_SURFACE;
    bool tried = false;

    ~ThreadFallback() {
        if (ctx == EGL_NO_CONTEXT) return;
        // Runs at thread exit. The display may already be gone; the calls are
        // best-effort and a failure here is harmless.
        LOAD_EGL(eglGetCurrentContext);
        LOAD_EGL(eglMakeCurrent);
        LOAD_EGL(eglDestroySurface);
        LOAD_EGL(eglDestroyContext);
        if (egl_eglGetCurrentContext && egl_eglGetCurrentContext() == ctx) {
            egl_eglMakeCurrent(eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        }
        if (egl_eglDestroySurface) egl_eglDestroySurface(eglDisplay, surf);
        if (egl_eglDestroyContext) egl_eglDestroyContext(eglDisplay, ctx);
        ctx = EGL_NO_CONTEXT;
        surf = EGL_NO_SURFACE;
    }
};

thread_local ThreadFallback t_fallback;

} // namespace

bool BindFallbackEGLContextIfNeeded() {
    if (!g_fallback_context_ready) return false;

    LOAD_EGL(eglGetCurrentContext);
    LOAD_EGL(eglMakeCurrent);
    LOAD_EGL(eglGetError);

    // Already have a context — nothing to do. This is the path every correctly
    // configured thread takes after the first call, and the only path a thread
    // with a real context ever takes.
    if (egl_eglGetCurrentContext() != EGL_NO_CONTEXT) return false;

    if (!t_fallback.tried) {
        t_fallback.tried = true;

        LOAD_EGL(eglCreateContext);
        LOAD_EGL(eglCreatePbufferSurface);

        if (g_pbuf_config_valid && egl_eglCreateContext && egl_eglCreatePbufferSurface) {
            const EGLint ctxAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
            const EGLint pbAttribs[] = {EGL_WIDTH, 32, EGL_HEIGHT, 32, EGL_NONE};

            // Share with the primary context so objects created on this thread
            // are visible to the others.
            EGLContext ctx = egl_eglCreateContext(eglDisplay, g_pbuf_config, eglContext, ctxAttribs);
            if (ctx != EGL_NO_CONTEXT) {
                EGLSurface surf = egl_eglCreatePbufferSurface(eglDisplay, g_pbuf_config, pbAttribs);
                if (surf != EGL_NO_SURFACE) {
                    t_fallback.ctx = ctx;
                    t_fallback.surf = surf;
                } else {
                    LOG_W_FORCE("BindFallbackEGLContext: eglCreatePbufferSurface failed (0x%x)", egl_eglGetError());
                    LOAD_EGL(eglDestroyContext);
                    egl_eglDestroyContext(eglDisplay, ctx);
                }
            } else {
                LOG_W_FORCE("BindFallbackEGLContext: eglCreateContext for this thread failed (0x%x)",
                            egl_eglGetError());
            }
        }
    }

    EGLContext ctx = t_fallback.ctx != EGL_NO_CONTEXT ? t_fallback.ctx : eglContext;
    EGLSurface surf = t_fallback.ctx != EGL_NO_CONTEXT ? t_fallback.surf : eglSurface;

    if (ctx == EGL_NO_CONTEXT) return false;

    if (egl_eglMakeCurrent(eglDisplay, surf, surf, ctx) != EGL_TRUE) {
        LOG_W_FORCE("BindFallbackEGLContext: eglMakeCurrent failed (0x%x) — host GL calls from this thread will be "
                    "silently discarded",
                    egl_eglGetError());
        return false;
    }

    LOG_W_FORCE("BindFallbackEGLContext: this thread had no current EGL context, so MobileGLES created a fallback "
                "pbuffer context for it and left it bound for the life of the thread. Every host GL call made "
                "from a context-less thread is discarded without error, which is how a shader compile or a "
                "buffer map fails with no explanation. This indicates the launcher never called eglMakeCurrent "
                "on this thread.");
    return true;
}

// Pairs a successful BindFallbackEGLContextIfNeeded().
//
// Deliberately does nothing: the context belongs to this thread alone, so there
// is nothing to give back. Releasing it would reintroduce both failures above —
// the per-call eglMakeCurrent churn that corrupts driver state, and the window
// in which another thread could take the context.
void UnbindFallbackEGLContext() {}