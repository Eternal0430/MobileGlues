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
#include <pthread.h>
#include <stdio.h>
#include <string.h>

#define DEBUG 0

static EGLDisplay eglDisplay = EGL_NO_DISPLAY;
static EGLContext eglContext = EGL_NO_CONTEXT;
// Not used. Kept as a named constant at the binding sites instead, so that the
// absence of a surface is visible where it matters: this context is
// surfaceless, see init_target_egl().
static const EGLSurface kNoSurface = EGL_NO_SURFACE;

// The config chosen during init, kept so that per-thread contexts can be
// created from it later. A companion context has to come from a config the
// driver considers compatible with the one the application's context uses.
static EGLConfig g_context_config = nullptr;
static bool g_context_config_valid = false;

// Set once init_target_egl() succeeds. The context is intentionally kept alive
// (see the note in loader.h) so that host GLES queries can be answered on
// threads that have no current context of their own.
static bool g_fallback_context_ready = false;

void init_target_egl() {
    LOAD_EGL(eglGetProcAddress);
    LOAD_EGL(eglBindAPI);
    LOAD_EGL(eglInitialize);
    LOAD_EGL(eglGetDisplay);
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
                              EGL_RENDERABLE_TYPE,
                              EGL_OPENGL_ES2_BIT,
                              EGL_NONE};

    EGLint ctxAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};

    EGLConfig config;
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

    if (egl_eglChooseConfig(eglDisplay, configAttribs, &config, 1, &configsFound) != EGL_TRUE) {
        LOG_E("eglChooseConfig failed (0x%x)", egl_eglGetError());
        goto cleanup;
    }

    if (configsFound == 0) {
        // EGL_ALPHA_SIZE, at index 6.
        configAttribs[6] = 0;
        if (egl_eglChooseConfig(eglDisplay, configAttribs, &config, 1, &configsFound) != EGL_TRUE) {
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

    g_context_config = config;
    g_context_config_valid = true;

    eglContext = egl_eglCreateContext(eglDisplay, config, EGL_NO_CONTEXT, ctxAttribs);
    if (eglContext == EGL_NO_CONTEXT) {
        LOG_E("eglCreateContext failed (0x%x)", egl_eglGetError());
        goto cleanup;
    }

    // Bind with no surface at all. EGL_KHR_surfaceless_context is required for
    // this, and it is available here — config/gpu_utils.cpp has been doing
    // exactly this on this device for the whole session: it is how
    // "Using graphics device: Adreno (TM) 619" got into the log, and that
    // string comes from a glGetString issued on a surfaceless context.
    //
    // No pbuffer is created anywhere in this library any more. A pbuffer is a
    // valid render target that is not connected to the display, which is
    // precisely the wrong thing to hand a thread that expects to be drawing to
    // the screen: every call succeeds and nothing is ever visible.
    if (egl_eglMakeCurrent(eglDisplay, kNoSurface, kNoSurface, eglContext) != EGL_TRUE) {
        LOG_E("eglMakeCurrent failed (0x%x)", egl_eglGetError());
        goto cleanup;
    }

    LOG_V("EGL initialized successfully");
    g_fallback_context_ready = true;
    return;

cleanup:
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

    // Only unbind the context from this thread. The display and the context
    // themselves are kept alive: they are the last-resort fallback for host GLES
    // queries issued before the application has created a context of its own
    // (see BindFallbackEGLContextIfNeeded below).
    egl_eglMakeCurrent(eglDisplay, kNoSurface, kNoSurface, EGL_NO_CONTEXT);
}
// ---------------------------------------------------------------------------
// Fallback contexts
//
// A host driver discards every GL call made from a thread with no current EGL
// context: no object is created, no error is raised, queries simply return 0 or
// NULL. That produced the NULL glGetString, the zeroed glGetIntegerv, the NULL
// glMapBufferRange and the zero maxAnisotropy seen earlier in this session.
//
// Which context to hand a context-less thread is the whole difficulty, and four
// designs were tried on this device before this one:
//
//   1. One shared pbuffer context, bound and released around every call. Two
//      eglMakeCurrent round-trips per GL entry point corrupts state on
//      Adreno/turnip: 15 glMapBufferRange failures on buffers that had storage,
//      a glBufferStorage rejection, 14 texture uploads with nothing bound, and
//      a vertex shader that failed to compile with an empty info log.
//
//   2. One shared pbuffer context, bound and kept. An EGLContext may be current
//      on only one thread, so the render thread holding it across
//      CompletableFuture.join() starved the background executor's shader
//      compile. The game hung.
//
//   3. One pbuffer context per thread, bound and kept. No contention, but a
//      pbuffer is a render target that is not connected to the display — every
//      draw succeeded and nothing was ever visible, while eglSwapBuffers
//      presented a window surface nothing had been drawn into. Black screen.
//
//   4. Bind the application's own context to its own window surface. Right idea
//      for the render thread, fatal everywhere else: the application's context
//      is one object, so the first context-less thread takes it and every
//      other thread is refused with EGL_BAD_ACCESS. Worse, the fallout is
//      silent — a thread that ends up with no context at all still "succeeds"
//      at every GL call.
//
// This one gives each thread a context of its own, so nothing is contended and
// nothing can deadlock, and creates it sharing with the application's context,
// so every object it makes is visible to the game. It binds with NO SURFACE:
// surfaceless contexts are supported here (config/gpu_utils.cpp has used one
// all session — "Adreno (TM) 619" was queried on one), and a surface is what
// made design 3 draw into something invisible. No pbuffer is created anywhere.
//
// The trade is that these contexts cannot present, which is fine: the thread
// that draws is the one already holding the application's context on the
// application's window surface, and that thread is left completely alone.
// ---------------------------------------------------------------------------
namespace {

struct ThreadFallback {
    EGLContext ctx = EGL_NO_CONTEXT;
    // What this context was created to share with. Rebuilt if the application's
    // context turns up later: a thread that arrived first would otherwise be
    // stranded on an unshared context, where anything it creates is invisible
    // to the game.
    EGLContext share_with = EGL_NO_CONTEXT;
    bool tried = false;
};

thread_local ThreadFallback t_fb;
thread_local unsigned t_seen_generation = 0;

// Identifies the thread in logs. Several threads reach the fallback, and which
// is which turned out to be the thing worth knowing.
//
// Deliberately not pthread_getname_np(): it is only declared from API 26 and
// this project builds for minSdk 21, where using it is a hard error rather than
// a warning. pthread_self() is available at every level, and it is the same
// value eglMakeCurrent logs for the application's binding, so the two lines can
// be compared directly to tell whether a fallback thread is the render thread.
const char* CurrentThreadLabel() {
    static thread_local char label[48];
    if (label[0] == '\0') snprintf(label, sizeof(label), "pthread=%lu", (unsigned long)pthread_self());
    return label;
}

// Creates a context for this thread and binds it with no surface.
bool CreateThreadContext(EGLDisplay dpy, EGLContext share) {
    LOAD_EGL(eglCreateContext);
    LOAD_EGL(eglDestroyContext);
    LOAD_EGL(eglMakeCurrent);
    LOAD_EGL(eglGetError);
    if (!egl_eglCreateContext || !g_context_config_valid) return false;

    // Sharing requires the driver to consider the two contexts compatible, and
    // the application's may be a later ES version than ours. Try the newest
    // first and work down.
    static const EGLint attrs_v3[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    static const EGLint attrs_v2[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    static const EGLint attrs_none[] = {EGL_NONE};
    const EGLint* candidates[] = {attrs_v3, attrs_v2, attrs_none};

    for (const EGLint* attrs : candidates) {
        EGLContext ctx = egl_eglCreateContext(dpy, g_context_config, share, attrs);
        if (ctx == EGL_NO_CONTEXT) continue;
        if (egl_eglMakeCurrent(dpy, kNoSurface, kNoSurface, ctx) != EGL_TRUE) {
            egl_eglGetError();
            if (egl_eglDestroyContext) egl_eglDestroyContext(dpy, ctx);
            continue;
        }
        t_fb.ctx = ctx;
        t_fb.share_with = share;
        return true;
    }
    return false;
}

} // namespace

bool BindFallbackEGLContextIfNeeded() {
    if (!g_fallback_context_ready) return false;

    LOAD_EGL(eglGetCurrentContext);
    LOAD_EGL(eglMakeCurrent);
    LOAD_EGL(eglDestroyContext);
    LOAD_EGL(eglGetError);

    // Hot path: one atomic load. The record itself is behind a lock, so it is
    // only read when the generation says something actually changed.
    const unsigned gen = mg_egl_app_target_generation();
    if (gen != t_seen_generation) {
        t_seen_generation = gen;
        const AppRenderTarget& t = mg_egl_app_target();
        if (t.have_binding && t.context != EGL_NO_CONTEXT && t_fb.ctx != EGL_NO_CONTEXT &&
            t_fb.ctx != eglContext && t_fb.share_with != t.context) {
            const EGLContext old = t_fb.ctx;
            t_fb.ctx = EGL_NO_CONTEXT;
            if (CreateThreadContext(t.display, t.context)) {
                if (egl_eglDestroyContext) egl_eglDestroyContext(eglDisplay, old);
                LOG_W_FORCE("BindFallbackEGLContext: [%s] rebuilt this thread's context to share with the "
                            "application's context %p. It was created before that context existed, so anything "
                            "made on it was invisible to the game",
                            CurrentThreadLabel(), t.context);
            } else {
                t_fb.ctx = old;  // keep what we have rather than lose the context
            }
        }
    }

    // Already have a context. This is the path the render thread takes, and the
    // only path any correctly configured thread takes after its first call.
    if (egl_eglGetCurrentContext() != EGL_NO_CONTEXT) return false;

    if (t_fb.tried) return false;
    t_fb.tried = true;

    const AppRenderTarget& t = mg_egl_app_target();
    const bool have_app = t.have_binding && t.context != EGL_NO_CONTEXT;
    const EGLDisplay dpy = have_app ? t.display : eglDisplay;
    const EGLContext share = have_app ? t.context : EGL_NO_CONTEXT;

    if (CreateThreadContext(dpy, share)) {
        LOG_W_FORCE("BindFallbackEGLContext: [%s] this thread had no current EGL context, so MobileGLES gave it "
                    "one of its own%s, bound with no surface. It cannot present — only the thread holding the "
                    "application's context can — but objects it creates are visible to the game.",
                    CurrentThreadLabel(), have_app ? " sharing with the application's context" : "");
        return true;
    }

    // Last resort: the startup context. It shares with nothing, so it is only
    // good for queries, and it is one object — a second thread will be refused.
    if (eglContext != EGL_NO_CONTEXT &&
        egl_eglMakeCurrent(eglDisplay, kNoSurface, kNoSurface, eglContext) == EGL_TRUE) {
        t_fb.ctx = eglContext;
        t_fb.share_with = EGL_NO_CONTEXT;
        LOG_W_FORCE("BindFallbackEGLContext: [%s] could not create a context for this thread; using the startup "
                    "context, which shares with nothing. Objects created here are invisible to the game, and "
                    "other threads will be refused it",
                    CurrentThreadLabel());
        return true;
    }

    LOG_W_FORCE("BindFallbackEGLContext: [%s] NO CONTEXT could be bound. Every host GL call from this thread will "
                "be discarded without error — a shader compile or a buffer map will fail with no explanation.",
                CurrentThreadLabel());
    return false;
}

// Pairs a successful BindFallbackEGLContextIfNeeded().
//
// Deliberately does nothing: the context stays current for the life of the
// thread. Releasing it would reintroduce the per-call churn that corrupts
// driver state, and the window in which another thread could take it.
void UnbindFallbackEGLContext() {}
