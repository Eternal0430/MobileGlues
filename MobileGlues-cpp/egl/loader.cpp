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
static EGLContext eglContext = EGL_NO_CONTEXT;
// Not used. Kept as a named constant at the binding sites instead, so that the
// absence of a surface is visible where it matters: this context is
// surfaceless, see init_target_egl().
static const EGLSurface kNoSurface = EGL_NO_SURFACE;

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
// Fallback contexts, without pbuffers
//
// Three designs were tried on this device and the shape of each failure is what
// motivates this one. All three created an off-screen pbuffer; none of them can
// work, and the reason is the same in every case.
//
//   1. One shared pbuffer context, bound and released around every call.
//      Two eglMakeCurrent round-trips per GL entry point. On Adreno/turnip that
//      is not merely slow, it corrupts state across calls: a session logged
//      15 glMapBufferRange failures on buffers that demonstrably had storage,
//      a glBufferStorage rejection, 14 texture uploads with nothing bound, and
//      a vertex shader that failed to compile with an empty info log — the
//      signature of a driver discarding the call outright.
//
//   2. One shared pbuffer context, bound and kept. State survives, but an
//      EGLContext may be current on only one thread. The render thread held it
//      across CompletableFuture.join() while the background executor was trying
//      to compile shaders, and neither could proceed. That is the hang.
//
//   3. One pbuffer context per thread, bound and kept. No contention and no
//      churn — and a permanently black screen, because a pbuffer is a valid
//      render target that is not connected to the display. The render thread
//      asks for a fallback before SDL has created the window, so it gets a
//      pbuffer, and nothing re-examined that decision afterwards. Every draw
//      call succeeded; every pixel went somewhere invisible; eglSwapBuffers
//      presented a window surface nothing had been drawn into.
//
// The common mistake is that all three invent a context. This one does not.
// It binds the context the application already made, to the surface the
// application already presents from. That is not a convenience — it is the only
// arrangement in which drawing from a fallback thread can be visible, and it
// also settles object sharing for free, because it is the same context: every
// texture, buffer, shader and program the application created is already there.
//
// The application's context exists before the window does (SDL's
// eglCreateContext is logged before the fallback is first needed), so the
// binding starts surfaceless — which EGL_KHR_surfaceless_context allows, and
// which this device supports — and moves onto the window surface as soon as one
// appears. Rebinding a context to a different surface is defined behaviour and
// keeps every object created so far.
//
// No pbuffer surface and no EGLContext is created here at all.
// ---------------------------------------------------------------------------
namespace {

struct ThreadFallback {
    EGLContext ctx = EGL_NO_CONTEXT;
    EGLDisplay dpy = EGL_NO_DISPLAY;
    // The context is bound to the application's window surface, so drawing is
    // visible. False means surfaceless: GL calls work, nothing can be seen.
    bool on_window = false;
    bool tried = false;
};

thread_local ThreadFallback t_fb;
// Last render-target generation this thread acted on; see
// BindFallbackEGLContextIfNeeded.
thread_local unsigned t_seen_generation = 0;

// Binds the context to the surface the application presents from, or
// surfaceless when no window surface exists yet.
//
// Returns false when the binding fails, which for a context means another
// thread holds it — that is worth reporting rather than papering over, because
// it means the application is rendering somewhere other than here.
bool BindAppContext(EGLDisplay dpy, EGLContext ctx, EGLSurface draw, EGLSurface read) {
    LOAD_EGL(eglMakeCurrent);
    LOAD_EGL(eglGetError);
    return egl_eglMakeCurrent(dpy, draw, read, ctx) == EGL_TRUE;
}

} // namespace

bool BindFallbackEGLContextIfNeeded() {
    LOAD_EGL(eglGetCurrentContext);
    LOAD_EGL(eglMakeCurrent);
    LOAD_EGL(eglGetError);

    const AppRenderTarget& t = mg_egl_app_target();

    // Re-examine a surfaceless binding whenever the application's render target
    // changes, so that a thread which arrived before the window existed is not
    // left drawing into nothing for the rest of the process. This is the entire
    // cost of the guard on the hot path: one atomic load, and a real bind only
    // on the frames where the target actually changed.
    const unsigned gen = mg_egl_app_target_generation();
    if (gen != t_seen_generation) {
        t_seen_generation = gen;
        if (t_fb.ctx != EGL_NO_CONTEXT && !t_fb.on_window && t.have_surface) {
            if (BindAppContext(t.display, t_fb.ctx, t.draw_surface, t.read_surface)) {
                t_fb.on_window = true;
                LOG_W_FORCE("BindFallbackEGLContext: this thread was surfaceless and is now bound to the "
                            "application's window surface %p, so its drawing will be visible",
                            t.draw_surface);
            }
        }
    }

    // Already have a context: the path every correctly configured thread takes,
    // and the only path a thread holding a real context ever takes.
    if (egl_eglGetCurrentContext() != EGL_NO_CONTEXT) return false;

    if (t_fb.tried) return false;
    t_fb.tried = true;

    // The application's own context, onto the window surface if there is one.
    // Same context means the application's objects are already visible.
    if (t.have_binding && t.context != EGL_NO_CONTEXT) {
        const bool have_surface = t.have_surface && t.draw_surface != EGL_NO_SURFACE;
        if (BindAppContext(t.display, t.context, have_surface ? t.draw_surface : kNoSurface,
                           have_surface ? t.read_surface : kNoSurface)) {
            t_fb.ctx = t.context;
            t_fb.dpy = t.display;
            t_fb.on_window = have_surface;
            if (have_surface) {
                LOG_W_FORCE("BindFallbackEGLContext: bound the application's own context %p to the application's "
                            "window surface %p, so drawing from this thread is visible",
                            t.context, t.draw_surface);
            } else {
                LOG_W_FORCE("BindFallbackEGLContext: bound the application's own context %p with NO SURFACE — no "
                            "window surface exists yet. GL calls work but nothing can be presented; this thread "
                            "will move onto the window surface as soon as one is created",
                            t.context);
            }
            return true;
        }
        const EGLint err = egl_eglGetError();
        LOG_W_FORCE("BindFallbackEGLContext: the application's context %p is current on another thread (0x%x). "
                    "This thread is not the one the application is rendering on, and no substitute is being "
                    "created: a new context cannot be bound to a surface another context already holds, so any "
                    "drawing here would be invisible anyway",
                    t.context, err);
    }

    // Startup context, as a last resort: it predates the application's window
    // and exists only so that queries issued before any context is available
    // have somewhere to run.
    if (eglContext != EGL_NO_CONTEXT && BindAppContext(eglDisplay, eglContext, kNoSurface, kNoSurface)) {
        t_fb.ctx = eglContext;
        t_fb.dpy = eglDisplay;
        t_fb.on_window = false;
        LOG_W_FORCE("BindFallbackEGLContext: no application context was available, so this thread is bound to the "
                    "startup context surfacelessly. Queries and shader compiles work; drawing does not. This "
                    "runs before the application has created a context and should not persist once it has.");
        return true;
    }

    LOG_W_FORCE("BindFallbackEGLContext: no context could be bound at all — host GL calls from this thread will be "
                "silently discarded");
    return false;
}

// Pairs a successful BindFallbackEGLContextIfNeeded().
//
// Deliberately does nothing: the context stays current for the life of the
// thread. Releasing it would reintroduce the per-call churn that corrupts
// driver state, and the window in which another thread could take it.
void UnbindFallbackEGLContext() {}
