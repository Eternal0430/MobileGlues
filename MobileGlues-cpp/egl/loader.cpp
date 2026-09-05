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
#include <chrono>
#include <thread>

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
    // True when `ctx` is the application's own context, bound to the surface it
    // presents from. Such a thread can draw and present; it must never be
    // swapped for a substitute.
    bool using_app_context = false;
    // Which surface it is bound to. Compared against the presenting surface on
    // every generation change, because the first binding happens before the
    // presenting surface is known — the surface only becomes observable once
    // eglSwapBuffers runs — so the binding has to be corrected afterwards.
    EGLSurface bound_surface = EGL_NO_SURFACE;
    bool tried = false;
    // Generation at which the last attempt was made; see BindFallback.
    unsigned tried_generation = 0;
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

// Creates a context for this thread and binds it to `surf` (EGL_NO_SURFACE for
// surfaceless).
bool CreateThreadContext(EGLDisplay dpy, EGLContext share, EGLSurface surf) {
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
        if (egl_eglMakeCurrent(dpy, surf, surf, ctx) != EGL_TRUE) {
            egl_eglGetError();
            if (egl_eglDestroyContext) egl_eglDestroyContext(dpy, ctx);
            continue;
        }
        t_fb.ctx = ctx;
        t_fb.share_with = share;
        t_fb.bound_surface = surf;
        return true;
    }
    return false;
}

// One binding attempt, judged by what actually took effect.
//
// The return value of eglMakeCurrent is not trustworthy on this device. It was
// measured returning EGL_FALSE while eglGetError() reported 0x3000, which is
// EGL_SUCCESS — "failed, and nothing went wrong". A real log line:
//
//     app context + recorded window surface: FAILED (0x3000)
//     app context surfaceless: bound
//
// Taking that at face value is what kept the picture frozen: the ladder treated
// the first step as a failure, fell through, and bound the same context again
// with NO SURFACE. That second call succeeded and wiped out the binding that had
// a surface. Everything afterwards was drawn into nothing while eglSwapBuffers
// kept presenting the frame that never changed.
//
// So the result is decided by observed state — eglGetCurrentContext() and
// eglGetCurrentSurface() — not by the return value. Both are still logged,
// because the disagreement between them is itself worth seeing.
bool TryBind(EGLDisplay dpy, EGLContext ctx, EGLSurface surf, const char* what) {
    LOAD_EGL(eglMakeCurrent);
    LOAD_EGL(eglGetCurrentContext);
    LOAD_EGL(eglGetCurrentSurface);
    LOAD_EGL(eglGetError);

    const EGLBoolean ret = egl_eglMakeCurrent(dpy, surf, surf, ctx);
    const EGLint err = egl_eglGetError();
    const EGLContext now_ctx = egl_eglGetCurrentContext();
    const EGLSurface now_surf = egl_eglGetCurrentSurface(EGL_DRAW);

    // A surfaceless bind is satisfied by the context alone. A bind that asked
    // for a surface must actually be on that surface — landing on EGL_NO_SURFACE
    // instead means the surface was refused and drawing will be discarded.
    const bool took_effect = (now_ctx == ctx) && (surf == kNoSurface || now_surf == surf);

    if (took_effect) {
        LOG_W_FORCE("BindFallbackEGLContext: [%s] %s: bound (ret=%d err=0x%x surf=%p)", CurrentThreadLabel(), what,
                    (int)ret, err, now_surf);
        return true;
    }
    LOG_W_FORCE("BindFallbackEGLContext: [%s] %s: NOT BOUND (ret=%d err=0x%x) — wanted ctx=%p surf=%p, got ctx=%p "
                "surf=%p",
                CurrentThreadLabel(), what, (int)ret, err, ctx, surf, now_ctx, now_surf);

    // 0x300D = EGL_BAD_SURFACE: the surface is gone. Forget it so the next
    // attempt does not repeat the failure and silently draw into nothing.
    if (surf != kNoSurface && err == 0x300D) mg_egl_forget_surface(surf);
    return false;
}

} // namespace

bool BindFallbackEGLContextIfNeeded() {
    if (!g_fallback_context_ready) return false;

    LOAD_EGL(eglGetCurrentContext);
    LOAD_EGL(eglGetCurrentSurface);
    LOAD_EGL(eglGetCurrentDisplay);
    LOAD_EGL(eglMakeCurrent);
    LOAD_EGL(eglDestroyContext);
    LOAD_EGL(eglGetError);

    // Hot path: one atomic load. The record itself is behind a lock, so it is
    // only read when the generation says something actually changed.
    const unsigned gen = mg_egl_app_target_generation();
    if (gen != t_seen_generation) {
        t_seen_generation = gen;
        const AppRenderTarget& t = mg_egl_app_target();

        // Follow the application's render target as it changes.
        //
        // The first binding is necessarily provisional: SDL binds a context and
        // a surface early, then destroys that surface and creates another, so
        // whatever was recorded at that moment is dead before the first frame.
        // Measured on real hardware — the bind that succeeded at startup later
        // failed with 0x300d (EGL_BAD_SURFACE) — and without this correction the
        // thread keeps drawing into a surface that no longer exists while
        // eglSwapBuffers presents another one. Frozen picture, working audio.
        //
        // Who may move where:
        //   - A thread holding the application's own context follows the
        //     application's surface. Only one thread can hold that context, so
        //     there is no contention.
        //   - A thread with a context of its own may only take the presenting
        //     surface if it is the thread that calls eglSwapBuffers; otherwise a
        //     worker would block the very thread the picture depends on.
        const bool is_presenting_thread = t.have_presenting && t.presenting_thread != 0 &&
                                          pthread_equal(t.presenting_thread, pthread_self()) != 0;

        EGLSurface target = EGL_NO_SURFACE;
        bool should_move = false;
        if (t_fb.ctx != EGL_NO_CONTEXT) {
            if (t_fb.using_app_context) {
                if (t.have_surface && t.draw_surface != EGL_NO_SURFACE) {
                    target = t.draw_surface;
                    should_move = true;
                }
            } else if (is_presenting_thread && t.have_presenting && t.presenting_surface != EGL_NO_SURFACE) {
                target = t.presenting_surface;
                should_move = true;
            }
        }

        if (should_move && t_fb.bound_surface != target) {
            if (TryBind(t.display, t_fb.ctx, target, "follow the application's render target")) {
                LOG_W_FORCE("BindFallbackEGLContext: [%s] moved onto surface %p (was %p), so drawing here reaches "
                            "the screen",
                            CurrentThreadLabel(), target, t_fb.bound_surface);
                t_fb.bound_surface = target;
            } else if (t.have_binding && t.context != EGL_NO_CONTEXT) {
                // Refused — the application's context is probably current on
                // another thread. The presenting thread is the one thread that
                // must not be left unable to draw, so give it a context of its
                // own on that surface, sharing with the application's so the
                // game's objects stay visible.
                if (is_presenting_thread) {
                    const EGLContext old = t_fb.ctx;
                    t_fb.ctx = EGL_NO_CONTEXT;
                    if (CreateThreadContext(t.display, t.context, target)) {
                        if (egl_eglDestroyContext && old != eglContext) egl_eglDestroyContext(t.display, old);
                        LOG_W_FORCE("BindFallbackEGLContext: [%s] could not rebind the application's context, so "
                                    "this thread was given a new one on surface %p — drawing here reaches the "
                                    "screen",
                                    CurrentThreadLabel(), target);
                    } else {
                        t_fb.ctx = old;  // losing the context entirely is worse
                        egl_eglMakeCurrent(t.display, t_fb.bound_surface, t_fb.bound_surface, old);
                    }
                }
            }
        }

        if (!t_fb.using_app_context && t.have_binding && t.context != EGL_NO_CONTEXT &&
            t_fb.ctx != EGL_NO_CONTEXT && t_fb.ctx != eglContext && t_fb.share_with != t.context) {
            const EGLContext old = t_fb.ctx;
            const EGLSurface old_surf = t_fb.bound_surface;
            t_fb.ctx = EGL_NO_CONTEXT;
            if (CreateThreadContext(t.display, t.context, old_surf)) {
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

    // `tried` used to be latched for the life of the thread, so a thread whose
    // first attempt failed — because no context was available yet, which is the
    // normal state early in startup — was never given another chance. Every GL
    // call it made afterwards was silently discarded, with no log, because the
    // ladder was never entered again.
    //
    // Retry whenever the application's render target has changed, since that is
    // exactly when a previously impossible binding may have become possible.
    if (t_fb.tried && gen == t_fb.tried_generation) return false;
    t_fb.tried = true;
    t_fb.tried_generation = gen;

    const AppRenderTarget& t = mg_egl_app_target();
    const bool have_app = t.have_binding && t.context != EGL_NO_CONTEXT;
    const bool have_presenting = t.have_presenting && t.presenting_surface != EGL_NO_SURFACE;
    const bool have_recorded = t.have_surface && t.draw_surface != EGL_NO_SURFACE;
    const EGLDisplay dpy = have_app ? t.display : eglDisplay;

    // Everything needed to explain a refusal, in one line. Five designs were
    // built on guesses about these values; this prints them.
    LOG_W_FORCE("BindFallbackEGLContext: [%s] no current context. this thread: ctx=%p draw=%p dpy=%p | recorded "
                "app: dpy=%p ctx=%p draw=%p presenting=%p | own dpy=%p",
                CurrentThreadLabel(), egl_eglGetCurrentContext(), egl_eglGetCurrentSurface(EGL_DRAW),
                egl_eglGetCurrentDisplay(), t.display, t.context, t.draw_surface, t.presenting_surface, eglDisplay);
    if (have_app && t.display != eglDisplay) {
        LOG_W_FORCE("BindFallbackEGLContext: [%s] the application is on a DIFFERENT EGLDisplay (%p) than the one "
                    "MobileGLES initialized (%p). A context may only be bound on the display it was created for, "
                    "so bindings here are likely to be refused",
                    CurrentThreadLabel(), t.display, eglDisplay);
    }

    const bool is_presenting_thread = have_presenting && t.presenting_thread != 0 &&
                                      pthread_equal(t.presenting_thread, pthread_self()) != 0;
    const bool can_use_presenting = have_presenting && is_presenting_thread;

    // Before the first eglSwapBuffers the presenting thread is unknown, so
    // refusing the window surface here would leave the render thread with
    // nothing drawable on the very first frames. Once presenting is known, only
    // the presenting thread may take it — otherwise a worker could block the
    // thread the picture depends on.
    const bool may_use_window_surface = !have_presenting || is_presenting_thread;

    // 1. The application's own context on the presenting surface. This is what a
    //    render thread needs: it is the context the game's objects live in and
    //    the only one that can present.
    if (have_app && can_use_presenting &&
        TryBind(dpy, t.context, t.presenting_surface, "app context + presenting surface")) {
        t_fb.ctx = t.context;
        t_fb.share_with = t.context;
        t_fb.using_app_context = true;
        t_fb.bound_surface = t.presenting_surface;
        return true;
    }

    // 2. The application's context on the window surface recorded at creation.
    if (have_app && have_recorded && may_use_window_surface &&
        (!have_presenting || t.draw_surface != t.presenting_surface) &&
        TryBind(dpy, t.context, t.draw_surface, "app context + recorded window surface")) {
        t_fb.ctx = t.context;
        t_fb.share_with = t.context;
        t_fb.using_app_context = true;
        t_fb.bound_surface = t.draw_surface;
        return true;
    }

    // 3. The application's context, surfaceless. Queries and compiles work;
    //    drawing does not, but it is better than nothing and it is corrected
    //    later by the presenting-surface logic above.
    if (have_app && TryBind(dpy, t.context, kNoSurface, "app context surfaceless")) {
        t_fb.ctx = t.context;
        t_fb.share_with = t.context;
        t_fb.using_app_context = true;
        t_fb.bound_surface = kNoSurface;
        return true;
    }

    // 4/5. A context of this thread's own, sharing with the application's so
    //      that objects it creates are visible to the game. Only the presenting
    //      thread binds it to the presenting surface.
    const EGLContext share = have_app ? t.context : EGL_NO_CONTEXT;
    if (can_use_presenting && CreateThreadContext(dpy, share, t.presenting_surface)) {
        LOG_W_FORCE("BindFallbackEGLContext: [%s] gave this thread a context of its own on the presenting surface "
                    "%p, so drawing here reaches the screen",
                    CurrentThreadLabel(), t.presenting_surface);
        return true;
    }
    if (CreateThreadContext(dpy, share, kNoSurface)) {
        LOG_W_FORCE("BindFallbackEGLContext: [%s] gave this thread a context of its own%s, surfaceless. It cannot "
                    "present, but objects it creates are visible to the game",
                    CurrentThreadLabel(), have_app ? " sharing with the application's context" : "");
        return true;
    }

    // Last resort: the startup context. It shares with nothing, so it is only
    // good for queries, and it is one object — a second thread will be refused.
    if (eglContext != EGL_NO_CONTEXT && TryBind(eglDisplay, eglContext, kNoSurface, "startup context surfaceless")) {
        t_fb.ctx = eglContext;
        t_fb.share_with = EGL_NO_CONTEXT;
        t_fb.using_app_context = false;
        t_fb.bound_surface = kNoSurface;
        return true;
    }

    LOG_W_FORCE("BindFallbackEGLContext: [%s] NO CONTEXT could be bound. Every host GL call from this thread will "
                "be discarded without error — a shader compile or a buffer map will fail with no explanation.",
                CurrentThreadLabel());
    return false;
}

namespace {

// ---------------------------------------------------------------------------
// Watchdog
//
// The context layer is now observably correct: the log shows the render thread
// bound to the surface the application created, with the application's own
// context. And yet the game stops before a single eglSwapBuffers. Everything
// that follows is therefore outside what this library logs, and guessing at it
// has cost several rounds.
//
// This prints, every 20 seconds, whether GL work is still flowing and what the
// render target looks like. A log that stops growing with a correct binding
// means the stall is in game or driver code; one that keeps growing while
// nothing is presented means the stall is in the presentation path.
// ---------------------------------------------------------------------------
std::atomic<unsigned long> g_guarded_calls{0};
std::atomic<bool> g_watchdog_started{false};

void WatchdogLoop() {
    unsigned long last = 0;
    int tick = 0;
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(20));
        const unsigned long now = g_guarded_calls.load(std::memory_order_relaxed);
        const unsigned long delta = now - last;
        last = now;

        const AppRenderTarget& t = mg_egl_app_target();
        LOG_W_FORCE("watchdog #%d: %lu guarded GL calls this period (%lu total)%s | window surface=%p "
                    "presenting=%p presenting thread=%s",
                    ++tick, delta, now, delta == 0 ? " — NO GL CALLS, the game is not drawing" : "",
                    t.draw_surface, t.presenting_surface,
                    t.have_presenting ? "known" : "unknown (eglSwapBuffers has never run)");
    }
}

}  // namespace

void mg_egl_note_guarded_call() {
    g_guarded_calls.fetch_add(1, std::memory_order_relaxed);
    if (!g_watchdog_started.exchange(true)) {
        // Detached and deliberately never joined: it outlives the GL session and
        // costs one wake-up every 20 seconds.
        std::thread(WatchdogLoop).detach();
    }
}

// Pairs a successful BindFallbackEGLContextIfNeeded().
//
// Deliberately does nothing: the context stays current for the life of the
// thread. Releasing it would reintroduce the per-call churn that corrupts
// driver state, and the window in which another thread could take it.
void UnbindFallbackEGLContext() {}
