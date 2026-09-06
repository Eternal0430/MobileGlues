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
#include "../config/settings.h"
#include "../gles/loader.h"
#include "../includes.h"
#include <EGL/egl.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <thread>
#include <utility>
#include <vector>

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
// Call histogram
//
// Lock-free open addressing keyed on (calling thread, entry point name). The
// name is a __func__ string literal, so its address is stable for the life of
// the process and can be compared directly — no hashing of contents, no
// allocation, no lock. Guarded calls run at several thousand per second, so
// this has to be cheaper than what it is measuring.
//
// The thread is part of the key because the aggregate version could not answer
// the question that mattered. A run measured 276k guarded calls in 20 seconds —
// glTexParameteri, glBindTexture, glBindVertexArray, glDrawArraysInstanced —
// a render loop at roughly 880 draws per second, while eglSwapBuffers was never
// called once and the game logged nothing further. Without the thread, there is
// no way to tell "the render thread is spinning without presenting" from "a
// worker is burning GL calls that go nowhere", and those have nothing in common
// except the symptom.
// ---------------------------------------------------------------------------
constexpr size_t kHistogramSlots = 512;

struct CallSlot {
    std::atomic<const char*> name{nullptr};
    std::atomic<pthread_t> tid{0};
    std::atomic<unsigned long> count{0};
};

CallSlot g_call_table[kHistogramSlots];

void histogram_add(const char* name, pthread_t tid) {
    const size_t mix = (reinterpret_cast<uintptr_t>(name) >> 4) ^ (static_cast<uintptr_t>(tid) * 0x9E3779B97F4A7C15ull);
    size_t i = mix & (kHistogramSlots - 1);
    for (size_t probe = 0; probe < kHistogramSlots; ++probe) {
        const char* cur = g_call_table[i].name.load(std::memory_order_acquire);
        if (cur == name && g_call_table[i].tid.load(std::memory_order_acquire) == tid) {
            g_call_table[i].count.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (cur == nullptr) {
            const char* expected = nullptr;
            if (g_call_table[i].name.compare_exchange_strong(expected, name, std::memory_order_acq_rel)) {
                g_call_table[i].tid.store(tid, std::memory_order_release);
                g_call_table[i].count.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            // Someone else claimed the slot. If it was for this same key, count
            // here; otherwise keep probing.
            if (g_call_table[i].name.load(std::memory_order_acquire) == name &&
                g_call_table[i].tid.load(std::memory_order_acquire) == tid) {
                g_call_table[i].count.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
        i = (i + 1) & (kHistogramSlots - 1);
    }
    // Table full. Losing a few counts is harmless; the point is the ranking.
}

struct CallSample {
    pthread_t tid;
    const char* name;
    unsigned long count;
};

std::vector<CallSample> histogram_snapshot() {
    std::vector<CallSample> out;
    out.reserve(64);
    for (size_t i = 0; i < kHistogramSlots; ++i) {
        const char* n = g_call_table[i].name.load(std::memory_order_acquire);
        if (!n) continue;
        out.push_back(CallSample{g_call_table[i].tid.load(std::memory_order_acquire), n,
                                 g_call_table[i].count.load(std::memory_order_relaxed)});
    }
    return out;
}

// ---------------------------------------------------------------------------
// Staleness check
//
// The fallback binds a context once per thread and then assumes, forever, that
// it is still current. Nothing re-checks. That assumption is the one MobileGL's
// DirectGLES backend refuses to make — see the comment on its
// g_backendContextOwnerThread: a stale "current" claim "makes buffer ops issue
// GL calls that silently no-op (no context is current on that thread) while
// still updating shadow bookkeeping, permanently desynchronizing backend buffer
// state."
//
// So re-verify periodically instead of trusting the first bind. Cheap: one
// call every kVerifyInterval guarded calls. A thread whose context vanished is
// re-bound, and the recovery is logged exactly once so it shows up next to the
// watchdog lines rather than being buried.
// ---------------------------------------------------------------------------
constexpr unsigned long kVerifyInterval = 4096;

void VerifyContextStillCurrent(unsigned long calls_on_this_thread) {
    if ((calls_on_this_thread & (kVerifyInterval - 1)) != 0) return;

    LOAD_EGL(eglGetCurrentContext);
    if (!egl_eglGetCurrentContext) return;
    if (egl_eglGetCurrentContext() != EGL_NO_CONTEXT) return;

    // The context this thread was given is gone. Re-running the ladder is the
    // only correct response: every GL call from here on would otherwise be
    // discarded with no error, which is indistinguishable from success.
    static thread_local bool warned = false;
    if (!warned) {
        warned = true;
        LOG_W_FORCE("staleness check: [pthread=%lu] had a context on an earlier call but has NONE now — %lu calls "
                    "since the last check may have been discarded. Rebinding.",
                    (unsigned long)pthread_self(), kVerifyInterval);
    }
    t_fb.tried = false;
    BindFallbackEGLContextIfNeeded();
}

// ---------------------------------------------------------------------------
// Watchdog
//
// Added because the context layer became observably correct while the game
// still never reached a frame, and everything past that point was outside what
// this library logged. It answers, every 20 seconds, three questions that took
// several rounds to even ask properly:
//
//   1. Is GL work still flowing, or is the game stalled in its own code?
//   2. Has anything ever been presented?
//   3. Which entry points account for the traffic?
//
// The third is the one that matters most. A sustained few thousand calls per
// second with no eglSwapBuffers is either a render loop that cannot present, or
// a compile loop that never finishes; those look identical from the outside and
// are told apart instantly by which call dominates.
// ---------------------------------------------------------------------------
std::atomic<unsigned long> g_guarded_calls{0};
std::atomic<bool> g_watchdog_started{false};

void WatchdogLoop() {
    unsigned long last_total = 0;
    std::vector<CallSample> last_snapshot;
    int tick = 0;

    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(20));

        const unsigned long now = g_guarded_calls.load(std::memory_order_relaxed);
        const unsigned long delta = now - last_total;
        last_total = now;

        auto snapshot = histogram_snapshot();
        std::vector<CallSample> deltas;
        deltas.reserve(snapshot.size());
        for (const auto& kv : snapshot) {
            unsigned long before = 0;
            for (const auto& old : last_snapshot)
                if (old.tid == kv.tid && old.name == kv.name) { before = old.count; break; }
            if (kv.count > before) deltas.push_back(CallSample{kv.tid, kv.name, kv.count - before});
        }
        std::sort(deltas.begin(), deltas.end(),
                  [](const CallSample& a, const CallSample& b) { return a.count > b.count; });
        last_snapshot = std::move(snapshot);

        const AppRenderTarget& t = mg_egl_app_target();
        LOG_W_FORCE("watchdog #%d: %lu guarded GL calls this period (%lu total)%s | window surface=%p "
                    "presenting=%p presenting thread=%s",
                    ++tick, delta, now, delta == 0 ? " — NO GL CALLS, the game is not drawing" : "",
                    t.draw_surface, t.presenting_surface,
                    t.have_presenting ? "known" : "unknown (eglSwapBuffers has never run)");
        LOG_W_FORCE("watchdog #%d: the application bound its context on pthread=%lu%s", tick,
                    (unsigned long)t.binding_thread,
                    t.presenting_thread ? "" : " (eglSwapBuffers has never run, so no presenting thread is known)");

        const size_t n = deltas.size() < 10 ? deltas.size() : 10;
        for (size_t i = 0; i < n; ++i) {
            LOG_W_FORCE("watchdog #%d:   pthread=%lu  %lu  %s", tick, (unsigned long)deltas[i].tid,
                        deltas[i].count, deltas[i].name);
        }
    }
}

}  // namespace

// Bind a context to a freshly created window surface, right away, on whichever
// thread created it.
//
// This is the MobileGL/DirectGLES model: DirectGLES::InitWindowSurface()
// (MG_Backend/DirectGLES/DirectGLES.cpp) creates a window surface and then
// immediately calls its own MakeCurrent() with its OWN global g_Context —
// g_Surface = eglCreateWindowSurface(...); if (!MakeCurrent()) return false;
// It does NOT wait for the application to make a context current first.
// Therefore SDL reusing the primary window (creating a second surface before
// the application has bound its context to this thread) cannot leave the
// surface unbound. Measured on real hardware: without this, the second
// surface was created but never bound, and the screen stayed black while
// audio and buttons worked normally.
//
// There are two cases, and they are MUTUALLY EXCLUSIVE — case 1 returns on
// success and must never be followed by case 2 (see the note at the return).
//
// 1. The application already has a context bound somewhere (t.have_binding).
//    Use it — but only if it is not current on another thread. A context
//    current on another thread is left alone: EGL allows a context to be
//    current on one thread at a time, and stealing it would break the thread
//    that already has it.
//
// 2. No application context is bound yet (the SDL-reuse-window case), or the
//    application's context refused the surface. Fall back to this library's
//    own startup context (eglContext, created in init_target_egl()). This is
//    the key difference from the original version, which simply returned here
//    and left the surface orphaned. Binding our own context makes the surface
//    a valid, current drawable so the game's GL calls have somewhere to land;
//    when SDL later binds the application's context it simply replaces it.
void mg_egl_activate_window_surface(EGLDisplay dpy, EGLSurface surface) {
    if (surface == EGL_NO_SURFACE) return;

    const AppRenderTarget& t = mg_egl_app_target();

    // Case 1: application context is available and lives on THIS thread.
    if (t.have_binding && t.context != EGL_NO_CONTEXT) {
        const EGLDisplay target_dpy = (t.display != EGL_NO_DISPLAY) ? t.display : dpy;
        if (TryBind(target_dpy, t.context, surface, "activate the new window surface")) {
            t_fb.ctx = t.context;
            t_fb.bound_surface = surface;
            t_fb.using_app_context = true;
            LOG_W_FORCE("window surface %p bound to the application's context on pthread=%lu at creation time, so the "
                        "swap chain is complete even if SDL never calls eglMakeCurrent for it",
                        surface, (unsigned long)pthread_self());
            // STOP HERE. Falling through to case 2 was a regression: it called
            // CreateThreadContext for the SAME surface already bound above, and
            // that issues another eglMakeCurrent for a surface the application's
            // context now holds. The call fails (EGL_BAD_MATCH / BAD_ACCESS), and
            // on this driver a failing MakeCurrent still disturbs the draw
            // binding — the session that had this fall-through logged a
            // successful activate followed by "CreateThreadContext failed", and
            // eglSwapBuffers then never ran once, black screen. The session
            // without the fall-through bound the same surface and the
            // application swapped normally.
            return;
        }
        LOG_W_FORCE("window surface %p could not be bound to the application's context on pthread=%lu — trying the "
                    "library's own context instead",
                    surface, (unsigned long)pthread_self());
    }

    // Case 2: bind this library's own context. MobileGL uses its single global
    // g_Context for this; we cannot reuse eglContext directly because it may
    // already be current on the init/startup thread, and MakeCurrent would then
    // return EGL_BAD_ACCESS on this thread. So create a fresh per-thread context
    // that shares with eglContext (resources created on either are visible to
    // both) and bind THAT to the new surface. This is exactly the same shape as
    // the thread contexts BindFallbackEGLContextIfNeeded() already creates, just
    // done eagerly at surface creation rather than lazily on first GL call.
    //
    // Sharing with eglContext (rather than EGL_NO_CONTEXT) matters: any GL
    // objects the library has already created on eglContext must remain visible
    // once the game's rendering moves onto this surface.
    if (eglContext != EGL_NO_CONTEXT) {
        const EGLDisplay target_dpy = (t.display != EGL_NO_DISPLAY) ? t.display : eglDisplay;
        if (CreateThreadContext(target_dpy, /*share=*/eglContext, surface)) {
            // CreateThreadContext sets t_fb.ctx / share_with / bound_surface.
            t_fb.using_app_context = false;  // not the app's context; may be migrated later
            LOG_W_FORCE("window surface %p bound with a library-owned context (sharing with the startup context) on "
                        "pthread=%lu — the surface is now a valid drawable even though the application has not bound "
                        "its context yet (SDL window-reuse case)",
                        surface, (unsigned long)pthread_self());
        } else {
            LOG_W_FORCE("window surface %p could not be bound: CreateThreadContext(share=eglContext, surface) failed on "
                        "pthread=%lu — the surface remains unbound and may produce a black screen",
                        surface, (unsigned long)pthread_self());
        }
    }
}

namespace {

// ---------------------------------------------------------------------------
// Present fallback
//
// This library's eglSwapBuffers was called zero times during a session that
// logged 1.4 million guarded GL calls — a render loop running flat out, drawing
// into a surface that is never shown. The game is alive (audio and input work)
// and the surface is correct: swapping it from here made the picture appear.
//
// SDL is the one that would normally present, and it does not. It resolves
// eglSwapBuffers with dlsym rather than eglGetProcAddress, so the pointer it
// calls is not this library's entry point and those swaps, if they happen at
// all, go straight to the host and leave no trace here.
//
// Two things make this different from the earlier attempt that was reverted:
//
//   1. eglWaitGL() before the swap. Without it the swap presents whatever the
//      GPU happens to have finished, which is a half-drawn frame — that was the
//      flicker and the "two surfaces overlapping" look.
//
//   2. A hard interval instead of guessing where a frame ends. The previous
//      version tried to detect frame boundaries by timing gaps and got it
//      badly wrong: it logged "presented after 1 GL calls in that frame",
//      because a 1.2 ms gap occurs *inside* a frame during texture uploads.
//      A fixed interval plus the wait above presents whole frames.
//
// It stops permanently the moment the application presents on its own.
// ---------------------------------------------------------------------------
std::atomic<bool> g_present_fallback_active{true};

// Counted so a log can prove whether two swap chains ever ran at once. That is
// the question behind the "picture rolls back" symptom, and it cannot be
// answered from the picture itself.
std::atomic<unsigned long> g_present_fallback_swaps{0};

// Only on the drawing thread, where the context and surface are current.
thread_local std::chrono::steady_clock::time_point t_last_present{};

// The two most recent guarded-GL-call timestamps on this thread. Their
// difference is the gap *into* the current call, and that gap is the only signal
// available for where one frame ends and the next begins.
thread_local std::chrono::steady_clock::time_point t_prev_gl_call{};
thread_local std::chrono::steady_clock::time_point t_last_gl_call{};
thread_local bool t_gl_timestamps_primed = false;

// Draw calls issued since the last fallback presentation. A swap with none of
// them behind it presents a back buffer the application has not touched since
// the previous swap, and on a triple-buffered display that drags an older frame
// back to the front — the picture appears to roll back. Counting them is what
// keeps the fallback's rate tied to the application's actual frame rate instead
// of to the wall clock.
thread_local unsigned long t_draws_since_present = 0;

// Recent per-frame draw counts, used to recognise how big a frame is on this
// machine. A handful of draws followed by a pause is NOT the end of a frame:
// the log showed presentations after 2 and 3 draw calls, which split a single
// frame into dozens of partial ones and produced the constant rolling back.
//
// Timing cannot tell the two apart here, because the application never swaps and
// therefore never waits for vsync — frames run back to back, and the only gaps
// are stalls inside a frame (texture and chunk uploads). So the count of draw
// calls is the signal, and how many draws make a frame is learned at runtime
// rather than guessed: it varies with view distance, dimension and scene.
constexpr unsigned kDrawHistorySize = 8;
thread_local unsigned long t_draw_history[kDrawHistorySize] = {0};
thread_local unsigned t_draw_history_next = 0;

// Recent gaps between consecutive GL calls, in microseconds. The largest of
// them approximates the frame boundary, and it ages out on its own as the ring
// fills — so one unusually long stall (a loading hitch) raises the bar only
// briefly instead of permanently.
constexpr unsigned kGapHistorySize = 32;
thread_local long long t_gap_history_us[kGapHistorySize] = {0};
thread_local unsigned t_gap_history_next = 0;

long long RecentMaxGapUs() {
    long long largest = 0;
    for (unsigned i = 0; i < kGapHistorySize; ++i) {
        if (t_gap_history_us[i] > largest) largest = t_gap_history_us[i];
    }
    return largest;
}

void RecordGapUs(long long gap_us) {
    t_gap_history_us[t_gap_history_next] = gap_us;
    t_gap_history_next = (t_gap_history_next + 1) % kGapHistorySize;
}

// Largest frame seen recently. Frames are consistent in size, so anything well
// below this is a partially drawn frame.
unsigned long RecentMaxDrawsPerFrame() {
    unsigned long largest = 0;
    for (unsigned i = 0; i < kDrawHistorySize; ++i) {
        if (t_draw_history[i] > largest) largest = t_draw_history[i];
    }
    return largest;
}

void RecordDrawsPerFrame(unsigned long draws) {
    t_draw_history[t_draw_history_next] = draws;
    t_draw_history_next = (t_draw_history_next + 1) % kDrawHistorySize;
}

// A gap this wide between two consecutive GL calls means the renderer stopped
// issuing work — the previous frame is complete and a new one has not started.
// Inside a frame the calls are microseconds apart; between frames the thread
// waits on vsync, runs game logic and polls events.
//
// The earlier attempt used 1.2 ms and was wrong: a texture upload leaves a gap
// of that size *inside* a frame, so it fired dozens of times per frame and
// presented half-finished frames. 4 ms is above any intra-frame stall and below
// the idle time of a frame boundary at any playable frame rate.
constexpr auto kQuietPeriod = std::chrono::microseconds(4000);

// Floor on the presentation rate, so a quiet period cannot be followed by a
// burst of swaps.
constexpr auto kMinPresentInterval = std::chrono::microseconds(6000);

// Ceiling on how long to go without presenting. If the renderer never goes
// quiet — a continuous loop with no idle — presenting late is still better than
// never presenting at all, which is what a black screen is.
constexpr auto kMaxPresentLatency = std::chrono::milliseconds(100);

void MaybePresentFrame() {
    if (!g_present_fallback_active.load(std::memory_order_acquire)) return;
    if (!global_settings.present_fallback) return;

    const AppRenderTarget& t = mg_egl_app_target();
    if (t.display == EGL_NO_DISPLAY) return;

    // Prefer the surface this thread is actually drawing into; fall back to the
    // most recent window surface the application created.
    EGLSurface target = (t_fb.bound_surface != EGL_NO_SURFACE) ? t_fb.bound_surface : t.draw_surface;
    if (target == EGL_NO_SURFACE) return;

    const auto now = std::chrono::steady_clock::now();

    if (t_last_present.time_since_epoch().count() != 0 && now - t_last_present < kMinPresentInterval) return;

    // Present only at a moment when the renderer has stopped issuing work.
    // The gap into the current call is measured between the two most recent
    // calls, not between now and the previous one: this function runs as part of
    // the current call, so "now" is that call and the gap would always be zero.
    const bool primed = t_gl_timestamps_primed;
    const bool quiet = primed && (t_last_gl_call - t_prev_gl_call) >= kQuietPeriod;
    const bool overdue =
        !primed || (t_last_present.time_since_epoch().count() != 0 && now - t_last_present >= kMaxPresentLatency);
    if (!quiet && !overdue) return;

    // Nothing has been drawn since the last presentation, so there is no new
    // frame to show. Swapping anyway is what made the picture roll back: the
    // fallback ran at a fixed ~60 Hz while the game, busy loading and uploading
    // terrain, produced frames more slowly, so some swaps published a back
    // buffer the game had not rewritten — an older frame.
    if (t_draws_since_present == 0) return;

    // A pause alone is not enough: it also has to be a *frame-sized* pause.
    // Intra-frame stalls (texture and chunk uploads) are a few milliseconds;
    // the gap at a frame boundary is the largest one around, because that is
    // where game logic and event polling happen. So the threshold is derived
    // from the largest gaps seen recently rather than being a fixed constant.
    //
    // The rate was already right — the log shows ~60 presents per second, about
    // one per frame — so the remaining fault is phase, not frequency: a swap
    // landing mid-frame splits that frame across two buffers, and the screen
    // alternates between the two halves. Aligning to the largest recent gap is
    // what fixes the phase.
    //
    // Gap, not draw count, is the signal here for a practical reason: it can be
    // measured continuously, so it needs no correct answer to get started.
    // A draw-count threshold can only be learned by presenting first, and it
    // locks onto whatever the first (small, loading-screen) frames happened to
    // be — which is how an earlier attempt at this ended up demanding more
    // draws than a real frame ever issues, and stalling to 10 fps.
    if (!overdue) {
        const long long largest = RecentMaxGapUs();
        if (largest > 0) {
            const long long required_us = (largest * 3) / 5;
            const auto gap_us = std::chrono::duration_cast<std::chrono::microseconds>(t_last_gl_call - t_prev_gl_call)
                                    .count();
            if (gap_us < required_us) return;
        }
    }

    LOAD_EGL(eglSwapBuffers);
    LOAD_EGL(eglWaitGL);
    LOAD_EGL(eglGetError);
    if (!egl_eglSwapBuffers) return;

    // Finish everything issued so far, or the swap shows a half-drawn frame.
    if (egl_eglWaitGL) egl_eglWaitGL();

    t_last_present = now;
    const unsigned long draws = t_draws_since_present;
    t_draws_since_present = 0;
    RecordDrawsPerFrame(draws);
    const EGLBoolean ok = egl_eglSwapBuffers(t.display, target);
    if (ok != EGL_TRUE) {
        LOG_W_FORCE("present fallback: eglSwapBuffers(%p) failed (0x%x) — the picture will not update", target,
                    egl_eglGetError ? egl_eglGetError() : 0);
        return;
    }

    const unsigned long n = g_present_fallback_swaps.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n == 1) {
        LOG_W_FORCE("present fallback: the application never presents, so this library is swapping surface %p itself. "
                    "This compensates for a missing swap, it does not replace a working one.",
                    target);
    }
    if (n == 1 || n == 60 || n == 600) {
        LOG_W_FORCE("present fallback: %lu swaps so far (this one on %s, %lu draw calls)%s", n,
                    quiet ? "a quiet period — the frame looked complete" : "the latency ceiling — frame may be partial",
                    draws,
                    g_present_fallback_active.load(std::memory_order_acquire)
                        ? " — still active, so the application is not presenting"
                        : " — already stopped, so the application took over presenting");
    }
    mg_egl_note_swap(t.display, target, ok, /*from_fallback=*/true);
}

}  // namespace

void mg_egl_note_app_swap() {
    if (g_present_fallback_active.exchange(false, std::memory_order_acq_rel)) {
        LOG_W_FORCE("present fallback: stopping after %lu swaps, because the application is presenting on its own now",
                    g_present_fallback_swaps.load(std::memory_order_relaxed));
    }
}

void mg_egl_note_call(const char* entry_point) {
    histogram_add(entry_point, pthread_self());
    if (!g_watchdog_started.exchange(true)) {
        std::thread(WatchdogLoop).detach();
    }
}

void mg_egl_note_guarded_call(const char* entry_point) {
    static thread_local unsigned long t_calls = 0;
    ++t_calls;

    g_guarded_calls.fetch_add(1, std::memory_order_relaxed);
    histogram_add(entry_point, pthread_self());
    VerifyContextStillCurrent(t_calls);

    // Draw calls are what makes a frame worth showing. Multidraw is included
    // because that is the path this game actually takes.
    if (entry_point) {
        if (strncmp(entry_point, "glDraw", 6) == 0 || strncmp(entry_point, "glMultiDraw", 11) == 0) {
            ++t_draws_since_present;
        }
    }

    // Slide the call timestamps before the fallback reads them: it runs inside
    // this call, so the gap it must judge is the one leading into it.
    {
        const auto now = std::chrono::steady_clock::now();
        if (!t_gl_timestamps_primed) {
            t_gl_timestamps_primed = true;
            t_prev_gl_call = now;
        } else {
            t_prev_gl_call = t_last_gl_call;
            RecordGapUs(std::chrono::duration_cast<std::chrono::microseconds>(now - t_prev_gl_call).count());
        }
        t_last_gl_call = now;
    }

    MaybePresentFrame();
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

// ---------------------------------------------------------------------------
// Self-promotion into the global symbol scope
//
// Root cause of every bypass seen so far. This library is built with
// -fvisibility=hidden (CMakeLists.txt:32) and loaded with dlopen(...,
// RTLD_LOCAL) — the default — so NONE of its symbols are in the global symbol
// scope. Anything that resolves with dlsym(RTLD_DEFAULT, name) therefore gets
// the host driver's function instead of ours and calls straight through to it,
// leaving no trace in this library:
//
//   glx/lookup.cpp:56   proc = dlsym(RTLD_DEFAULT, name)
//                       -> eglGetProcAddress(glGetError) returned HOST driver
//                       -> eglGetProcAddress(glGetString) returned HOST driver
//                       (measured, five separate entry points)
//
// That is also why SDL's swap never reaches this library even though every
// other EGL call does (eglGetDisplay / eglInitialize / eglChooseConfig /
// eglCreateContext / eglSwapInterval all ran here). Those arrive because SDL
// holds a handle to this library; the swap is resolved by name lookup, which
// lands on the host.
//
// Re-opening this library with RTLD_NOLOAD | RTLD_GLOBAL adds it to the global
// scope without loading it a second time, so subsequent RTLD_DEFAULT lookups
// can resolve to these entry points.
// ---------------------------------------------------------------------------
namespace {

__attribute__((constructor)) void MobileGluesPromoteSelfToGlobalScope() {
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&MobileGluesPromoteSelfToGlobalScope), &info) == 0 || !info.dli_fname) {
        LOG_W_FORCE("self-promotion: could not determine this library's own path");
        return;
    }
    void* self = dlopen(info.dli_fname, RTLD_NOLOAD | RTLD_GLOBAL);
    if (!self) {
        LOG_W_FORCE("self-promotion: dlopen(%s, RTLD_NOLOAD|RTLD_GLOBAL) failed", info.dli_fname);
        return;
    }

    void* ours = dlsym(self, "eglSwapBuffers");
    void* via_default = dlsym(RTLD_DEFAULT, "eglSwapBuffers");
    LOG_W_FORCE("self-promotion: this library is in the global symbol scope now. eglSwapBuffers: ours=%p "
                "dlsym(RTLD_DEFAULT)=%p -> %s",
                ours, via_default,
                (ours != nullptr && ours == via_default) ? "RESOLVES TO THIS LIBRARY" : "still resolves elsewhere");
}

}  // namespace
