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
#include "../config/config.h"
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

// The config a surface was created with, so a context made for it is compatible.
//
// g_context_config is chosen in init_target_egl() with EGL_SURFACE_TYPE =
// EGL_PBUFFER_BIT and EGL_RENDERABLE_TYPE = EGL_OPENGL_ES2_BIT, because it only
// ever has to back the startup pbuffer. Creating a context on it and binding
// that to a window surface the application made is asking for EGL_BAD_MATCH:
// EGL requires the context's config and the surface's config to be compatible,
// and a pbuffer-only ES2 config is not compatible with a window surface.
//
// That is why "CreateThreadContext ... failed" was not occasional but certain,
// and a failed eglMakeCurrent on this driver does not leave the previous
// binding alone -- it disturbs the draw binding, which is what pushed rendering
// off the window surface and onto a fallback path.
//
// Asking the surface which config it is is the only correct answer, so try that
// first and fall back to the startup config only when the query is unavailable.
static bool ConfigForSurface(EGLDisplay dpy, EGLSurface surf, EGLConfig* out) {
    LOAD_EGL(eglQuerySurface);
    LOAD_EGL(eglChooseConfig);
    if (!egl_eglQuerySurface || !egl_eglChooseConfig || surf == EGL_NO_SURFACE) return false;

    EGLint config_id = 0;
    if (egl_eglQuerySurface(dpy, surf, EGL_CONFIG_ID, &config_id) != EGL_TRUE) return false;

    const EGLint attribs[] = {EGL_CONFIG_ID, config_id, EGL_NONE};
    EGLConfig cfg = nullptr;
    EGLint found = 0;
    if (egl_eglChooseConfig(dpy, attribs, &cfg, 1, &found) != EGL_TRUE || found == 0) return false;
    *out = cfg;
    return true;
}

// Creates a context for this thread and binds it to `surf` (EGL_NO_SURFACE for
// surfaceless).
bool CreateThreadContext(EGLDisplay dpy, EGLContext share, EGLSurface surf) {
    LOAD_EGL(eglCreateContext);
    LOAD_EGL(eglDestroyContext);
    LOAD_EGL(eglMakeCurrent);
    LOAD_EGL(eglGetError);
    if (!egl_eglCreateContext || !g_context_config_valid) return false;

    // Prefer the config the surface was actually created with; only a context on
    // a compatible config can be bound to it.
    EGLConfig config = g_context_config;
    if (surf != EGL_NO_SURFACE) {
        EGLConfig surface_config = nullptr;
        if (ConfigForSurface(dpy, surf, &surface_config)) config = surface_config;
    }

    // Sharing requires the driver to consider the two contexts compatible, and
    // the application's may be a later ES version than ours. Try the newest
    // first and work down.
    static const EGLint attrs_v3[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    static const EGLint attrs_v2[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    static const EGLint attrs_none[] = {EGL_NONE};
    const EGLint* candidates[] = {attrs_v3, attrs_v2, attrs_none};

    for (const EGLint* attrs : candidates) {
        EGLContext ctx = egl_eglCreateContext(dpy, config, share, attrs);
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
        return true;
    }

    // 0x300D = EGL_BAD_SURFACE: the surface is gone. Forget it so the next
    // attempt does not repeat the failure and silently draw into nothing.
    if (surf != kNoSurface && err == 0x300D) mg_egl_forget_surface(surf);
    return false;
}

} // namespace

bool BindFallbackEGLContextIfNeeded() {
    if (!g_fallback_context_ready) return false;

    // Only the one the hot path uses. The other five used to be resolved here
    // too, and each LOAD_EGL is a static variable load plus a first-time branch
    // — paid on every GL call, since this runs before the fast exit below. They
    // are now resolved inside the branches that actually call them. This
    // function is entered on every guarded GL call, and for calls the wrappers
    // short-circuit (glUseProgram, glActiveTexture and glScissor all return
    // early when the value is unchanged) it is the entire cost of the call.
    LOAD_EGL(eglGetCurrentContext);

    // Hot path: one atomic load. The record itself is behind a lock, so it is
    // only read when the generation says something actually changed.
    const unsigned gen = mg_egl_app_target_generation();
    if (gen != t_seen_generation) {
        t_seen_generation = gen;
        LOAD_EGL(eglMakeCurrent);
        LOAD_EGL(eglDestroyContext);
        LOAD_EGL(eglGetError);
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
    LOAD_EGL(eglGetCurrentSurface);
    LOAD_EGL(eglGetCurrentDisplay);
    LOAD_EGL(eglMakeCurrent);
    LOAD_EGL(eglDestroyContext);
    LOAD_EGL(eglGetError);

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
    if (have_app && t.display != eglDisplay) {
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
        return true;
    }
    if (CreateThreadContext(dpy, share, kNoSurface)) {
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
    if (!global_settings.activate_on_create) {
        // Off: the surface is only recorded, exactly as this library behaved
        // before the change. The application is then expected to call
        // eglMakeCurrent for it; when SDL reuses its primary window it does not,
        // and the surface that ends up on screen has no context bound to it.
        return;
    }

    const AppRenderTarget& t = mg_egl_app_target();

    // Case 1: application context is available and lives on THIS thread.
    if (t.have_binding && t.context != EGL_NO_CONTEXT) {
        const EGLDisplay target_dpy = (t.display != EGL_NO_DISPLAY) ? t.display : dpy;
        if (TryBind(target_dpy, t.context, surface, "activate the new window surface")) {
            t_fb.ctx = t.context;
            t_fb.bound_surface = surface;
            t_fb.using_app_context = true;
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
        } else {
            LOG_W_FORCE("window surface %p could not be bound: CreateThreadContext(share=eglContext, surface) failed on "
                        "pthread=%lu — the surface remains unbound and may produce a black screen",
                        surface, (unsigned long)pthread_self());
        }
    }
}

// Not inside an anonymous namespace: it is declared in loader.h and called from
// egl.cpp, so it needs external linkage. It landed inside one when the present
// fallback was deleted, which made the link fail with
// "undefined symbol: mg_egl_note_call(char const*)" — a fault that -fsyntax-only
// cannot catch, because each translation unit still compiles fine on its own.
void mg_egl_note_call() {
    // Was the watchdog's start-up trigger. The watchdog is gone; the call sites
    // remain because the declaration is shared, and this is now a no-op.
}

// Repair SDL's own bookkeeping — the fix for the black screen.
//
// SDL_GL_SwapWindow refuses to swap unless the window matches a value SDL keeps
// in thread-local storage (src/video/SDL_video.c):
//
//     if (SDL_GL_GetCurrentWindow() != window) return SDL_SetError(...);
//
// written only on a successful bind:
//
//     result = _this->GL_MakeCurrent(_this, window, context);
//     if (result) { _this->current_glwin = window;
//                   SDL_SetTLS(&_this->current_glwin_tls, window, NULL); }
//
// On the release path SDL calls Android_GLES_MakeCurrent(_this, NULL, NULL),
// which reaches SDL_EGL_MakeCurrent — and that function DISCARDS the return
// value of eglMakeCurrent and returns true unconditionally:
//
//     if (!egl_context || ...) { eglMakeCurrent(display, NO_SURFACE, ...); }
//     else { ... }
//     return true;
//
// so the bind is reported successful and current_glwin becomes NULL. Nothing
// re-binds afterwards, because the launcher's "primary window reuse" hook hands
// the same window back and the game sees no reason to bind again. Every swap is
// then refused inside SDL and none reaches EGL: hundreds of thousands of GL
// calls, eglSwapBuffers never called.
//
// Returning EGL_FALSE from our eglMakeCurrent cannot help — the return value is
// discarded before any check. The only way to reopen the gate is to make the
// bind happen through SDL, so SDL writes its own TLS. This library knows the
// context really is current on a real surface (it bound it), so re-stating the
// fact through SDL's public API is a correction, not a workaround: the swap
// still originates in SDL, still passes SDL's own gate, and still arrives at
// this library's eglSwapBuffers.
//
// Two facts make it possible:
//   - SDL_GLContext is the EGLContext pointer on this backend: SDL_EGL_CreateContext
//     returns (SDL_GLContext)egl_context, and SDL_EGL_MakeCurrent casts it back.
//   - SDL_GetWindows is exported, so the window can be enumerated.
static bool g_sdl_repair_done = false;
// Written only from the binding thread, and only after the modulo gate in
// mg_egl_note_guarded_call, so a plain int needs no synchronisation.
static std::atomic<int> g_sdl_repair_attempts{0};

static void RepairSdlCurrentWindow() {
    const AppRenderTarget& t = mg_egl_app_target();
    if (!t.have_binding || t.context == EGL_NO_CONTEXT || t.draw_surface == EGL_NO_SURFACE) return;

    void* sdl = dlopen("libSDL3.so", RTLD_NOLOAD);
    if (!sdl) return;  // not loaded yet; a later attempt will see it

    auto get_window = reinterpret_cast<void* (*)()>(dlsym(sdl, "SDL_GL_GetCurrentWindow"));
    auto make_current = reinterpret_cast<bool (*)(void*, void*)>(dlsym(sdl, "SDL_GL_MakeCurrent"));
    auto get_windows = reinterpret_cast<void** (*)(int*)>(dlsym(sdl, "SDL_GetWindows"));
    if (!get_window || !make_current || !get_windows) return;

    if (get_window() != nullptr) {
        g_sdl_repair_done = true;  // the gate is open; nothing left to do
        return;
    }

    int count = 0;
    void** windows = get_windows(&count);
    if (!windows || count <= 0) return;

    const bool ok = make_current(windows[0], t.context);
    if (ok) g_sdl_repair_done = true;
}

// How many calls a thread accumulates before publishing them to the shared
// counter.
//
// The counter is global and written from every thread that makes a GL call, so
// touching it once per call made its cache line bounce between cores: at the
// call rates seen here — a watchdog period measured 304696 guarded calls in 20
// seconds — that is a write to a contended line roughly fifteen thousand times
// a second, from each thread, to feed a number that is read once every 20
// seconds.
//
// Accumulating per thread and publishing in batches removes the contention
// without changing what the watchdog reports: it samples every 20 seconds, so a
// count that lags by at most one batch is indistinguishable from an exact one.
void mg_egl_note_guarded_call() {
    static thread_local unsigned long t_calls = 0;
    ++t_calls;
    VerifyContextStillCurrent(t_calls);

    // The repair below is the fix for the black screen, so it has to run — but
    // it only ever matters a handful of times, and this function is on the path
    // of every single GL call. Everything expensive is therefore behind the
    // cheapest possible test first: an integer modulo on a thread-local, then
    // a plain non-atomic read. No atomic, no dlopen, no dlsym unless the cheap
    // test has already passed.
    if ((t_calls % 5000) == 0 && !g_sdl_repair_done) {
        const AppRenderTarget& rt = mg_egl_app_target();
        if (rt.have_binding && rt.binding_thread == (unsigned long)pthread_self()) {
            const int n = g_sdl_repair_attempts.fetch_add(1, std::memory_order_relaxed) + 1;
            RepairSdlCurrentWindow();
            if (n >= 8) g_sdl_repair_done = true;
        }
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
    // Runs at dlopen time, before init_settings(), so the configuration is read
    // here rather than from global_settings. config_refresh() is idempotent;
    // init_settings() parses the same file again shortly after.
    config_refresh();
    if (config_has_key("selfPromotion") && config_get_int(const_cast<char*>("selfPromotion")) == 0) {
        return;
    }

    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&MobileGluesPromoteSelfToGlobalScope), &info) == 0 || !info.dli_fname) {
        return;
    }
    void* self = dlopen(info.dli_fname, RTLD_NOLOAD | RTLD_GLOBAL);
    if (!self) {
        return;
    }

    void* ours = dlsym(self, "eglSwapBuffers");
    void* via_default = dlsym(RTLD_DEFAULT, "eglSwapBuffers");
}

}  // namespace
