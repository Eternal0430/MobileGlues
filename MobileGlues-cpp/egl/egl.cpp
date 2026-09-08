// MobileGlues - egl/egl.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

#include "egl.h"
#include "../config/settings.h"
#include "../gl/FSR1/FSR1.h"
#include "../gl/log.h"
#include "../gl/mg.h"
#include "../gles/loader.h"
#include "../glx/lookup.h"
#include "loader.h"
#include <algorithm>
#include <unordered_map>
#include <vector>

#include <atomic>
#include <mutex>

#define DEBUG 0

// ---------------------------------------------------------------------------
// Recording the application's render target
//
// Every EGL entry point here is a passthrough, so the only way to know which
// surface and context the application presents from is to watch the calls go
// by. The fallback context in loader.cpp needs exactly that: a context bound to
// a 32x32 pbuffer renders correctly and shows nothing.
//
// The writes happen during startup and the reads on whatever thread needs a
// fallback, so the whole record sits behind one mutex.
// ---------------------------------------------------------------------------
namespace {

std::mutex g_target_mutex;
AppRenderTarget g_target;
std::atomic<unsigned> g_target_generation{0};

// ---------------------------------------------------------------------------
// Window-surface redirection
//
// Why this exists, from the log:
//
//   306  eglCreateWindowSurface -> 0x77530f5980      (first window)
//   313  eglMakeCurrent         -> 0x77530f5980
//   324  SDL_Hook: reusing primary window 0x78638a0600, refs=2
//   326  eglMakeCurrent: released
//   327  eglDestroySurface      -> 0x77530f5980
//   338  eglCreateWindowSurface -> 0x77c7356e00      (second window)
//
// SDL caches the EGLSurface inside its own window (0x78638a0600, reused). Its
// cached copy is 0x77530f5980, which by line 338 is already destroyed. Anything
// SDL then does with that handle is an operation on a dead surface, so its swap
// cannot reach the screen — and indeed this library's eglSwapBuffers and both
// damage variants were called zero times in every session logged.
//
// The fix is to stop letting that handle die. Window surfaces are kept alive in
// a short history, and any operation that arrives on an older one is redirected
// to the newest. SDL keeps using its cached handle, we quietly serve the surface
// that is actually on screen, and the swap chain completes — after which the
// present fallback switches itself off and the rolling-back goes away, because
// the application is presenting at its own frame boundaries again.
//
// Only surfaces created by eglCreateWindowSurface / eglCreatePlatformWindowSurface
// are tracked. Pbuffers and pixmaps are destroyed immediately as before.
// ---------------------------------------------------------------------------
// ===========================================================================
// Window surface — MobileGL's single-slot model
//
// MobileGL (MG_Backend/DirectGLES/DirectGLES.cpp) keeps exactly one of each:
//
//     g_Display, g_Config, g_Context, g_Surface
//
// and every operation goes through them:
//
//     InitWindowSurface(w)  DestroyEGLContext();              // old one dies first
//                           g_Surface = eglCreateWindowSurface(g_Display, g_Config, w, nullptr)
//                           MakeCurrent()                     // bound immediately
//     MakeCurrent()         eglMakeCurrent(g_Display, g_Surface, g_Surface, g_Context)
//     Present()             eglSwapBuffers(g_Display, g_Surface)
//     DestroyEGLContext()   eglMakeCurrent(NO_SURFACE,...); eglDestroyContext; eglDestroySurface
//
// Two properties follow, and both matter here:
//
//   1. The old surface is destroyed BEFORE the new one is created. On Android a
//      native window accepts only one EGLSurface, so any other order makes the
//      next eglCreateWindowSurface fail outright.
//   2. Nothing downstream reads the surface the application passed in. A stale
//      handle still held by the caller cannot break the swap chain, because the
//      chain is driven by the slot, not by the argument.
//
// This library is a pass-through rather than a backend, so it cannot ignore the
// application's argument the way MobileGL does. It keeps the same single slot
// and TRANSLATES whatever handle the application passes onto that slot. That is
// what lets SDL keep using a handle it cached before the surface was destroyed.
//
// Only handles from eglCreateWindowSurface / eglCreatePlatformWindowSurface are
// tracked. Pbuffers and pixmaps pass through untouched and are destroyed at once.
// ===========================================================================
std::mutex g_window_surfaces_mutex;
// The one surface currently on screen. This is the slot.
EGLSurface g_live_window_surface = EGL_NO_SURFACE;
// Handle VALUES of destroyed window surfaces. Never dereferenced — there is no
// use-after-free. They exist only so an operation issued on a dead handle can be
// pointed at the live one.
std::vector<EGLSurface> g_stale_window_surfaces;
constexpr size_t kMaxStaleWindowSurfaces = 8;

static bool IsLiveOrStale(EGLSurface surface) {
    return surface == g_live_window_surface ||
           std::find(g_stale_window_surfaces.begin(), g_stale_window_surfaces.end(), surface) !=
               g_stale_window_surfaces.end();
}

// Called right after a window surface is created: it becomes the slot.
void RecordWindowSurface(EGLDisplay, EGLSurface surface) {
    if (surface == EGL_NO_SURFACE) return;
    std::lock_guard<std::mutex> lock(g_window_surfaces_mutex);
    // A driver may reuse an address it just freed. The stale entry has to go, or
    // a perfectly good new surface would be mistaken for a dead one.
    g_stale_window_surfaces.erase(
        std::remove(g_stale_window_surfaces.begin(), g_stale_window_surfaces.end(), surface),
        g_stale_window_surfaces.end());
    if (g_live_window_surface != surface) {
        LOG_W_FORCE("window surface: %p is now the surface this library presents from", surface);
    }
    g_live_window_surface = surface;
}

// Called when the application destroys a surface. Returns true for a window
// surface we track. The real destroy happens in eglDestroySurface either way:
// the native window has to be released before the next surface is created.
bool ForgetWindowSurface(EGLSurface surface) {
    std::lock_guard<std::mutex> lock(g_window_surfaces_mutex);
    if (!IsLiveOrStale(surface)) return false;
    if (surface == g_live_window_surface) g_live_window_surface = EGL_NO_SURFACE;
    g_stale_window_surfaces.erase(
        std::remove(g_stale_window_surfaces.begin(), g_stale_window_surfaces.end(), surface),
        g_stale_window_surfaces.end());
    g_stale_window_surfaces.push_back(surface);
    while (g_stale_window_surfaces.size() > kMaxStaleWindowSurfaces) {
        g_stale_window_surfaces.erase(g_stale_window_surfaces.begin());
    }
    return true;
}

// The application bound this surface. MobileGL does the same in MakeEGLCurrent:
// "if (draw != m_eglSurface) ActivateEGLSurface(draw)".
void RecordCurrentWindowSurface(EGLSurface surface) {
    if (surface == EGL_NO_SURFACE) return;
    std::lock_guard<std::mutex> lock(g_window_surfaces_mutex);
    if (!IsLiveOrStale(surface)) return;  // not ours; leave the slot alone
    if (surface == g_live_window_surface) return;
    g_live_window_surface = surface;
    LOG_W_FORCE("window surface: the application bound %p, so that is what this library presents from", surface);
}

// Translate a dead handle onto the live surface. Anything else is returned as-is.
EGLSurface ResolveWindowSurface(EGLDisplay, EGLSurface surface) {
    std::lock_guard<std::mutex> lock(g_window_surfaces_mutex);
    if (surface == EGL_NO_SURFACE || g_live_window_surface == EGL_NO_SURFACE) return surface;
    if (IsLiveOrStale(surface)) return g_live_window_surface;
    return surface;  // not a window surface we know about
}

// ---------------------------------------------------------------------------
// Present — the MobileGL DirectGLES::Present counterpart
//
//     void Present() { ...; g_EGLFuncs.eglSwapBuffers(g_Display, g_Surface); ... }
//
// Same shape, same arguments: the display and the surface come from the slot,
// never from the caller. MobileGL reaches this from its own eglSwapBuffers
// export, after validating that the draw surface is current. This library is a
// pass-through, so the chain is inverted: the application's present never
// arrives (every logged session shows eglSwapBuffers called zero times, while
// tens of thousands of GL calls run), so the slot drives the swap itself.
//
// Called only while no surface is current-lost and only for a window surface we
// actually created, so it can never swap something that is not on screen.
// ---------------------------------------------------------------------------

} // namespace

static EGLDisplay g_slot_display = EGL_NO_DISPLAY;
static std::atomic<unsigned long> g_present_count{0};

void mg_egl_record_display(EGLDisplay dpy) {
    if (dpy != EGL_NO_DISPLAY) g_slot_display = dpy;
}

// The ONE present. Mirrors DirectGLES::Present(), which is likewise the only
// place that swaps and likewise takes both arguments from the slot.
//
// Every entry point funnels here, exactly as MobileGL's do:
//     EGLImpl::SwapBuffers -> SwapEGLBuffers -> Present()
// It is not that MobileGL has several presents; it has one, with callers.
//
// There is no pacing here, and none is needed: an application-driven call
// arrives once per frame by construction. The library no longer presents on its
// own. Every attempt to do so had to guess where a frame ends — a fixed
// interval, then an idle-gap threshold — and each guess was wrong in exactly the
// way that produces rollback: the swap lands mid-frame and the next one brings
// the earlier contents back.
// Recovery for a creation that failed because the previous surface was still
// attached. Destroys the recorded window surface and tries once. Corresponds to
// the DestroyEGLContext() at the top of MobileGL's InitWindowSurface, applied
// lazily: only when the creation has already failed, so the normal path is
// untouched.
// ===========================================================================
// Surface handle ownership — MobileGL's model
//
//     MG_State/EGLState/Core.cpp:830
//         const auto surface = EncodeHandle<EGLSurfaceHandle>(m_nextSurfaceHandle++);
//         m_surfaces[surface] = SurfaceObject{ ... };
//
// MobileGL mints its own handles and keeps the host surface beside them. The
// application never holds a host pointer, so the host surface can be destroyed
// and re-created underneath without the handle the application holds ever
// changing. That is why SDL reusing a cached window is harmless there: the
// handle it kept is still one MobileGL knows about.
//
// This library used to hand out the host pointer directly. Once that surface was
// destroyed the application's copy became a dangling value, and every later
// eglMakeCurrent / eglSwapBuffers using it failed or was silently dropped —
// which is how the picture ended up missing.
//
// From here on every surface created through this library gets a handle from the
// counter below. The host surface lives alongside it and is re-created on
// demand.
// ===========================================================================
struct OwnedSurface {
    EGLDisplay dpy = EGL_NO_DISPLAY;
    EGLConfig config = nullptr;
    EGLNativeWindowType win = 0;  // window surfaces only
    bool is_window = false;
    EGLSurface host = EGL_NO_SURFACE;
    bool host_destroyed = false;
};

std::mutex g_owned_surfaces_mutex;
std::unordered_map<EGLSurface, OwnedSurface> g_owned_surfaces;
// Deliberately small integers: they cannot be mistaken for a host pointer, and
// every handle is looked up rather than dereferenced.
uintptr_t g_next_surface_handle = 0x1000;

EGLSurface MintSurfaceHandle(const OwnedSurface& rec) {
    std::lock_guard<std::mutex> lock(g_owned_surfaces_mutex);
    const auto handle = reinterpret_cast<EGLSurface>(g_next_surface_handle++);
    g_owned_surfaces[handle] = rec;
    return handle;
}

// The handle the application holds stays valid for as long as it is not
// explicitly forgotten, even across destruction of the host surface. When the
// application keeps using it, the host surface is created again from the native
// window recorded at creation time.
static EGLSurface EnsureHostSurface(EGLSurface handle) {
    std::lock_guard<std::mutex> lock(g_owned_surfaces_mutex);
    auto it = g_owned_surfaces.find(handle);
    if (it == g_owned_surfaces.end()) return handle;  // not minted here: pass through
    OwnedSurface& rec = it->second;
    if (rec.host != EGL_NO_SURFACE && !rec.host_destroyed) return rec.host;
    if (!rec.is_window) return EGL_NO_SURFACE;

    LOAD_EGL(eglCreateWindowSurface)
    if (!egl_eglCreateWindowSurface) return EGL_NO_SURFACE;
    rec.host = egl_eglCreateWindowSurface(rec.dpy, rec.config, rec.win, nullptr);
    rec.host_destroyed = (rec.host == EGL_NO_SURFACE);
    LOG_W_FORCE("surface handle %p: host surface was gone, re-created it as %p — the application still holds this "
                "handle, which is the point of minting handles here",
                handle, rec.host);
    if (rec.host != EGL_NO_SURFACE) RecordWindowSurface(rec.dpy, rec.host);
    return rec.host;
}

// What the application last bound, so eglGetCurrentSurface can answer with the
// handle the application knows rather than the host's.
static thread_local EGLSurface t_current_draw_handle = EGL_NO_SURFACE;
static thread_local EGLSurface t_current_read_handle = EGL_NO_SURFACE;

// Destroy the host surface of every owned window surface bound to this native
// window. One ANativeWindow accepts only one EGLSurface, so a creation that
// failed for that reason can only succeed once the previous one is gone.
static void ReleaseHostSurfacesForWindow(EGLNativeWindowType win) {
    std::lock_guard<std::mutex> lock(g_owned_surfaces_mutex);
    LOAD_EGL(eglDestroySurface)
    if (!egl_eglDestroySurface) return;
    for (auto& kv : g_owned_surfaces) {
        OwnedSurface& rec = kv.second;
        if (!rec.is_window || rec.win != win) continue;
        if (rec.host == EGL_NO_SURFACE || rec.host_destroyed) continue;
        LOG_W_FORCE("surface handle %p: releasing host surface %p so the native window is free again", kv.first,
                    rec.host);
        egl_eglDestroySurface(rec.dpy, rec.host);
        rec.host_destroyed = true;
    }
}

static EGLSurface RetryWindowSurfaceAfterReleasingOld(EGLDisplay dpy, EGLConfig config, EGLNativeWindowType win,
                                                      const EGLint* attrib_list) {
    EGLSurface old_surface;
    {
        std::lock_guard<std::mutex> lock(g_window_surfaces_mutex);
        old_surface = g_live_window_surface;
    }
    if (old_surface == EGL_NO_SURFACE) return EGL_NO_SURFACE;

    LOG_W_FORCE("eglCreateWindowSurface: failed, and window surface %p is still attached to the native window — "
                "releasing it (as MobileGL does before creating) and trying once",
                old_surface);

    LOAD_EGL(eglCreateWindowSurface)
    if (!egl_eglCreateWindowSurface) return EGL_NO_SURFACE;
    // Now that handles are minted here, the old surface may be one of ours even
    // when the application's handle differs, so release by native window.
    ReleaseHostSurfacesForWindow(win);
    LOAD_EGL(eglDestroySurface)
    if (egl_eglDestroySurface) egl_eglDestroySurface(dpy, old_surface);
    ForgetWindowSurface(old_surface);

    EGLSurface retry = egl_eglCreateWindowSurface(dpy, config, win, attrib_list);
    LOG_W_FORCE("eglCreateWindowSurface: retry after releasing the old surface -> %p", retry);
    return retry;
}

// ValidateSurfaceOnDisplay — the MobileGL EGLImpl::SwapBuffers gate.
//
//     if (!state->ValidateSurfaceOnDisplay(dpy, draw)) {
//         state->SetError(EGL_BAD_SURFACE);
//         return EGL_FALSE;
//     }
//
// MobileGL refuses to swap a surface that does not belong to the display, and
// BackendObject::SwapEGLBuffers then refuses again unless the draw surface is
// the one it recorded. Both checks run BEFORE anything reaches eglSwapBuffers.
// This library never had them: it forwarded whatever handle it was given, so a
// surface from another display — or none at all — was passed straight to the
// host, which silently ignored it.
static bool ValidateSurfaceOnDisplay(EGLDisplay dpy, EGLSurface surface) {
    if (dpy == EGL_NO_DISPLAY || surface == EGL_NO_SURFACE) return false;
    std::lock_guard<std::mutex> lock(g_window_surfaces_mutex);
    // The slot is authoritative: that is the surface actually on screen. A
    // caller's handle is translated onto it, not trusted, so the only thing
    // worth validating is that a slot exists and belongs to this display.
    return g_live_window_surface != EGL_NO_SURFACE && g_slot_display == dpy;
}

EGLBoolean mg_egl_present(EGLDisplay dpy, EGLSurface surface) {
    EGLSurface slot_surface;
    {
        std::lock_guard<std::mutex> lock(g_window_surfaces_mutex);
        slot_surface = g_live_window_surface;
    }
    // MobileGL's SwapEGLBuffers rejects a draw surface that is not m_eglSurface.
    // Here the caller's handle is not rejected but TRANSLATED: this is a
    // pass-through library, so SDL keeps using handles it cached before the
    // surface was destroyed. Whatever it passes, what gets swapped is the slot.
    if (slot_surface != EGL_NO_SURFACE) surface = slot_surface;
    if (g_slot_display != EGL_NO_DISPLAY) dpy = g_slot_display;

    if (!ValidateSurfaceOnDisplay(dpy, surface)) {
        // No slot yet is the ordinary state during startup: extension probes and
        // shader compilation both run long before a window surface exists.
        if (g_live_window_surface == EGL_NO_SURFACE || g_slot_display == EGL_NO_DISPLAY) return EGL_FALSE;
        LOG_W_FORCE("present (MobileGL DirectGLES): refused — surface %p is not valid on dpy %p (EGL_BAD_SURFACE)",
                    surface, dpy);
        return EGL_FALSE;
    }

    LOAD_EGL(eglSwapBuffers)
    if (!egl_eglSwapBuffers) return EGL_FALSE;

    const unsigned long n = g_present_count.fetch_add(1, std::memory_order_relaxed) + 1;
    const EGLBoolean ok = egl_eglSwapBuffers(dpy, surface);
    if (n <= 3 || (n % 600) == 0) {
        LOG_W_FORCE("present (MobileGL DirectGLES): #%lu eglSwapBuffers(dpy=%p, surface=%p) -> %d", n, dpy, surface,
                    static_cast<int>(ok));
    }
    return ok;
}


const AppRenderTarget& mg_egl_app_target() {
    std::lock_guard<std::mutex> lock(g_target_mutex);
    return g_target;
}

unsigned mg_egl_app_target_generation() {
    return g_target_generation.load(std::memory_order_acquire);
}

void mg_egl_note_window_surface(EGLDisplay dpy, EGLConfig config, EGLSurface surface) {
    if (surface == EGL_NO_SURFACE) return;
    std::lock_guard<std::mutex> lock(g_target_mutex);
    if (g_target.have_surface && g_target.draw_surface == surface) return;

    const bool replaced = g_target.have_surface;
    g_target.display = dpy;
    g_target.config = config;
    g_target.draw_surface = surface;
    g_target.read_surface = surface;
    g_target.have_surface = true;
    g_target_generation.fetch_add(1, std::memory_order_acq_rel);
    LOG_W_FORCE("eglCreateWindowSurface: the application's window surface is now %p (config %p)%s", surface, config,
                replaced ? " — replacing a previously recorded surface" : "");
}

void mg_egl_note_make_current(EGLDisplay dpy, EGLSurface draw, EGLSurface read, EGLContext ctx, EGLBoolean ok) {
    if (ok != EGL_TRUE) return;
    std::lock_guard<std::mutex> lock(g_target_mutex);

    // Release: the application let go of the context on this thread. Nothing to
    // record beyond the fact, but it is logged — a release on the render thread
    // is what leaves it with no context and starts this whole search.
    if (ctx == EGL_NO_CONTEXT) {
        LOG_W_FORCE("eglMakeCurrent: the application released its context on pthread=%lu",
                    (unsigned long)pthread_self());
        return;
    }

    // Tracked live, on every call, rather than latched on the first.
    //
    // Latching the first binding was a mistake measured on real hardware. SDL
    // binds a context and a surface, then destroys that surface and creates
    // another — the log shows "reusing primary window" and a second
    // eglChooseConfig right after the first successful bind. Recording only the
    // first meant the recorded surface was dead by the time it was used:
    //
    //     eglMakeCurrent: recorded ... surface=0x77bbd1d380
    //     app context + recorded window surface: NOT BOUND (err=0x300d)
    //
    // 0x300d is EGL_BAD_SURFACE. The surface was real when bound and gone by
    // the time it was needed, so the fallback fell through to a surfaceless
    // bind and every draw went nowhere.
    const bool ctx_changed = !g_target.have_binding || g_target.context != ctx;
    const bool surf_changed = g_target.draw_surface != draw;

    g_target.display = dpy;
    g_target.context = ctx;
    g_target.have_binding = true;
    g_target.binding_thread = pthread_self();
    if (draw != EGL_NO_SURFACE) {
        g_target.draw_surface = draw;
        g_target.read_surface = read;
        g_target.have_surface = true;
    }

    if (ctx_changed || surf_changed) {
        g_target_generation.fetch_add(1, std::memory_order_acq_rel);
        LOG_W_FORCE("eglMakeCurrent: the application's live binding is now surface=%p ctx=%p (pthread=%lu)",
                    draw, ctx, (unsigned long)pthread_self());
    }

    if (draw == EGL_NO_SURFACE) {
        LOG_W_FORCE("eglMakeCurrent: the application bound a context with NO SURFACE — it cannot present. "
                    "Drawing will go nowhere unless a surface is bound later.");
    }
}

void mg_egl_note_destroy_surface(EGLDisplay dpy, EGLSurface surface) {
    if (surface == EGL_NO_SURFACE) return;
    std::lock_guard<std::mutex> lock(g_target_mutex);

    bool matched = false;
    if (g_target.have_surface && g_target.draw_surface == surface) {
        g_target.draw_surface = EGL_NO_SURFACE;
        g_target.read_surface = EGL_NO_SURFACE;
        g_target.have_surface = false;
        matched = true;
    }
    if (g_target.have_presenting && g_target.presenting_surface == surface) {
        g_target.presenting_surface = EGL_NO_SURFACE;
        g_target.presenting_thread = 0;
        g_target.have_presenting = false;
        matched = true;
    }
    if (!matched) return;

    g_target_generation.fetch_add(1, std::memory_order_acq_rel);
    LOG_W_FORCE("eglDestroySurface: the application destroyed surface %p, which was the recorded render target — "
                "forgotten it, so the next bind will not fail with EGL_BAD_SURFACE",
                surface);
}

void mg_egl_forget_surface(EGLSurface surface) {
    if (surface == EGL_NO_SURFACE) return;
    std::lock_guard<std::mutex> lock(g_target_mutex);

    bool matched = false;
    if (g_target.have_surface && g_target.draw_surface == surface) {
        g_target.draw_surface = EGL_NO_SURFACE;
        g_target.read_surface = EGL_NO_SURFACE;
        g_target.have_surface = false;
        matched = true;
    }
    if (g_target.have_presenting && g_target.presenting_surface == surface) {
        g_target.presenting_surface = EGL_NO_SURFACE;
        g_target.presenting_thread = 0;
        g_target.have_presenting = false;
        matched = true;
    }
    if (!matched) return;

    g_target_generation.fetch_add(1, std::memory_order_acq_rel);
    LOG_W_FORCE("forgot surface %p after the driver rejected it — it will not be used again", surface);
}

// The present fallback was removed, so every swap reaching this function now
// really is the application presenting — the line below can say so without
// qualification. It previously had a from_fallback parameter because the fallback
// called in here too, and the resulting "the application presents from" line
// appeared in logs where the application had in fact never presented. That
// misread sent several rounds of debugging down the wrong path.
void mg_egl_note_swap(EGLDisplay dpy, EGLSurface surface, EGLBoolean ok) {
    if (ok != EGL_TRUE || surface == EGL_NO_SURFACE) return;

    std::lock_guard<std::mutex> lock(g_target_mutex);

    // This is the surface the application really presents from. Recording it
    // here rather than in eglCreateWindowSurface is deliberate: the surface that
    // reaches the screen is created by a call this library does not intercept
    // (eglCreatePlatformWindowSurface), so the one recorded at creation time was
    // a different, discarded surface.
    if (!g_target.have_presenting || surface != g_target.presenting_surface) {
        const bool first = !g_target.have_presenting;
        g_target.presenting_surface = surface;
        g_target.have_presenting = true;
        g_target.presenting_thread = pthread_self();
        g_target.display = dpy;
        if (!g_target.have_surface) {
            g_target.draw_surface = surface;
            g_target.read_surface = surface;
            g_target.have_surface = true;
        }
        g_target_generation.fetch_add(1, std::memory_order_acq_rel);
        LOG_W_FORCE("eglSwapBuffers: this is the surface %s presents from: %p, presented by "
                    "pthread=%lu%s",
                    "the application", surface,
                    (unsigned long)pthread_self(), first ? "" : " (changed from the previous one)");
    } else if (g_target.presenting_thread != 0 && !pthread_equal(g_target.presenting_thread, pthread_self())) {
        // Worth reporting: if the presenting thread moves, the render thread
        // inferred from the first swap is no longer the one that matters.
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            LOG_W_FORCE("eglSwapBuffers: surface %p is being presented from pthread=%lu, but the first swap was "
                        "made by pthread=%lu",
                        surface, (unsigned long)pthread_self(), (unsigned long)g_target.presenting_thread);
        }
    }
}


namespace {

// EGL 1.5 entry points take EGLAttrib (pointer-sized, 64-bit on arm64); the
// EGL_EXT_platform_base ones this library actually loads take EGLint (32-bit).
// Casting the pointer would make the host read two 32-bit halves of each
// 64-bit value, so the values are copied one by one instead.
std::vector<EGLint> NarrowAttribs(const EGLAttrib* attribs) {
    std::vector<EGLint> out;
    if (!attribs) return out;
    for (size_t i = 0; attribs[i] != EGL_NONE; i += 2) {
        out.push_back(static_cast<EGLint>(attribs[i]));
        out.push_back(static_cast<EGLint>(attribs[i + 1]));
    }
    out.push_back(EGL_NONE);
    return out;
}

}  // namespace

extern "C"
{
#define EGL_API __attribute__((visibility("default")))
    EGL_API EGLint eglGetError(void) {
        mg_egl_note_call(__func__);
        LOG_D("eglGetError");
        LOAD_EGL(eglGetError)

        return egl_eglGetError();
    }
    EGL_API EGLDisplay eglGetDisplay(EGLNativeDisplayType display_id) {
        LOG_W_FORCE("EGL-TRACE: eglGetDisplay called");
        mg_egl_note_call(__func__);
        LOG_D("eglGetDisplay, display_id: %p", display_id);
        LOAD_EGL(eglGetDisplay)
        return egl_eglGetDisplay(display_id);
    }

    EGL_API EGLBoolean eglInitialize(EGLDisplay dpy, EGLint* major, EGLint* minor) {
        LOG_W_FORCE("EGL-TRACE: eglInitialize called (dpy=%p)", (void*)dpy);
        mg_egl_note_call(__func__);
        LOG_D("eglInitialize, dpy: %p, major: %p, minor: %p", dpy, major, minor);
        LOAD_EGL(eglInitialize)
        const EGLBoolean ok = egl_eglInitialize(dpy, major, minor);
        if (ok == EGL_TRUE) mg_egl_record_display(dpy);
        return ok;
    }

    EGL_API EGLBoolean eglTerminate(EGLDisplay dpy) {
        mg_egl_note_call(__func__);
        LOG_D("eglTerminate, dpy: %p", dpy);
        LOAD_EGL(eglTerminate)
        return egl_eglTerminate(dpy);
    }

    EGL_API const char* eglQueryString(EGLDisplay dpy, EGLint name) {
        mg_egl_note_call(__func__);
        LOAD_EGL(eglQueryString)
        const char* result = egl_eglQueryString ? egl_eglQueryString(dpy, name) : nullptr;
        // What the application learns here decides which surface creation path
        // it takes: an EGL 1.5 version string is what makes it call
        // eglCreatePlatformWindowSurface instead of eglCreateWindowSurface.
        LOG_W_FORCE("eglQueryString(dpy=%p, name=%d) -> %s", dpy, name, result ? result : "(null)");
        return result;
    }

    EGL_API EGLBoolean eglGetConfigs(EGLDisplay dpy, EGLConfig* configs, EGLint config_size, EGLint* num_config) {
        mg_egl_note_call(__func__);
        LOG_D("eglGetConfigs, dpy: %p, configs: %p, config_size: %d, num_config: %p", dpy, configs, config_size,
              num_config);
        LOAD_EGL(eglGetConfigs)
        return egl_eglGetConfigs(dpy, configs, config_size, num_config);
    }

    EGL_API EGLBoolean eglChooseConfig(EGLDisplay dpy, const EGLint* attrib_list, EGLConfig* configs,
                                       EGLint config_size, EGLint* num_config) {
        LOG_W_FORCE("EGL-TRACE: eglChooseConfig called (dpy=%p)", (void*)dpy);
        mg_egl_note_call(__func__);
        LOG_D("eglChooseConfig, dpy: %p, attrib_list: %p, configs: %p, config_size: "
              "%d, num_config: %p",
              dpy, attrib_list, configs, config_size, num_config);
        LOAD_EGL(eglChooseConfig)
        return egl_eglChooseConfig(dpy, attrib_list, configs, config_size, num_config);
    }

    EGL_API EGLBoolean eglGetConfigAttrib(EGLDisplay dpy, EGLConfig config, EGLint attribute, EGLint* value) {
        mg_egl_note_call(__func__);
        LOG_D("eglGetConfigAttrib, dpy: %p, config: %p, attribute: %d, value: %p", dpy, config, attribute, value);
        LOAD_EGL(eglGetConfigAttrib)
        return egl_eglGetConfigAttrib(dpy, config, attribute, value);
    }

    EGL_API EGLSurface eglCreateWindowSurface(EGLDisplay dpy, EGLConfig config, EGLNativeWindowType win,
                                              const EGLint* attrib_list) {
        mg_egl_note_call(__func__);
        LOG_D("eglCreateWindowSurface, dpy: %p, config: %p, win: %p, attrib_list: %p", dpy, config, win, attrib_list);
        LOAD_EGL(eglCreateWindowSurface)
        EGLSurface surf = egl_eglCreateWindowSurface(dpy, config, win, attrib_list);
        // MobileGL destroys the old surface before creating the next one, because
        // one ANativeWindow accepts only one EGLSurface. This library forwards
        // the application's calls, so it cannot reorder them — but it can recover
        // when the application got the order wrong. A failed creation here has
        // been seen on this driver as "eglCreateWindowSurface failed, reporting an
        // error of EGL_SUCCESS", i.e. EGL_NO_SURFACE with no error code, which is
        // exactly the signature of the native window still being attached.
        if (surf == EGL_NO_SURFACE) surf = RetryWindowSurfaceAfterReleasingOld(dpy, config, win, attrib_list);
        if (surf == EGL_NO_SURFACE) return EGL_NO_SURFACE;
        mg_egl_note_window_surface(dpy, config, surf);
        RecordWindowSurface(dpy, surf);
        mg_egl_activate_window_surface(dpy, surf);

        OwnedSurface rec;
        rec.dpy = dpy; rec.config = config; rec.win = win;
        rec.is_window = true; rec.host = surf;
        const EGLSurface handle = MintSurfaceHandle(rec);
        LOG_W_FORCE("eglCreateWindowSurface: host surface %p — handing the application handle %p (ours, not the "
                    "host's, so it stays valid)",
                    surf, handle);
        return handle;
    }

    // EGL 1.5 platform surface creation.
    //
    // This was declared in loader.h but never exported, and that gap is the
    // most likely reason the picture never appears. It was measured once, in a
    // session that logged a recorded window surface of 0x77c45f3d80 and then
    // 1251 swaps against 0x7806563d80 — two different surfaces. The second one
    // could not be seen from anywhere in this library, because the only call
    // that could have created it is this one, and it was not exported.
    //
    // The consequence is not a failure but a silent mismatch: the fallback
    // binds the render thread to the one surface it can see, the application
    // presents the other, and every frame is drawn to a surface that is never
    // shown. No error is raised at any point.
    EGL_API EGLSurface eglCreatePlatformWindowSurface(EGLDisplay dpy, EGLConfig config, void* native_window,
                                                      const EGLAttrib* attrib_list) {
        mg_egl_note_call(__func__);
        LOG_D("eglCreatePlatformWindowSurface, dpy: %p, config: %p, native_window: %p", dpy, config, native_window);
        LOAD_EGL(eglCreatePlatformWindowSurface)
        if (!egl_eglCreatePlatformWindowSurface) {
            // EGL 1.4 host: fall back to the classic entry point rather than
            // returning EGL_NO_SURFACE, which the application would read as a
            // hard failure.
            LOG_W_FORCE("eglCreatePlatformWindowSurface: unavailable, using eglCreateWindowSurface");
            return eglCreateWindowSurface(dpy, config,
                                          reinterpret_cast<EGLNativeWindowType>(native_window), nullptr);
        }
        const std::vector<EGLint> narrow = NarrowAttribs(attrib_list);
        EGLSurface surf =
            egl_eglCreatePlatformWindowSurface(dpy, config, native_window, narrow.empty() ? nullptr : narrow.data());
        if (surf == EGL_NO_SURFACE) return EGL_NO_SURFACE;
        mg_egl_note_window_surface(dpy, config, surf);
        RecordWindowSurface(dpy, surf);
        mg_egl_activate_window_surface(dpy, surf);

        OwnedSurface rec;
        rec.dpy = dpy; rec.config = config;
        rec.win = reinterpret_cast<EGLNativeWindowType>(native_window);
        rec.is_window = true; rec.host = surf;
        const EGLSurface handle = MintSurfaceHandle(rec);
        LOG_W_FORCE("eglCreatePlatformWindowSurface: host surface %p — handing the application handle %p", surf, handle);
        return handle;
    }

    EGL_API EGLSurface eglCreatePbufferSurface(EGLDisplay dpy, EGLConfig config, const EGLint* attrib_list) {
        mg_egl_note_call(__func__);
        LOG_D("eglCreatePbufferSurface, dpy: %p, config: %p, attrib_list: %p", dpy, config, attrib_list);
        LOAD_EGL(eglCreatePbufferSurface)
        const EGLSurface surf = egl_eglCreatePbufferSurface(dpy, config, attrib_list);
        if (surf == EGL_NO_SURFACE) return EGL_NO_SURFACE;
        OwnedSurface rec;
        rec.dpy = dpy; rec.config = config; rec.host = surf;
        return MintSurfaceHandle(rec);
    }

    EGL_API EGLSurface eglCreatePixmapSurface(EGLDisplay dpy, EGLConfig config, EGLNativePixmapType pixmap,
                                              const EGLint* attrib_list) {
        mg_egl_note_call(__func__);
        LOG_D("eglCreatePixmapSurface, dpy: %p, config: %p, pixmap: %p, attrib_list: "
              "%p",
              dpy, config, pixmap, attrib_list);
        LOAD_EGL(eglCreatePixmapSurface)
        const EGLSurface surf = egl_eglCreatePixmapSurface(dpy, config, pixmap, attrib_list);
        if (surf == EGL_NO_SURFACE) return EGL_NO_SURFACE;
        OwnedSurface rec;
        rec.dpy = dpy; rec.config = config; rec.host = surf;
        return MintSurfaceHandle(rec);
    }

    EGL_API EGLBoolean eglDestroySurface(EGLDisplay dpy, EGLSurface surface) {
        mg_egl_note_call(__func__);
        LOG_D("eglDestroySurface, dpy: %p, surface: %p", dpy, surface);

        // A window surface must really be destroyed: the native window behind it
        // can only be attached to one EGLSurface at a time, and SDL creates the
        // next surface for the SAME native window right after this. Keeping the
        // old one alive made that next eglCreateWindowSurface fail.
        //
        // Its handle value is remembered so operations SDL still issues on the
        // stale handle can be redirected to the live surface.
        EGLSurface host = EGL_NO_SURFACE;
        bool owned = false;
        {
            std::lock_guard<std::mutex> lock(g_owned_surfaces_mutex);
            auto it = g_owned_surfaces.find(surface);
            if (it != g_owned_surfaces.end()) {
                owned = true;
                host = it->second.host;
                it->second.host_destroyed = true;
            }
        }
        if (!owned) {
            // Not minted here: an older caller passing a raw host handle.
            const bool known = ForgetWindowSurface(surface);
            LOAD_EGL(eglDestroySurface)
            EGLBoolean ok = egl_eglDestroySurface(dpy, surface);
            if (ok == EGL_TRUE) {
                if (known) LOG_W_FORCE("eglDestroySurface(%p): raw host surface destroyed", surface);
                mg_egl_note_destroy_surface(dpy, surface);
            }
            return ok;
        }
        if (host != EGL_NO_SURFACE) {
            LOAD_EGL(eglDestroySurface)
            egl_eglDestroySurface(dpy, host);
            ForgetWindowSurface(host);
            mg_egl_note_destroy_surface(dpy, host);
            LOG_W_FORCE("eglDestroySurface(%p): host surface %p destroyed; the handle the application holds stays "
                        "valid and will be re-created on next use",
                        surface, host);
        }
        return EGL_TRUE;
    }

    EGL_API EGLBoolean eglQuerySurface(EGLDisplay dpy, EGLSurface surface, EGLint attribute, EGLint* value) {
        LOG_W_FORCE("EGL-TRACE: eglQuerySurface called (dpy=%p)", (void*)dpy);
        mg_egl_note_call(__func__);
        LOG_D("eglQuerySurface, dpy: %p, surface: %p, attribute: %d, value: %p", dpy, surface, attribute, value);
        LOAD_EGL(eglQuerySurface)
        return egl_eglQuerySurface(dpy, EnsureHostSurface(surface), attribute, value);
    }

    EGL_API EGLBoolean eglBindAPI(EGLenum api) {
        mg_egl_note_call(__func__);
        LOG_D("eglBindAPI, api: %d", api);
        LOAD_EGL(eglBindAPI)
        return egl_eglBindAPI(api);
    }

    EGL_API EGLenum eglQueryAPI(void) {
        mg_egl_note_call(__func__);
        LOG_D("eglQueryAPI");
        LOAD_EGL(eglQueryAPI)
        return egl_eglQueryAPI();
    }

    EGL_API EGLBoolean eglWaitClient(void) {
        mg_egl_note_call(__func__);
        LOG_D("eglWaitClient");
        LOAD_EGL(eglWaitClient)
        return egl_eglWaitClient();
    }

    EGL_API EGLBoolean eglReleaseThread(void) {
        mg_egl_note_call(__func__);
        LOG_D("eglReleaseThread");
        LOAD_EGL(eglReleaseThread)
        return egl_eglReleaseThread();
    }

    EGL_API EGLSurface eglCreatePbufferFromClientBuffer(EGLDisplay dpy, EGLenum buftype, EGLClientBuffer buffer,
                                                        EGLConfig config, const EGLint* attrib_list) {
        mg_egl_note_call(__func__);
        LOG_D("eglCreatePbufferFromClientBuffer, dpy: %p, buftype: %d, buffer: %p, "
              "config: %p, attrib_list: %p",
              dpy, buftype, buffer, config, attrib_list);
        LOAD_EGL(eglCreatePbufferFromClientBuffer)
        return egl_eglCreatePbufferFromClientBuffer(dpy, buftype, buffer, config, attrib_list);
    }

    EGL_API EGLBoolean eglSurfaceAttrib(EGLDisplay dpy, EGLSurface surface, EGLint attribute, EGLint value) {
        mg_egl_note_call(__func__);
        LOG_D("eglSurfaceAttrib, dpy: %p, surface: %p, attribute: %d, value: %d", dpy, surface, attribute, value);
        LOAD_EGL(eglSurfaceAttrib)
        return egl_eglSurfaceAttrib(dpy, EnsureHostSurface(surface), attribute, value);
    }

    EGL_API EGLBoolean eglBindTexImage(EGLDisplay dpy, EGLSurface surface, EGLint buffer) {
        mg_egl_note_call(__func__);
        LOG_D("eglBindTexImage, dpy: %p, surface: %p, buffer: %d", dpy, surface, buffer);
        LOAD_EGL(eglBindTexImage)
        return egl_eglBindTexImage(dpy, EnsureHostSurface(surface), buffer);
    }

    EGL_API EGLBoolean eglReleaseTexImage(EGLDisplay dpy, EGLSurface surface, EGLint buffer) {
        mg_egl_note_call(__func__);
        LOG_D("eglReleaseTexImage, dpy: %p, surface: %p, buffer: %d", dpy, surface, buffer);
        LOAD_EGL(eglReleaseTexImage)
        return egl_eglReleaseTexImage(dpy, EnsureHostSurface(surface), buffer);
    }

    EGL_API EGLBoolean eglSwapInterval(EGLDisplay dpy, EGLint interval) {
        LOG_W_FORCE("EGL-TRACE: eglSwapInterval called (dpy=%p)", (void*)dpy);
        mg_egl_note_call(__func__);
        LOG_D("eglSwapInterval, dpy: %p, interval: %d", dpy, interval);
        LOAD_EGL(eglSwapInterval)
        return egl_eglSwapInterval(dpy, interval);
    }

    EGL_API EGLContext eglCreateContext(EGLDisplay dpy, EGLConfig config, EGLContext share_context,
                                        const EGLint* attrib_list) {
        LOG_W_FORCE("EGL-TRACE: eglCreateContext called (dpy=%p)", (void*)dpy);
        mg_egl_note_call(__func__);
        LOG_D("eglCreateContext, dpy: %p, config: %p, share_context: %p, "
              "attrib_list: %p",
              dpy, config, share_context, attrib_list);
        LOAD_EGL(eglCreateContext)
        return egl_eglCreateContext(dpy, config, share_context, attrib_list);
    }

    EGL_API EGLBoolean eglDestroyContext(EGLDisplay dpy, EGLContext ctx) {
        mg_egl_note_call(__func__);
        LOG_D("eglDestroyContext, dpy: %p, ctx: %p", dpy, ctx);
        LOAD_EGL(eglDestroyContext)
        return egl_eglDestroyContext(dpy, ctx);
    }

    EGL_API EGLBoolean eglMakeCurrent(EGLDisplay dpy, EGLSurface draw, EGLSurface read, EGLContext ctx) {
        mg_egl_note_call(__func__);
        LOG_D("eglMakeCurrent, dpy: %p, draw: %p, read: %p, ctx: %p", dpy, draw, read, ctx);
        // Same stale-handle problem as the swap: SDL binds the surface it cached
        // when it created the window, which may have been destroyed since.
        // Refuse to let go of the render target.
        //
        // SDL3 (src/video/SDL_video.c, SDL_GL_MakeCurrent) decides which window
        // may be swapped from a thread-local:
        //
        //     if (!context) { window = NULL; }
        //     result = _this->GL_MakeCurrent(_this, window, context);
        //     if (result) {
        //         _this->current_glwin = window;
        //         SDL_SetTLS(&_this->current_glwin_tls, window, NULL);
        //     }
        //
        // so a release that SUCCEEDS sets current_glwin to NULL. And
        // SDL_GL_SwapWindow refuses to swap unless it matches:
        //
        //     if (SDL_GL_GetCurrentWindow() != window) return SDL_SetError(...)
        //
        // This session does exactly that: the launcher's "primary window reuse"
        // hook released the context (logged at line 327), a new window surface
        // was then created (line 340), and no eglMakeCurrent ever followed.
        // current_glwin stayed NULL, so every swap was dropped inside SDL and
        // eglSwapBuffers never reached this library — 324653 GL calls ran while
        // the swap count stayed at zero. Sound and input were unaffected because
        // they do not go through the swap.
        //
        // Returning EGL_FALSE makes GL_MakeCurrent fail, so SDL leaves
        // current_glwin alone and the reused window stays current. The swap then
        // passes the check and reaches this library.
        const bool is_release = (draw == EGL_NO_SURFACE && read == EGL_NO_SURFACE && ctx == EGL_NO_CONTEXT);
        if (is_release && global_settings.keep_current_on_release) {
            EGLSurface live = EGL_NO_SURFACE;
            {
                std::lock_guard<std::mutex> lock(g_window_surfaces_mutex);
                live = g_live_window_surface;
            }
            if (live != EGL_NO_SURFACE) {
                LOG_W_FORCE("eglMakeCurrent: refusing to release the context (keepCurrentOnRelease=1) — SDL would "
                            "clear its current window and drop every later swap, so the binding to %p is kept",
                            live);
                return EGL_FALSE;
            }
        }

        const EGLSurface host_draw = ResolveWindowSurface(dpy, EnsureHostSurface(draw));
        const EGLSurface host_read = ResolveWindowSurface(dpy, EnsureHostSurface(read));

        LOAD_EGL(eglMakeCurrent)
        EGLBoolean ok = egl_eglMakeCurrent(dpy, host_draw, host_read, ctx);
        if (ok != EGL_TRUE) {
            // Previously invisible. A refused bind is exactly the kind of thing
            // that leaves the application unable to present, and it was being
            // dropped on the floor with no trace.
            LOAD_EGL(eglGetError)
            LOG_W_FORCE("eglMakeCurrent: the application's bind FAILED (0x%x) — dpy=%p draw=%p ctx=%p",
                        egl_eglGetError(), dpy, host_draw, ctx);
        }
        mg_egl_note_make_current(dpy, host_draw, host_read, ctx, ok);
        if (ok == EGL_TRUE) {
            RecordCurrentWindowSurface(host_draw);
            t_current_draw_handle = draw;
            t_current_read_handle = read;
        }
        return ok;
    }

    EGL_API EGLContext eglGetCurrentContext(void) {
        mg_egl_note_call(__func__);
        LOG_D("eglGetCurrentContext");
        LOAD_EGL(eglGetCurrentContext)
        return egl_eglGetCurrentContext();
    }

    EGL_API EGLSurface eglGetCurrentSurface(EGLint readdraw) {
        mg_egl_note_call(__func__);
        LOG_D("eglGetCurrentSurface, readdraw: %d", readdraw);
        LOAD_EGL(eglGetCurrentSurface)
        const EGLSurface theirs = egl_eglGetCurrentSurface(readdraw);
        // Answer with the handle the application was given, not the host's.
        const EGLSurface mine = (readdraw == EGL_READ) ? t_current_read_handle : t_current_draw_handle;
        if (mine != EGL_NO_SURFACE) {
            std::lock_guard<std::mutex> lock(g_owned_surfaces_mutex);
            auto it = g_owned_surfaces.find(mine);
            if (it != g_owned_surfaces.end() && it->second.host == theirs) return mine;
        }
        return theirs;
    }

    EGL_API EGLDisplay eglGetCurrentDisplay(void) {
        mg_egl_note_call(__func__);
        LOG_D("eglGetCurrentDisplay");
        LOAD_EGL(eglGetCurrentDisplay)
        return egl_eglGetCurrentDisplay();
    }

    EGL_API EGLBoolean eglQueryContext(EGLDisplay dpy, EGLContext ctx, EGLint attribute, EGLint* value) {
        mg_egl_note_call(__func__);
        LOG_D("eglQueryContext, dpy: %p, ctx: %p, attribute: %d, value: %p", dpy, ctx, attribute, value);
        LOAD_EGL(eglQueryContext)
        return egl_eglQueryContext(dpy, ctx, attribute, value);
    }

    EGL_API EGLBoolean eglWaitGL(void) {
        mg_egl_note_call(__func__);
        LOG_D("eglWaitGL");
        LOAD_EGL(eglWaitGL)
        return egl_eglWaitGL();
    }

    EGL_API EGLBoolean eglWaitNative(EGLint engine) {
        mg_egl_note_call(__func__);
        LOG_D("eglWaitNative, engine: %d", engine);
        LOAD_EGL(eglWaitNative)
        return egl_eglWaitNative(engine);
    }

    EGL_API EGLBoolean eglSwapBuffers(EGLDisplay dpy, EGLSurface surface) {
        mg_egl_note_call(__func__);
        // Forced, rate-limited. The whole point is to know whether the swap ever
        // reaches this library at all, and debug logging is off in release
        // builds — the one run where it is needed is the run where nothing else
        // is printed.
        {
            static std::atomic<unsigned long> count{0};
            const unsigned long n = count.fetch_add(1, std::memory_order_relaxed);
            if (n < 5 || (n % 600) == 0) {
                LOG_W_FORCE("EGL-TRACE: eglSwapBuffers #%lu dpy=%p surface=%p", n, dpy, surface);
            }
        }
        LOG_D("eglSwapBuffers, dpy: %p, surface: %p", dpy, surface);
        // SDL presents through the handle cached in its own window, which is a
        // surface it has already asked to destroy. Redirect it to the live one
        // so this present actually lands.
        surface = ResolveWindowSurface(dpy, EnsureHostSurface(surface));
        // FSR upscales the finished frame, so it has to run BEFORE the swap.
        // It used to run after one swap and then swap again, which put two
        // presents on the same surface per frame — one without FSR and one
        // with it — and showed as the picture snapping back and forth.
        if (global_settings.fsr1_setting != FSR1_Quality_Preset::Disabled) {
            ApplyFSR();
            CheckResolutionChange();
        }
        const EGLBoolean result = mg_egl_present(dpy, surface);
        if (result != EGL_TRUE) {
            LOAD_EGL(eglGetError)
            LOG_W_FORCE("eglSwapBuffers: FAILED (0x%x) surface=%p", egl_eglGetError(), surface);
        }
        // The application presented on its own, so the present fallback has
        // nothing left to do. Turned off permanently: if a real swap chain is
        // working, adding a second one would only cause tearing.
        mg_egl_note_swap(dpy, surface, result);
        return result;
    }

    // Partial-swap variants.
    //
    // These were declared in the loader but never exported, and that gap
    // explains a measurement that otherwise made no sense. On the device the
    // watchdog reported a sustained few thousand guarded GL calls per second —
    // a render loop, plainly running — while reporting, at the same time, that
    // eglSwapBuffers had never once been called.
    //
    // If the application presents through one of these instead, every swap goes
    // straight to the host: the presenting surface is never recorded, nothing
    // here can tell where the frames are going, and the fallback has no way to
    // know which surface the picture depends on. Exporting them makes those
    // swaps visible and feeds the same tracking as eglSwapBuffers.
    EGL_API EGLBoolean eglSwapBuffersWithDamageEXT(EGLDisplay dpy, EGLSurface surface, const EGLint* rects,
                                                   EGLint n_rects) {
        LOG_W_FORCE("EGL-TRACE: eglSwapBuffersWithDamageEXT called surface=%p n_rects=%d", surface, n_rects);
        mg_egl_note_call(__func__);
        LOG_D("eglSwapBuffersWithDamageEXT, dpy: %p, surface: %p, n_rects: %d", dpy, surface, n_rects);
        surface = ResolveWindowSurface(dpy, EnsureHostSurface(surface));
        LOAD_EGL(eglSwapBuffersWithDamageEXT)
        if (!egl_eglSwapBuffersWithDamageEXT) {
            // Not available: present through the one path instead of opening a
            // second one by calling back into this library's own entry point.
            LOG_W_FORCE("eglSwapBuffersWithDamageEXT: unavailable, presenting through the single path for surface %p",
                        surface);
            return mg_egl_present(dpy, surface);
        }
        const EGLBoolean result = mg_egl_present(dpy, surface);
        if (result != EGL_TRUE) {
            LOAD_EGL(eglGetError)
            LOG_W_FORCE("eglSwapBuffersWithDamageEXT: FAILED (0x%x) surface=%p", egl_eglGetError(), surface);
        }
        // The application presented on its own, so the present fallback must
        // stop — same as in eglSwapBuffers. Omitting this was the direct cause
        // of the "picture rolls back" flicker: once self-promotion made the
        // application's swap resolve to this library, SDL presented through
        // this entry point while the fallback was still presenting from the
        // drawing thread, so each frame reached the screen twice at different
        // points in time. A static screen hides it (both presents show the same
        // content); the moment the camera moves, the two presents carry
        // different frames and the view snaps backwards.
        mg_egl_note_swap(dpy, surface, result);
        return result;
    }

    EGL_API EGLBoolean eglSwapBuffersWithDamageKHR(EGLDisplay dpy, EGLSurface surface, const EGLint* rects,
                                                   EGLint n_rects) {
        LOG_W_FORCE("EGL-TRACE: eglSwapBuffersWithDamageKHR called surface=%p n_rects=%d", surface, n_rects);
        mg_egl_note_call(__func__);
        LOG_D("eglSwapBuffersWithDamageKHR, dpy: %p, surface: %p, n_rects: %d", dpy, surface, n_rects);
        surface = ResolveWindowSurface(dpy, EnsureHostSurface(surface));
        LOAD_EGL(eglSwapBuffersWithDamageKHR)
        if (!egl_eglSwapBuffersWithDamageKHR) {
            LOG_W_FORCE("eglSwapBuffersWithDamageKHR: unavailable, presenting through the single path for surface %p",
                        surface);
            return mg_egl_present(dpy, surface);
        }
        const EGLBoolean result = mg_egl_present(dpy, surface);
        if (result != EGL_TRUE) {
            LOAD_EGL(eglGetError)
            LOG_W_FORCE("eglSwapBuffersWithDamageKHR: FAILED (0x%x) surface=%p", egl_eglGetError(), surface);
        }
        // See the note in eglSwapBuffersWithDamageEXT: stopping the present
        // fallback here is what keeps two swap chains from running at once.
        mg_egl_note_swap(dpy, surface, result);
        return result;
    }

    EGL_API EGLBoolean eglCopyBuffers(EGLDisplay dpy, EGLSurface surface, EGLNativePixmapType target) {
        mg_egl_note_call(__func__);
        LOG_D("eglCopyBuffers, dpy: %p, surface: %p, target: %p", dpy, surface, target);
        LOAD_EGL(eglCopyBuffers)
        return egl_eglCopyBuffers(dpy, EnsureHostSurface(surface), target);
    }

    EGL_API EGLDisplay eglGetPlatformDisplay(EGLenum platform, void* native_display, const EGLAttrib* attrib_list) {
        mg_egl_note_call(__func__);
        LOG_D("eglGetPlatformDisplay, platform: %d, native_display: %p, attrib_list: "
              "%p",
              platform, native_display, attrib_list);
        LOAD_EGL(eglGetPlatformDisplay)
        if (!egl_eglGetPlatformDisplay) return EGL_NO_DISPLAY;
        const std::vector<EGLint> narrow = NarrowAttribs(attrib_list);
        return egl_eglGetPlatformDisplay(platform, native_display, narrow.empty() ? nullptr : narrow.data());
    }

    // The EXT variant takes EGLint attributes rather than EGLAttrib ones.
    //
    // Exported for the same reason as eglCreatePlatformWindowSurface below:
    // this was declared but never exported, so an application resolving it
    // through eglGetProcAddress — which forwards to dlsym(RTLD_DEFAULT, name) —
    // got the HOST driver's version instead of this library's.
    EGL_API EGLDisplay eglGetPlatformDisplayEXT(EGLenum platform, void* native_display, const EGLint* attrib_list) {
        mg_egl_note_call(__func__);
        LOG_W_FORCE("eglGetPlatformDisplayEXT, platform: %d, native_display: %p", platform, native_display);
        LOAD_EGL(eglGetPlatformDisplayEXT)
        if (egl_eglGetPlatformDisplayEXT) {
            return egl_eglGetPlatformDisplayEXT(platform, native_display, attrib_list);
        }
        // Host lacks it: route through the EGL 1.5 entry point, which takes
        // the wider attribute type. Widening EGLint -> EGLAttrib is lossless.
        LOAD_EGL(eglGetPlatformDisplay)
        if (!egl_eglGetPlatformDisplay) return EGL_NO_DISPLAY;
        return egl_eglGetPlatformDisplay(platform, native_display, attrib_list);
    }

        // ---------------------------------------------------------------------------
    // eglGetProcAddress: what this library hands back
    //
    // It used to forward everything to glXGetProcAddress, which is a bare
    // dlsym(RTLD_DEFAULT, name). RTLD_DEFAULT searches only the GLOBAL symbol
    // scope, and a launcher that loads this library with dlopen(path,
    // RTLD_LOCAL) — the default — puts none of its symbols there. The lookup
    // then falls through to the host driver's libEGL.so and hands the
    // application the HOST's function, so that call bypasses this library
    // completely: no log, no tracking, no wrapper.
    //
    // The measurement says exactly that is happening. A session logged 297k
    // guarded GL calls in 20 seconds at roughly 878 draws per second — the
    // render loop running flat out — while eglSwapBuffers was never called
    // once. Whatever the application uses to present is not reaching this
    // library, and eglGetProcAddress is the only place it could have obtained
    // such a pointer.
    //
    // MobileGL gets this right by construction: GetProcAddress.cpp answers
    // through an explicit GETPROC table of its own entry points. This is the
    // same fix.
    //
    // Answering for our own entry points first is also just correct: a library
    // that intercepts EGL must hand out the interception, not the thing it
    // wraps, or the interception only works for callers that link directly.
    // ---------------------------------------------------------------------------
    namespace {
    struct EglExport {
        const char* name;
        void* address;
    };

    const EglExport kEglExports[] = {
        {"eglBindAPI", reinterpret_cast<void*>(&::eglBindAPI)},
        {"eglBindTexImage", reinterpret_cast<void*>(&::eglBindTexImage)},
        {"eglChooseConfig", reinterpret_cast<void*>(&::eglChooseConfig)},
        {"eglCopyBuffers", reinterpret_cast<void*>(&::eglCopyBuffers)},
        {"eglCreateContext", reinterpret_cast<void*>(&::eglCreateContext)},
        {"eglCreatePbufferFromClientBuffer", reinterpret_cast<void*>(&::eglCreatePbufferFromClientBuffer)},
        {"eglCreatePbufferSurface", reinterpret_cast<void*>(&::eglCreatePbufferSurface)},
        {"eglCreatePixmapSurface", reinterpret_cast<void*>(&::eglCreatePixmapSurface)},
        {"eglCreatePlatformWindowSurface", reinterpret_cast<void*>(&::eglCreatePlatformWindowSurface)},
        {"eglCreateWindowSurface", reinterpret_cast<void*>(&::eglCreateWindowSurface)},
        {"eglDestroyContext", reinterpret_cast<void*>(&::eglDestroyContext)},
        {"eglDestroySurface", reinterpret_cast<void*>(&::eglDestroySurface)},
        {"eglGetConfigAttrib", reinterpret_cast<void*>(&::eglGetConfigAttrib)},
        {"eglGetConfigs", reinterpret_cast<void*>(&::eglGetConfigs)},
        {"eglGetCurrentContext", reinterpret_cast<void*>(&::eglGetCurrentContext)},
        {"eglGetCurrentDisplay", reinterpret_cast<void*>(&::eglGetCurrentDisplay)},
        {"eglGetCurrentSurface", reinterpret_cast<void*>(&::eglGetCurrentSurface)},
        {"eglGetDisplay", reinterpret_cast<void*>(&::eglGetDisplay)},
        {"eglGetError", reinterpret_cast<void*>(&::eglGetError)},
        {"eglGetPlatformDisplay", reinterpret_cast<void*>(&::eglGetPlatformDisplay)},
        {"eglGetPlatformDisplayEXT", reinterpret_cast<void*>(&::eglGetPlatformDisplayEXT)},
        {"eglGetProcAddress", reinterpret_cast<void*>(&::eglGetProcAddress)},
        {"eglInitialize", reinterpret_cast<void*>(&::eglInitialize)},
        {"eglMakeCurrent", reinterpret_cast<void*>(&::eglMakeCurrent)},
        {"eglQueryAPI", reinterpret_cast<void*>(&::eglQueryAPI)},
        {"eglQueryContext", reinterpret_cast<void*>(&::eglQueryContext)},
        {"eglQueryString", reinterpret_cast<void*>(&::eglQueryString)},
        {"eglQuerySurface", reinterpret_cast<void*>(&::eglQuerySurface)},
        {"eglReleaseTexImage", reinterpret_cast<void*>(&::eglReleaseTexImage)},
        {"eglReleaseThread", reinterpret_cast<void*>(&::eglReleaseThread)},
        {"eglSurfaceAttrib", reinterpret_cast<void*>(&::eglSurfaceAttrib)},
        {"eglSwapBuffers", reinterpret_cast<void*>(&::eglSwapBuffers)},
        {"eglSwapBuffersWithDamageEXT", reinterpret_cast<void*>(&::eglSwapBuffersWithDamageEXT)},
        {"eglSwapBuffersWithDamageKHR", reinterpret_cast<void*>(&::eglSwapBuffersWithDamageKHR)},
        {"eglSwapInterval", reinterpret_cast<void*>(&::eglSwapInterval)},
        {"eglTerminate", reinterpret_cast<void*>(&::eglTerminate)},
        {"eglWaitClient", reinterpret_cast<void*>(&::eglWaitClient)},
        {"eglWaitGL", reinterpret_cast<void*>(&::eglWaitGL)},
        {"eglWaitNative", reinterpret_cast<void*>(&::eglWaitNative)},
    };
    }  // namespace

EGL_API EGLAPI __eglMustCastToProperFunctionPointerType EGLAPIENTRY eglGetProcAddress(const char* procname) {
        mg_egl_note_call(__func__);
        if (procname) {
            for (const auto& entry : kEglExports) {
                if (strcmp(procname, entry.name) == 0) {
                    if (!global_settings.proc_address_own) {
                        // Off: answer from the host, as this library did before
                        // the change. Every call through the returned pointer then
                        // bypasses this library entirely.
                        LOG_W_FORCE("eglGetProcAddress(%s) -> HOST driver (procAddressOwn=0)", procname);
                        void* host = glXGetProcAddress(procname);
                        return reinterpret_cast<__eglMustCastToProperFunctionPointerType>(host);
                    }
                    LOG_W_FORCE("eglGetProcAddress(%s) -> this library's own entry point", procname);
                    return reinterpret_cast<__eglMustCastToProperFunctionPointerType>(entry.address);
                }
            }
        }
        void* host = glXGetProcAddress(procname);
        LOG_W_FORCE("eglGetProcAddress(%s) -> %s%s", procname ? procname : "(null)",
                    host ? "HOST driver" : "null",
                    host ? " — calls through this pointer bypass this library" : "");
        return reinterpret_cast<__eglMustCastToProperFunctionPointerType>(host);
    }
}