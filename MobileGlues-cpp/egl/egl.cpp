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

} // namespace

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
        LOG_W_FORCE("eglSwapBuffers: this is the surface the application presents from: %p, presented by "
                    "pthread=%lu%s",
                    surface, (unsigned long)pthread_self(), first ? "" : " (changed from the previous one)");
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
        return egl_eglInitialize(dpy, major, minor);
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
        mg_egl_note_window_surface(dpy, config, surf);
        mg_egl_activate_window_surface(dpy, surf);
        return surf;
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
        mg_egl_note_window_surface(dpy, config, surf);
        mg_egl_activate_window_surface(dpy, surf);
        return surf;
    }

    EGL_API EGLSurface eglCreatePbufferSurface(EGLDisplay dpy, EGLConfig config, const EGLint* attrib_list) {
        mg_egl_note_call(__func__);
        LOG_D("eglCreatePbufferSurface, dpy: %p, config: %p, attrib_list: %p", dpy, config, attrib_list);
        LOAD_EGL(eglCreatePbufferSurface)
        return egl_eglCreatePbufferSurface(dpy, config, attrib_list);
    }

    EGL_API EGLSurface eglCreatePixmapSurface(EGLDisplay dpy, EGLConfig config, EGLNativePixmapType pixmap,
                                              const EGLint* attrib_list) {
        mg_egl_note_call(__func__);
        LOG_D("eglCreatePixmapSurface, dpy: %p, config: %p, pixmap: %p, attrib_list: "
              "%p",
              dpy, config, pixmap, attrib_list);
        LOAD_EGL(eglCreatePixmapSurface)
        return egl_eglCreatePixmapSurface(dpy, config, pixmap, attrib_list);
    }

    EGL_API EGLBoolean eglDestroySurface(EGLDisplay dpy, EGLSurface surface) {
        mg_egl_note_call(__func__);
        LOG_D("eglDestroySurface, dpy: %p, surface: %p", dpy, surface);
        LOAD_EGL(eglDestroySurface)
        EGLBoolean ok = egl_eglDestroySurface(dpy, surface);
        if (ok == EGL_TRUE) mg_egl_note_destroy_surface(dpy, surface);
        return ok;
    }

    EGL_API EGLBoolean eglQuerySurface(EGLDisplay dpy, EGLSurface surface, EGLint attribute, EGLint* value) {
        LOG_W_FORCE("EGL-TRACE: eglQuerySurface called (dpy=%p)", (void*)dpy);
        mg_egl_note_call(__func__);
        LOG_D("eglQuerySurface, dpy: %p, surface: %p, attribute: %d, value: %p", dpy, surface, attribute, value);
        LOAD_EGL(eglQuerySurface)
        return egl_eglQuerySurface(dpy, surface, attribute, value);
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
        return egl_eglSurfaceAttrib(dpy, surface, attribute, value);
    }

    EGL_API EGLBoolean eglBindTexImage(EGLDisplay dpy, EGLSurface surface, EGLint buffer) {
        mg_egl_note_call(__func__);
        LOG_D("eglBindTexImage, dpy: %p, surface: %p, buffer: %d", dpy, surface, buffer);
        LOAD_EGL(eglBindTexImage)
        return egl_eglBindTexImage(dpy, surface, buffer);
    }

    EGL_API EGLBoolean eglReleaseTexImage(EGLDisplay dpy, EGLSurface surface, EGLint buffer) {
        mg_egl_note_call(__func__);
        LOG_D("eglReleaseTexImage, dpy: %p, surface: %p, buffer: %d", dpy, surface, buffer);
        LOAD_EGL(eglReleaseTexImage)
        return egl_eglReleaseTexImage(dpy, surface, buffer);
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
        LOAD_EGL(eglMakeCurrent)
        EGLBoolean ok = egl_eglMakeCurrent(dpy, draw, read, ctx);
        if (ok != EGL_TRUE) {
            // Previously invisible. A refused bind is exactly the kind of thing
            // that leaves the application unable to present, and it was being
            // dropped on the floor with no trace.
            LOAD_EGL(eglGetError)
            LOG_W_FORCE("eglMakeCurrent: the application's bind FAILED (0x%x) — dpy=%p draw=%p ctx=%p",
                        egl_eglGetError(), dpy, draw, ctx);
        }
        mg_egl_note_make_current(dpy, draw, read, ctx, ok);
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
        return egl_eglGetCurrentSurface(readdraw);
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
        LOAD_EGL(eglSwapBuffers)
        EGLBoolean result;
        if (global_settings.fsr1_setting != FSR1_Quality_Preset::Disabled) {
            ApplyFSR();
            result = egl_eglSwapBuffers(dpy, surface);
            CheckResolutionChange();
        } else {
            result = egl_eglSwapBuffers(dpy, surface);
        }
        if (result != EGL_TRUE) {
            LOAD_EGL(eglGetError)
            LOG_W_FORCE("eglSwapBuffers: FAILED (0x%x) surface=%p", egl_eglGetError(), surface);
        }
        // The application presented on its own, so the forced-swap fallback
        // has nothing left to do. Turned off permanently: if a real swap chain
        // is working, adding a second one would only cause tearing.
        mg_egl_note_app_swap();
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
        LOAD_EGL(eglSwapBuffersWithDamageEXT)
        if (!egl_eglSwapBuffersWithDamageEXT) {
            // Not available: fall back rather than drop the frame entirely.
            LOG_W_FORCE("eglSwapBuffersWithDamageEXT: unavailable, using eglSwapBuffers for surface %p", surface);
            return eglSwapBuffers(dpy, surface);
        }
        const EGLBoolean result = egl_eglSwapBuffersWithDamageEXT(dpy, surface, const_cast<EGLint*>(rects), n_rects);
        if (result != EGL_TRUE) {
            LOAD_EGL(eglGetError)
            LOG_W_FORCE("eglSwapBuffersWithDamageEXT: FAILED (0x%x) surface=%p", egl_eglGetError(), surface);
        }
        mg_egl_note_swap(dpy, surface, result);
        return result;
    }

    EGL_API EGLBoolean eglSwapBuffersWithDamageKHR(EGLDisplay dpy, EGLSurface surface, const EGLint* rects,
                                                   EGLint n_rects) {
        LOG_W_FORCE("EGL-TRACE: eglSwapBuffersWithDamageKHR called surface=%p n_rects=%d", surface, n_rects);
        mg_egl_note_call(__func__);
        LOG_D("eglSwapBuffersWithDamageKHR, dpy: %p, surface: %p, n_rects: %d", dpy, surface, n_rects);
        LOAD_EGL(eglSwapBuffersWithDamageKHR)
        if (!egl_eglSwapBuffersWithDamageKHR) {
            LOG_W_FORCE("eglSwapBuffersWithDamageKHR: unavailable, using eglSwapBuffers for surface %p", surface);
            return eglSwapBuffers(dpy, surface);
        }
        const EGLBoolean result = egl_eglSwapBuffersWithDamageKHR(dpy, surface, const_cast<EGLint*>(rects), n_rects);
        if (result != EGL_TRUE) {
            LOAD_EGL(eglGetError)
            LOG_W_FORCE("eglSwapBuffersWithDamageKHR: FAILED (0x%x) surface=%p", egl_eglGetError(), surface);
        }
        mg_egl_note_swap(dpy, surface, result);
        return result;
    }

    EGL_API EGLBoolean eglCopyBuffers(EGLDisplay dpy, EGLSurface surface, EGLNativePixmapType target) {
        mg_egl_note_call(__func__);
        LOG_D("eglCopyBuffers, dpy: %p, surface: %p, target: %p", dpy, surface, target);
        LOAD_EGL(eglCopyBuffers)
        return egl_eglCopyBuffers(dpy, surface, target);
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
                    LOG_W_FORCE("eglGetProcAddress(%s) -> this library\'s own entry point", procname);
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