// MobileGlues - egl/loader.h
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

#ifndef FOLD_CRAFT_LAUNCHER_EGL_LOADER_H
#define FOLD_CRAFT_LAUNCHER_EGL_LOADER_H

// Needed unconditionally: AppRenderTarget below identifies the presenting
// thread, and it is declared inside the extern "C" block's translation unit in
// some consumers too.
#include <pthread.h>

#ifdef __cplusplus
extern "C"
{
#endif

#include "egl.h"

    typedef EGLBoolean (*eglBindAPI_PTR)(EGLenum api);

    typedef EGLBoolean (*eglBindTexImage_PTR)(EGLDisplay dpy, EGLSurface surface, EGLint buffer);

    typedef EGLBoolean (*eglChooseConfig_PTR)(EGLDisplay dpy, const EGLint* attrib_list, EGLConfig* configs,
                                              EGLint config_size, EGLint* num_config);

    typedef EGLBoolean (*eglCopyBuffers_PTR)(EGLDisplay dpy, EGLSurface surface, EGLNativePixmapType target);

    typedef EGLContext (*eglCreateContext_PTR)(EGLDisplay dpy, EGLConfig config, EGLContext share_context,
                                               const EGLint* attrib_list);

    typedef EGLSurface (*eglCreatePbufferFromClientBuffer_PTR)(EGLDisplay dpy, EGLenum buftype, EGLClientBuffer buffer,
                                                               EGLConfig config, const EGLint* attrib_list);

    typedef EGLSurface (*eglCreatePbufferSurface_PTR)(EGLDisplay dpy, EGLConfig config, const EGLint* attrib_list);

    typedef EGLSurface (*eglCreatePixmapSurface_PTR)(EGLDisplay dpy, EGLConfig config, EGLNativePixmapType pixmap,
                                                     const EGLint* attrib_list);

    typedef EGLSurface (*eglCreatePlatformWindowSurface_PTR)(EGLDisplay display, EGLConfig config, void* native_window,
                                                             const EGLint* attrib_list);

    typedef EGLSurface (*eglCreateWindowSurface_PTR)(EGLDisplay dpy, EGLConfig config, EGLNativeWindowType win,
                                                     const EGLint* attrib_list);

    typedef EGLBoolean (*eglDestroyContext_PTR)(EGLDisplay dpy, EGLContext ctx);

    typedef EGLBoolean (*eglDestroySurface_PTR)(EGLDisplay dpy, EGLSurface surface);

    typedef EGLBoolean (*eglGetConfigAttrib_PTR)(EGLDisplay dpy, EGLConfig config, EGLint attribute, EGLint* value);

    typedef EGLBoolean (*eglGetConfigs_PTR)(EGLDisplay dpy, EGLConfig* configs, EGLint config_size, EGLint* num_config);

    typedef EGLContext (*eglGetCurrentContext_PTR)();

    typedef EGLDisplay (*eglGetCurrentDisplay_PTR)();

    typedef EGLSurface (*eglGetCurrentSurface_PTR)(EGLint readdraw);

    typedef EGLDisplay (*eglGetDisplay_PTR)(EGLNativeDisplayType display_id);

    typedef EGLDisplay (*eglGetPlatformDisplay_PTR)(EGLenum platform, void* native_display, const EGLint* attrib_list);


    typedef EGLDisplay (*eglGetPlatformDisplayEXT_PTR)(EGLenum platform, void* native_display, const EGLint* attrib_list);

    typedef EGLint (*eglGetError_PTR)();

    typedef __eglMustCastToProperFunctionPointerType (*eglGetProcAddress_PTR)(const char* procname);

    typedef EGLBoolean (*eglInitialize_PTR)(EGLDisplay dpy, EGLint* major, EGLint* minor);

    typedef EGLBoolean (*eglMakeCurrent_PTR)(EGLDisplay dpy, EGLSurface draw, EGLSurface read, EGLContext ctx);

    typedef EGLenum (*eglQueryAPI_PTR)();

    typedef EGLBoolean (*eglQueryContext_PTR)(EGLDisplay dpy, EGLContext ctx, EGLint attribute, EGLint* value);

    typedef const char* (*eglQueryString_PTR)(EGLDisplay dpy, EGLint name);

    typedef EGLBoolean (*eglQuerySurface_PTR)(EGLDisplay dpy, EGLSurface surface, EGLint attribute, EGLint* value);

    typedef EGLBoolean (*eglReleaseTexImage_PTR)(EGLDisplay dpy, EGLSurface surface, EGLint buffer);

    typedef EGLBoolean (*eglReleaseThread_PTR)();

    typedef EGLBoolean (*eglSurfaceAttrib_PTR)(EGLDisplay dpy, EGLSurface surface, EGLint attribute, EGLint value);

    typedef EGLBoolean (*eglSwapBuffers_PTR)(EGLDisplay dpy, EGLSurface surface);

    typedef EGLBoolean (*eglSwapBuffersWithDamageEXT_PTR)(EGLDisplay dpy, EGLSurface surface, EGLint* rects,
                                                          EGLint n_rects);
    typedef EGLBoolean (*eglSwapBuffersWithDamageKHR_PTR)(EGLDisplay dpy, EGLSurface surface, EGLint* rects,
                                                          EGLint n_rects);

    typedef EGLBoolean (*eglSwapInterval_PTR)(EGLDisplay dpy, EGLint interval);

    typedef EGLBoolean (*eglTerminate_PTR)(EGLDisplay dpy);

    typedef EGLBoolean (*eglUnlockSurfaceKHR_PTR)(EGLDisplay display, EGLSurface surface);

    typedef EGLBoolean (*eglWaitClient_PTR)();

    typedef EGLBoolean (*eglWaitGL_PTR)();

    typedef EGLBoolean (*eglWaitNative_PTR)(EGLint engine);

    struct egl_func_t {
        eglBindAPI_PTR eglBindAPI;
        eglBindTexImage_PTR eglBindTexImage;
        eglChooseConfig_PTR eglChooseConfig;
        eglCopyBuffers_PTR eglCopyBuffers;
        eglCreateContext_PTR eglCreateContext;
        eglCreatePbufferFromClientBuffer_PTR eglCreatePbufferFromClientBuffer;
        eglCreatePbufferSurface_PTR eglCreatePbufferSurface;
        eglCreatePixmapSurface_PTR eglCreatePixmapSurface;
        eglCreatePlatformWindowSurface_PTR eglCreatePlatformWindowSurface;
        eglCreateWindowSurface_PTR eglCreateWindowSurface;
        eglDestroyContext_PTR eglDestroyContext;
        eglDestroySurface_PTR eglDestroySurface;
        eglGetConfigAttrib_PTR eglGetConfigAttrib;
        eglGetConfigs_PTR eglGetConfigs;
        eglGetCurrentContext_PTR eglGetCurrentContext;
        eglGetCurrentDisplay_PTR eglGetCurrentDisplay;
        eglGetCurrentSurface_PTR eglGetCurrentSurface;
        eglGetDisplay_PTR eglGetDisplay;
        eglGetPlatformDisplay_PTR eglGetPlatformDisplay;
        eglGetPlatformDisplayEXT_PTR eglGetPlatformDisplayEXT;
        eglGetError_PTR eglGetError;
        eglGetProcAddress_PTR eglGetProcAddress;
        eglInitialize_PTR eglInitialize;
        eglMakeCurrent_PTR eglMakeCurrent;
        eglQueryAPI_PTR eglQueryAPI;
        eglQueryContext_PTR eglQueryContext;
        eglQueryString_PTR eglQueryString;
        eglQuerySurface_PTR eglQuerySurface;
        eglReleaseTexImage_PTR eglReleaseTexImage;
        eglReleaseThread_PTR eglReleaseThread;
        eglSurfaceAttrib_PTR eglSurfaceAttrib;
        eglSwapBuffers_PTR eglSwapBuffers;
        eglSwapBuffersWithDamageEXT_PTR eglSwapBuffersWithDamageEXT;
        eglSwapBuffersWithDamageKHR_PTR eglSwapBuffersWithDamageKHR;
        eglSwapInterval_PTR eglSwapInterval;
        eglTerminate_PTR eglTerminate;
        eglUnlockSurfaceKHR_PTR eglUnlockSurfaceKHR;
        eglWaitClient_PTR eglWaitClient;
        eglWaitGL_PTR eglWaitGL;
        eglWaitNative_PTR eglWaitNative;
    };

    /*
    struct egl_func_t {
        EGLGETPROCADDRESSPROCP eglGetProcAddress;
        EGLCREATECONTEXTPROCP eglCreateContext;
        EGLDESTROYCONTEXTPROCP eglDestroyContext;
        EGLMAKECURRENTPROCP eglMakeCurrent;
        EGLQUERYSTRINGPROCP eglQueryString;
        EGLTERMINATEPROCP eglTerminate;
        EGLCHOOSECONFIGPROCP eglChooseConfig;
        EGLBINDAPIPROCP eglBindAPI;
        EGLINITIALIZEPROCP eglInitialize;
        EGLGETDISPLAYP eglGetDisplay;
        EGLCREATEPBUFFERSURFACEPROCP eglCreatePbufferSurface;
        EGLDESTROYSURFACEPROCP eglDestroySurface;
        EGLGETERRORPROCP eglGetError;
        EGLCREATEWINDOWSURFACEPROCP eglCreateWindowSurface;
    };
    EGLContext mglues_eglCreateContext(EGLDisplay dpy, EGLConfig config, EGLContext share_context, const EGLint
    *attrib_list); EGLBoolean mglues_eglDestroyContext(EGLDisplay dpy, EGLContext ctx); EGLBoolean
    mglues_eglMakeCurrent(EGLDisplay dpy, EGLSurface draw, EGLSurface read, EGLContext ctx);
    */

    void init_target_egl();
    void destroy_temp_egl_ctx();

    // -------------------------------------------------------------------------
    // Fallback context for host string queries
    //
    // The pbuffer context created by init_target_egl() used to be destroyed
    // right after initialization. It is now kept alive instead, because host
    // GLES string queries (glGetString) return NULL when the calling thread has
    // no current EGL context — which is legal per the ES 3.2 spec ("If an error
    // is generated, glGetString returns 0") and used to crash MobileGlues in
    // strlen(NULL) when a game queried GL_RENDERER from a thread that had not
    // called eglMakeCurrent yet.
    //
    // BindFallbackEGLContextIfNeeded() gives the calling thread a fallback
    // context of its own when it has no current context, and leaves it current
    // for the life of the thread. UnbindFallbackEGLContext() is kept for
    // callers that pair the two but intentionally does nothing — see
    // "Per-thread fallback contexts" in loader.cpp.
    // -------------------------------------------------------------------------
    bool BindFallbackEGLContextIfNeeded();
    void UnbindFallbackEGLContext();

#ifdef __cplusplus
}
#endif

// ==========================================================================
// ScopedHostContext — RAII wrapper around the fallback context
//
//   Binds MobileGLES' fallback pbuffer context for the duration of a host GL
//   call, but only when the calling thread has no current EGL context of its
//   own. Host drivers leave results untouched when there is no current
//   context; that is how a NULL glGetString, a zeroed glGetIntegerv, and a
//   NULL glMapBufferRange all reach the application.
//
//   Originally local to gl/getter.cpp (which only covers string and integer
//   queries). Promoted here because buffer operations need it too: a buffer
//   allocated and mapped while no context is current is silently dropped by
//   the driver, which Minecraft reports as "Failed to map buffer".
//
//   Nesting is safe — an inner instance sees a current context and does
//   nothing.
//
//   The context stays current after the scope ends; see "Fallback contexts,
//   without pbuffers" in loader.cpp for the three designs this replaced.
//   Practically, that means the guard costs one eglGetCurrentContext() after the
//   first call on a thread, and that GL entry points which were never wrapped
//   still see a current context.
// ==========================================================================
#ifdef __cplusplus

// Counts every guarded host GL call, across all threads, and attributes it to
// the entry point that made it. Read by the watchdog below: if the count stops
// growing, the game is stalled somewhere that is not making GL calls; if it
// keeps growing while nothing is presented, the watchdog's per-entry breakdown
// shows exactly which call is being repeated.
void mg_egl_note_guarded_call(const char* entry_point);

class ScopedHostContext {
public:
    ScopedHostContext(const char* entry_point) : bound_(BindFallbackEGLContextIfNeeded()) {
        mg_egl_note_guarded_call(entry_point);
    }
    ~ScopedHostContext() {
        if (bound_) UnbindFallbackEGLContext();
    }
    ScopedHostContext(const ScopedHostContext&) = delete;
    ScopedHostContext& operator=(const ScopedHostContext&) = delete;
    // True when this instance is the one that bound the context, i.e. the
    // thread had none and any result read before this point is suspect.
    bool Bound() const { return bound_; }

private:
    bool bound_;
};

// ---------------------------------------------------------------------------
// The application's real render target
//
// A fallback context is only useful if drawing on it is visible, and that
// requires the context and the surface the application actually presents from.
// Anything else — a pbuffer, a context of our own, a context bound with no
// surface — renders correctly into something nobody can see, while
// eglSwapBuffers presents the untouched window surface. That is a black screen
// with working audio: the game runs, the pipeline is alive, only the pixels go
// nowhere.
//
// So the fallback binds the application's own context to the application's own
// window surface, and these calls exist to report which those are. They are made
// from egl.cpp, which sees every EGL call the application makes, and read by
// BindFallbackEGLContextIfNeeded().
// ---------------------------------------------------------------------------

struct AppRenderTarget {
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLSurface draw_surface = EGL_NO_SURFACE;
    EGLSurface read_surface = EGL_NO_SURFACE;
    EGLContext context = EGL_NO_CONTEXT;
    EGLConfig config = nullptr;
    bool have_surface = false;  // a window surface was created
    bool have_binding = false;  // a context was successfully made current

    // The surface the application actually presents from, observed in
    // eglSwapBuffers. This is the only reliable way to identify it: on real
    // hardware it was NOT the surface reported by eglCreateWindowSurface.
    // A session logged one recorded window surface (0x77c45f3d80) and then
    // 1251 swaps against a different one (0x7806563d80), because the surface
    // that ends up on screen is created by a call this library does not
    // intercept — eglCreatePlatformWindowSurface is not exported here.
    //
    // Anything bound for drawing must be bound to this surface, or the pixels
    // go to a surface that is never presented.
    EGLSurface presenting_surface = EGL_NO_SURFACE;
    bool have_presenting = false;

    // The thread that calls eglSwapBuffers, i.e. the render thread. Recorded so
    // that only it binds to the presenting surface: a surface may be current for
    // one context at a time, and a worker thread taking it would block the very
    // thread the picture depends on.
    pthread_t presenting_thread = 0;
    // The thread the application bound its context on — normally the render
    // thread. Recorded so the watchdog's per-thread breakdown can be read
    // against it: whether the top caller IS the render thread decides whether
    // the game is spinning or merely waiting.
    pthread_t binding_thread = 0;
};

// Called from eglCreateWindowSurface: records the surface the application
// presents from. Tracks the latest, not the first — SDL creates more than one.
void mg_egl_note_window_surface(EGLDisplay dpy, EGLConfig config, EGLSurface surface);

// Called from eglMakeCurrent after the host returns. Tracks the application's
// live binding, so it always reflects what is current now rather than whatever
// happened to be bound first.
void mg_egl_note_make_current(EGLDisplay dpy, EGLSurface draw, EGLSurface read, EGLContext ctx, EGLBoolean ok);

// Called from eglDestroySurface. A destroyed surface must be forgotten, or the
// next attempt to bind it fails with EGL_BAD_SURFACE and drawing goes nowhere.
void mg_egl_note_destroy_surface(EGLDisplay dpy, EGLSurface surface);

// Forgets a surface after the driver rejected it as EGL_BAD_SURFACE, without
// waiting for eglDestroySurface. Self-healing: a surface can die through a path
// this library does not intercept, and using a dead one silently discards every
// draw call.
void mg_egl_forget_surface(EGLSurface surface);

// Counts an application-initiated call to one of this library's EGL entry
// points, into the same histogram as the guarded GL calls. The two are told
// apart by name — every EGL entry point starts with "egl".
//
// The point is to see what the application is actually doing at the EGL level.
// A session measured a render loop running at roughly 878 draws per second with
// eglSwapBuffers never called even once, which cannot be diagnosed from inside
// the GL layer: whether the application never asks to present, or asks through
// an entry point this library does not export, is invisible without this.
void mg_egl_note_call(const char* entry_point);

// Called from this library's eglSwapBuffers once the application presents on
// its own. That turns off the forced-swap fallback below, which exists only for
// the case where the application never presents at all.
void mg_egl_note_app_swap();

// Called from eglSwapBuffers, for diagnostics.
void mg_egl_note_swap(EGLDisplay dpy, EGLSurface surface, EGLBoolean ok);

const AppRenderTarget& mg_egl_app_target();

// Bumped every time the record above changes, so a thread holding an off-screen
// fallback can notice that a real window surface has appeared and switch to it.
// Read on every guarded GL call, so it is an atomic load and nothing more.
unsigned mg_egl_app_target_generation();

#endif // __cplusplus

#endif // FOLD_CRAFT_LAUNCHER_EGL_LOADER_H
