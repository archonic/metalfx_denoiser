//! Sending and receiving `simd_float4x4` through `objc_msgSend`.
//!
//! The objc2 bindings don't expose `worldToViewMatrix` / `viewToClipMatrix`, so we send those
//! messages manually. In C, `simd_float4x4` is `struct { simd_float4 columns[4]; }`. On arm64 that
//! is a homogeneous vector aggregate, passed and returned in `v0`–`v3`.
//!
//! Rust can't express that ABI:
//! - rustc passes a `#[repr(C)]` struct of four `float32x4_t` through memory, not as a vector
//!   aggregate. Checked against clang: arguments and return values both come out as garbage.
//! - Passing SIMD types directly in `extern "C"` signatures needs the unstable `simd_ffi` feature.
//!
//! So these calls are inline asm that puts the columns in `v0`–`v3` and calls `objc_msgSend`.
//! Before any call, [`has_matrix_methods`] checks the runtime's type encoding of the selectors.

use core::arch::aarch64::float32x4_t;
use core::ffi::CStr;

use objc2::runtime::{AnyObject, Sel};

/// Column-major 4×4 matrix.
pub(crate) type Mat4 = [[f32; 4]; 4];

/// The runtime's encoding of `simd_float4x4`: an anonymous struct holding a 4-element array whose
/// element type (a SIMD vector) has no ObjC encoding.
const MATRIX_ENCODING: &CStr = c"{?=[4]}";

/// Whether `obj` implements the setter/getter pairs with the `simd_float4x4` signature this
/// module assumes.
pub(crate) fn has_matrix_methods(obj: &AnyObject, pairs: &[(Sel, Sel)]) -> bool {
    let class = obj.class();
    pairs.iter().all(|&(setter, getter)| {
        let setter_ok = class.instance_method(setter).is_some_and(|m| {
            m.arguments_count() == 3
                && *m.return_type() == *c"v"
                && m.argument_type(2).is_some_and(|t| *t == *MATRIX_ENCODING)
        });
        let getter_ok = class
            .instance_method(getter)
            .is_some_and(|m| m.arguments_count() == 2 && *m.return_type() == *MATRIX_ENCODING);
        setter_ok && getter_ok
    })
}

fn sel_ptr(sel: Sel) -> *const core::ffi::c_void {
    // SAFETY: `Sel` is `#[repr(transparent)]` over the runtime's `SEL` pointer.
    unsafe { core::mem::transmute::<Sel, *const core::ffi::c_void>(sel) }
}

/// Sends `[obj setter:m]` for a `simd_float4x4` argument.
///
/// # Safety
/// `obj` must be a valid object that implements `setter` with the `v@:{simd_float4x4}`
/// signature (see [`has_matrix_methods`]).
pub(crate) unsafe fn send_matrix(obj: &AnyObject, setter: Sel, m: Mat4) {
    // SAFETY: same size, and every bit pattern is valid for both types.
    let c: [float32x4_t; 4] = unsafe { core::mem::transmute(m) };
    // SAFETY: AAPCS64 call of `objc_msgSend(obj, setter, m)`: self in x0, _cmd in x1, and the
    // vector aggregate in v0–v3. `clobber_abi("C")` marks every caller-saved register as clobbered.
    unsafe {
        core::arch::asm!(
            "blr {f}",
            f = in(reg) objc2::ffi::objc_msgSend as *const () as usize,
            in("x0") obj as *const AnyObject,
            in("x1") sel_ptr(setter),
            in("v0") c[0], in("v1") c[1], in("v2") c[2], in("v3") c[3],
            clobber_abi("C"),
        );
    }
}

/// Sends `[obj getter]` for a `simd_float4x4` return value.
///
/// # Safety
/// `obj` must be a valid object that implements `getter` with the `{simd_float4x4}@:`
/// signature (see [`has_matrix_methods`]).
pub(crate) unsafe fn receive_matrix(obj: &AnyObject, getter: Sel) -> Mat4 {
    let (c0, c1, c2, c3): (float32x4_t, float32x4_t, float32x4_t, float32x4_t);
    // SAFETY: AAPCS64 call of `objc_msgSend(obj, getter)`. A vector aggregate is returned in
    // v0–v3.
    unsafe {
        core::arch::asm!(
            "blr {f}",
            f = in(reg) objc2::ffi::objc_msgSend as *const () as usize,
            in("x0") obj as *const AnyObject,
            in("x1") sel_ptr(getter),
            lateout("v0") c0, lateout("v1") c1, lateout("v2") c2, lateout("v3") c3,
            clobber_abi("C"),
        );
    }
    // SAFETY: same size, and every bit pattern is valid for both types.
    unsafe { core::mem::transmute([c0, c1, c2, c3]) }
}
