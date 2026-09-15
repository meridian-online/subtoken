//! A global allocator that counts what the calling thread holds.
//!
//! Included by the measurement tests with `#[path]` rather than being a module
//! of the library: it must be the global allocator of the process doing the
//! measuring, and it must not be the global allocator of the shipped cdylib.
//!
//! It lives under `tests/support/` rather than directly in `tests/` because
//! cargo compiles every file directly in `tests/` into a test binary of its
//! own, and a binary with no tests in it is noise in the output.

// Included by more than one test binary, and not every one of them needs every
// helper: the clear measurement does not look at the peak. Dead-code warnings
// are per binary, so this is the shared-support-module idiom rather than a lint
// being waved away.
#![allow(dead_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ffi::c_void;

// The usable size the allocator really handed out, which is what makes the
// measurement catch size-class rounding. Both of these live in the C library
// that is linked either way, so neither is a new dependency.
#[cfg(target_os = "macos")]
extern "C" {
    fn malloc_size(ptr: *const c_void) -> usize;
}
#[cfg(target_os = "linux")]
extern "C" {
    fn malloc_usable_size(ptr: *mut c_void) -> usize;
}

/// Bytes the allocator actually reserved for `ptr`.
///
/// Falls back to the requested size on platforms with no way to ask, which
/// under-counts rather than over-counts — so a measurement there is a lower
/// bound and the ceiling assertion stays sound.
///
/// # Safety
/// `block` must be a live allocation from the global allocator.
unsafe fn actual_bytes(block: *mut u8, layout: Layout) -> usize {
    #[cfg(target_os = "macos")]
    {
        let _ = layout;
        malloc_size(block.cast::<c_void>())
    }
    #[cfg(target_os = "linux")]
    {
        let _ = layout;
        malloc_usable_size(block.cast::<c_void>())
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = block;
        layout.size()
    }
}

// Per-thread, so nothing else in the process can perturb a measurement. `const`
// initialiser, so reading it never allocates and the allocator cannot re-enter
// itself.
thread_local! {
    static LIVE: Cell<isize> = const { Cell::new(0) };
    static PEAK: Cell<isize> = const { Cell::new(0) };
}

fn record(delta: isize) {
    let _ = LIVE.try_with(|live| {
        let now = live.get() + delta;
        live.set(now);
        let _ = PEAK.try_with(|peak| {
            if now > peak.get() {
                peak.set(now);
            }
        });
    });
}

/// Bytes this thread currently holds. The absolute value is meaningless; the
/// difference between two readings is not.
pub fn live_bytes() -> isize {
    LIVE.with(Cell::get)
}

/// The high-water mark since [`reset_peak`].
pub fn peak_bytes() -> isize {
    PEAK.with(Cell::get)
}

pub fn reset_peak() {
    PEAK.with(|peak| peak.set(LIVE.with(Cell::get)));
}

pub struct Counting;

// SAFETY: every method forwards to `System` unchanged and only records the size
// alongside it.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            record(actual_bytes(ptr, layout) as isize);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        record(-(actual_bytes(ptr, layout) as isize));
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(-(actual_bytes(ptr, layout) as isize));
        let moved = System.realloc(ptr, layout, new_size);
        if moved.is_null() {
            // The old block is still live; put it back on the books.
            record(actual_bytes(ptr, layout) as isize);
        } else {
            let grown = Layout::from_size_align_unchecked(new_size, layout.align());
            record(actual_bytes(moved, grown) as isize);
        }
        moved
    }
}
