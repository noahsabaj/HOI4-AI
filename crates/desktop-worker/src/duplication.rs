//! Desktop Duplication capture, as an alternative to blitting the screen through GDI.
//!
//! The GDI path costs 87.1 ms p50 at 3840x2160 (artifacts/pairing/integration), measured
//! between the worker's own `capture_start_ns` and `t_ns`, so it is the blit and the
//! readback and nothing else. That is the largest single term in a 200 ms tick. Desktop
//! Duplication hands back a texture the compositor already owns, and the only copy left
//! is one GPU-to-staging DMA plus a mapped read.
//!
//! Three behaviours differ from BitBlt and all three are handled here rather than
//! discovered later:
//!
//! - `AcquireNextFrame` reports a timeout when nothing on screen has changed. That is not
//!   a failure and must not stall the tick: the previous desktop image is still what is
//!   on the screen, so it is kept (on the GPU) and read again. HOI4 at speed does change
//!   every frame, but a paused game or a menu genuinely does not.
//! - The duplication is lost on a resolution change, a full-screen transition, a driver
//!   reset or a session switch, reported as `DXGI_ERROR_ACCESS_LOST`. Recovery is to
//!   rebuild the whole chain, and the caller is expected to fall back to GDI if that
//!   fails, because a match must not end over a display mode change.
//! - It duplicates an *output*, not a window. So does the GDI path already -- it blits
//!   the desktop DC at the game's client rectangle rather than the window's own DC -- so
//!   the cropping is equivalent. The foreground check in the caller is what keeps this
//!   honest, and it is not optional: an output capture will happily return whatever is
//!   actually in front.
//!
//! The cursor is the one thing that cannot be settled from the documentation. BitBlt
//! never draws it. Desktop Duplication separates the pointer into `PointerPosition` and a
//! shape the caller composites, so the desktop image should match -- but Microsoft
//! documents that the image may contain the pointer when it is drawn in software rather
//! than by the hardware plane. The policy sees whatever is in these pixels, so a
//! disagreement would be a train-deploy mismatch rather than a visual nit. The caller
//! compares the two backends on the same screen before trusting this one.

#![cfg(windows)]

use windows::core::Interface;
use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::{D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP};
use windows::Win32::Graphics::Direct3D11::{
    D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D, D3D11_BOX,
    D3D11_CPU_ACCESS_READ, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_MAP_READ, D3D11_SDK_VERSION,
    D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT, D3D11_USAGE_STAGING,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::{
    IDXGIDevice, IDXGIOutput1, IDXGIOutputDuplication, IDXGIResource, DXGI_ERROR_ACCESS_LOST,
    DXGI_ERROR_WAIT_TIMEOUT,
};

/// How long to wait for a new desktop frame before deciding the screen did not change.
///
/// None. The duplication hands over the newest complete desktop image presented since the
/// last acquire, so a capture every 200 ms (a game draws dozens of frames in between) always
/// gets the latest one at once, and a timeout means nothing was presented: the image kept
/// from before is still the screen. Waiting 8 ms here, as before, waited for the next present
/// on most calls and made a 1920x1080 capture take 14.9 ms p50 instead of 4.9 (this PC,
/// 2026-09-24).
const FRAME_TIMEOUT_MS: u32 = 0;

/// How long to keep asking before the very first frame, which has nothing cached behind
/// it. A freshly created duplication has no desktop image until something is presented,
/// and on a still screen that can be a while -- CI found this by duplicating an idle
/// runner desktop and getting nothing at all. Spread across several waits rather than
/// one long one so a screen that starts moving is picked up immediately.
const FIRST_FRAME_ATTEMPTS: u32 = 8;
const FIRST_FRAME_TIMEOUT_MS: u32 = 25;

/// Why a frame could not be produced. The distinction is not cosmetic: a duplication
/// that has simply not seen a present yet is healthy and should be kept, while a lost
/// one has to be rebuilt. Collapsing the two retires the fast path on a still screen,
/// which is the screen it handles best.
#[derive(Debug)]
pub enum Unavailable {
    /// Nothing has been presented yet, so there is no desktop image to hand back.
    NotReadyYet,
    /// The duplication is gone, or failed in a way that needs a new one.
    Lost(String),
}

impl std::fmt::Display for Unavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotReadyYet => write!(f, "no frame presented yet"),
            Self::Lost(reason) => write!(f, "{reason}"),
        }
    }
}

/// Whether this failure should spend one of the rebuilds that retire the fast path.
///
/// A window that has moved off the duplicated output is a geometry miss. Rebuilding
/// the duplicator for the new output is cheap. Counting it as a lost device retires
/// Desktop Duplication after three moves and leaves the rest of the match on the blit.
pub fn should_retire(reason: &Unavailable) -> bool {
    match reason {
        Unavailable::NotReadyYet => false,
        Unavailable::Lost(text) if text.contains("client_rect_outside") => false,
        Unavailable::Lost(_) => true,
    }
}

pub struct Duplicator {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    duplication: IDXGIOutputDuplication,
    /// The last desktop image, kept on the GPU because a timeout means "unchanged", not
    /// "no data". Only the game's rectangle of it is ever read back.
    desktop: ID3D11Texture2D,
    /// A CPU-readable texture the size of the game's client rectangle, and that size.
    staging: Option<(ID3D11Texture2D, usize, usize)>,
    /// The duplicated output's position on the virtual desktop, which client coordinates
    /// from `ClientToScreen` are relative to and this buffer is not.
    origin: (i32, i32),
    width: usize,
    height: usize,
    holding: bool,
    ready: bool,
}

impl Duplicator {
    /// Duplicate whichever output contains `point`, in virtual-desktop coordinates.
    pub fn new(point: (i32, i32)) -> Result<Self, String> {
        unsafe {
            let mut device = None;
            let mut context = None;
            // WARP is accepted as a fallback so a machine without a usable hardware
            // adapter reports a slow duplication rather than none, but it is not the
            // expected path and the caller can still choose GDI.
            let mut created = Err(windows::core::Error::empty());
            for driver in [D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP] {
                created = D3D11CreateDevice(
                    None,
                    driver,
                    HMODULE::default(),
                    D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                    None,
                    D3D11_SDK_VERSION,
                    Some(&mut device),
                    None,
                    Some(&mut context),
                );
                if created.is_ok() {
                    break;
                }
            }
            created.map_err(|e| format!("d3d11_create_device_failed_{:x}", e.code().0))?;
            let device = device.ok_or("d3d11_device_missing")?;
            let context = context.ok_or("d3d11_context_missing")?;

            let dxgi: IDXGIDevice = device.cast().map_err(|_| "dxgi_device_cast_failed")?;
            let adapter = dxgi.GetAdapter().map_err(|_| "dxgi_adapter_failed")?;
            let mut chosen = None;
            for index in 0.. {
                let Ok(output) = adapter.EnumOutputs(index) else {
                    break;
                };
                let Ok(desc) = output.GetDesc() else {
                    continue;
                };
                let r = desc.DesktopCoordinates;
                if (r.left..r.right).contains(&point.0) && (r.top..r.bottom).contains(&point.1) {
                    chosen = Some((output, r));
                    break;
                }
            }
            let (output, rect) = chosen.ok_or("no_output_contains_the_game_window")?;
            let output: IDXGIOutput1 = output.cast().map_err(|_| "dxgi_output1_unavailable")?;
            let duplication = output
                .DuplicateOutput(&device)
                .map_err(|e| format!("duplicate_output_failed_{:x}", e.code().0))?;

            let (width, height) = (
                (rect.right - rect.left) as usize,
                (rect.bottom - rect.top) as usize,
            );
            let desktop = texture(&device, width, height, false)
                .map_err(|e| format!("desktop_texture_failed_{e}"))?;
            Ok(Self {
                device,
                context,
                duplication,
                desktop,
                staging: None,
                origin: (rect.left, rect.top),
                width,
                height,
                holding: false,
                ready: false,
            })
        }
    }

    /// Refresh the cached desktop image, if the compositor has presented a new one.
    fn refresh(&mut self) -> Result<(), Unavailable> {
        unsafe {
            self.release();
            let mut info = Default::default();
            let mut resource: Option<IDXGIResource> = None;
            // With an image already cached a timeout is the answer, not a wait: the
            // screen did not change. Without one there is nothing to return, so the
            // first frame is worth waiting for.
            let (attempts, timeout) = if self.ready {
                (1, FRAME_TIMEOUT_MS)
            } else {
                (FIRST_FRAME_ATTEMPTS, FIRST_FRAME_TIMEOUT_MS)
            };
            let mut acquired = false;
            for _ in 0..attempts {
                match self
                    .duplication
                    .AcquireNextFrame(timeout, &mut info, &mut resource)
                {
                    Ok(()) => {
                        acquired = true;
                        break;
                    }
                    Err(error) if error.code() == DXGI_ERROR_WAIT_TIMEOUT => continue,
                    Err(error) if error.code() == DXGI_ERROR_ACCESS_LOST => {
                        return Err(Unavailable::Lost("duplication_access_lost".into()));
                    }
                    Err(error) => {
                        return Err(Unavailable::Lost(format!(
                            "acquire_frame_failed_{:x}",
                            error.code().0
                        )))
                    }
                }
            }
            if !acquired {
                // Nothing presented. With an image cached that is still the screen;
                // without one the caller has to get its pixels elsewhere this tick.
                return if self.ready {
                    Ok(())
                } else {
                    Err(Unavailable::NotReadyYet)
                };
            }
            self.holding = true;
            let resource = resource
                .ok_or_else(|| Unavailable::Lost("acquire_frame_returned_no_surface".into()))?;
            let texture: ID3D11Texture2D = resource
                .cast()
                .map_err(|_| Unavailable::Lost("frame_is_not_a_texture".into()))?;
            // A copy on the GPU, kept for the ticks where nothing new is presented. The
            // frame is held until `client` has read its rectangle back: released any
            // sooner, the compositor may draw the next frame into it before the queued
            // copies run.
            self.context.CopyResource(&self.desktop, &texture);
            self.ready = true;
            Ok(())
        }
    }

    /// Hand the frame back before asking for another; the API refuses two at once.
    fn release(&mut self) {
        if self.holding {
            unsafe {
                let _ = self.duplication.ReleaseFrame();
            }
            self.holding = false;
        }
    }

    /// One window's client rectangle, packed BGRA, from the current desktop image.
    ///
    /// `screen` is the client origin in virtual-desktop coordinates -- exactly what the
    /// GDI path hands to BitBlt -- so both backends return the identical rectangle and
    /// can be compared pixel for pixel.
    pub fn client(
        &mut self,
        screen: (i32, i32),
        w: usize,
        h: usize,
    ) -> Result<Vec<u8>, Unavailable> {
        let x = screen.0 - self.origin.0;
        let y = screen.1 - self.origin.1;
        if x < 0 || y < 0 || x as usize + w > self.width || y as usize + h > self.height {
            return Err(Unavailable::Lost(
                "client_rect_outside_duplicated_output".into(),
            ));
        }
        self.refresh()?;
        if w == 0 || h == 0 {
            return Ok(Vec::new());
        }
        let (x, y) = (x as usize, y as usize);
        // Only the game's rectangle crosses to the CPU: at 1920x1080 on a 3840x2160
        // desktop, a quarter of what copying the whole output read back.
        let staging = match &self.staging {
            Some((staging, sw, sh)) if (*sw, *sh) == (w, h) => staging.clone(),
            _ => {
                let staging = texture(&self.device, w, h, true)
                    .map_err(|e| Unavailable::Lost(format!("staging_texture_failed_{e}")))?;
                self.staging = Some((staging.clone(), w, h));
                staging
            }
        };
        let region = D3D11_BOX {
            left: x as u32,
            top: y as u32,
            front: 0,
            right: (x + w) as u32,
            bottom: (y + h) as u32,
            back: 1,
        };
        let mut out = vec![0u8; w * h * 4];
        unsafe {
            self.context.CopySubresourceRegion(
                &staging,
                0,
                0,
                0,
                0,
                &self.desktop,
                0,
                Some(&region),
            );
            let mut mapped = Default::default();
            // Map waits for the queued copies; only then is the frame given back.
            let map = self
                .context
                .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped));
            if let Err(e) = map {
                self.release();
                return Err(Unavailable::Lost(format!(
                    "map_staging_failed_{:x}",
                    e.code().0
                )));
            }
            // The staging row pitch is the driver's, not w * 4, so the rows are copied one
            // at a time into a packed buffer the rest of the worker can index.
            let pitch = mapped.RowPitch as usize;
            let row = w * 4;
            for line in 0..h {
                let source = (mapped.pData as *const u8).add(line * pitch);
                std::ptr::copy_nonoverlapping(source, out.as_mut_ptr().add(line * row), row);
            }
            self.context.Unmap(&staging, 0);
        }
        self.release();
        Ok(out)
    }
}

/// A BGRA texture of `width` x `height`: CPU-readable staging when `readable`, else a
/// GPU-only copy target.
fn texture(
    device: &ID3D11Device,
    width: usize,
    height: usize,
    readable: bool,
) -> Result<ID3D11Texture2D, String> {
    let descriptor = D3D11_TEXTURE2D_DESC {
        Width: width as u32,
        Height: height as u32,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: if readable {
            D3D11_USAGE_STAGING
        } else {
            D3D11_USAGE_DEFAULT
        },
        BindFlags: 0,
        CPUAccessFlags: if readable {
            D3D11_CPU_ACCESS_READ.0 as u32
        } else {
            0
        },
        MiscFlags: 0,
    };
    let mut created = None;
    unsafe {
        device
            .CreateTexture2D(&descriptor, None, Some(&mut created))
            .map_err(|e| format!("{:x}", e.code().0))?;
    }
    created.ok_or_else(|| "missing".to_string())
}

impl Drop for Duplicator {
    fn drop(&mut self) {
        self.release();
        // Named so the fields are visibly owned for the lifetime of the duplication;
        // dropping the device before the duplication would be a use-after-free.
        let _ = &self.device;
        let _ = &self.context;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Windows permits one duplication of an output per process, and returns
    /// E_INVALIDARG for a second. Tests run on threads by default, so without this they
    /// race: one gets the duplication and the others quietly take their
    /// nothing-to-test branch and pass having checked nothing.
    static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn exclusive() -> std::sync::MutexGuard<'static, ()> {
        ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
    }

    use windows_sys::Win32::Graphics::Gdi::{
        BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject, GetDC,
        GetDIBits, ReleaseDC, SelectObject, BITMAPINFO, BITMAPINFOHEADER, BI_RGB, CAPTUREBLT,
        DIB_RGB_COLORS, SRCCOPY,
    };
    use windows_sys::Win32::UI::HiDpi::{
        SetProcessDpiAwarenessContext, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
    };

    /// The same rectangle, through the path this one is replacing.
    fn blit(x: i32, y: i32, w: i32, h: i32) -> Option<Vec<u8>> {
        unsafe {
            let dc = GetDC(std::ptr::null_mut());
            let mem = CreateCompatibleDC(dc);
            let bitmap = CreateCompatibleBitmap(dc, w, h);
            let old = SelectObject(mem, bitmap);
            let ok = BitBlt(mem, 0, 0, w, h, dc, x, y, SRCCOPY | CAPTUREBLT);
            SelectObject(mem, old);
            let mut info: BITMAPINFO = std::mem::zeroed();
            info.bmiHeader.biSize = std::mem::size_of::<BITMAPINFOHEADER>() as u32;
            info.bmiHeader.biWidth = w;
            info.bmiHeader.biHeight = -h;
            info.bmiHeader.biPlanes = 1;
            info.bmiHeader.biBitCount = 32;
            info.bmiHeader.biCompression = BI_RGB;
            let mut bytes = vec![0u8; (w * h * 4) as usize];
            let lines = if ok != 0 {
                GetDIBits(
                    dc,
                    bitmap,
                    0,
                    h as u32,
                    bytes.as_mut_ptr() as *mut _,
                    &mut info,
                    DIB_RGB_COLORS,
                )
            } else {
                0
            };
            DeleteObject(bitmap);
            DeleteDC(mem);
            ReleaseDC(std::ptr::null_mut(), dc);
            (lines == h).then_some(bytes)
        }
    }

    /// A duplication either builds and yields a frame, or explains why it could not.
    ///
    /// Written to tolerate a machine with no usable duplication -- a headless CI runner,
    /// a remote session, an adapter that refuses -- because the worker tolerates that
    /// too and falls back to the blit. What it does not tolerate is a duplication that
    /// reports success and then produces nothing.
    #[test]
    fn a_duplication_either_works_or_says_why() {
        let _exclusive = exclusive();
        match Duplicator::new((0, 0)) {
            Err(reason) => {
                assert!(!reason.is_empty(), "a failure must name itself");
                eprintln!("no duplication on this machine: {reason}");
            }
            // A duplication that has not seen a present yet is healthy, not broken --
            // CI duplicated an idle runner desktop and got exactly this. It is a
            // distinct outcome from a lost duplication precisely so the worker can keep
            // one and fall through to the blit for that tick.
            Ok(mut duplicator) => match duplicator.client((0, 0), 64, 64) {
                Ok(frame) => {
                    assert_eq!(frame.len(), 64 * 64 * 4);
                    // A second call exercises the release-before-acquire path, which
                    // the API refuses if the previous frame is still held.
                    assert!(duplicator.client((0, 0), 64, 64).is_ok());
                }
                Err(Unavailable::NotReadyYet) => {
                    eprintln!("nothing presented on this desktop; the blit covers the tick")
                }
                Err(Unavailable::Lost(reason)) => panic!("duplication failed: {reason}"),
            },
        }
    }

    #[test]
    fn a_window_leaving_the_output_does_not_retire_duplication() {
        assert!(!should_retire(&Unavailable::Lost(
            "client_rect_outside_duplicated_output".into()
        )));
        assert!(!should_retire(&Unavailable::NotReadyYet));
        assert!(should_retire(&Unavailable::Lost(
            "duplication_access_lost".into()
        )));
    }

    /// Cropping outside the duplicated output is refused rather than read out of bounds.
    #[test]
    fn a_rectangle_outside_the_output_is_refused() {
        let _exclusive = exclusive();
        if let Ok(mut duplicator) = Duplicator::new((0, 0)) {
            // Out of bounds is a Lost, not a NotReadyYet: it is a caller error and must
            // not be mistaken for a screen that has not moved.
            for outside in [
                duplicator.client((-1, 0), 8, 8),
                duplicator.client((0, 0), 1 << 20, 8),
            ] {
                assert!(matches!(outside, Err(Unavailable::Lost(_))));
            }
        }
    }

    /// What the swap is worth, at the resolution a match actually runs.
    ///
    /// Ignored for the same reason as the parity test: it reads the live desktop and
    /// depends on the machine. Run it deliberately:
    /// `cargo test --release -- --ignored duplication_is_faster`. It asserts only that
    /// duplication is not slower, because the size of the win is a property of the
    /// hardware and the number belongs in a run's evidence rather than in a threshold.
    #[test]
    #[ignore = "reads the live desktop; run deliberately"]
    fn duplication_is_faster_than_the_blit() {
        let _exclusive = exclusive();
        unsafe {
            SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        }
        let mut duplicator = match Duplicator::new((0, 0)) {
            Ok(d) => d,
            Err(reason) => {
                eprintln!("skipped: {reason}");
                return;
            }
        };
        let (w, h) = (duplicator.width as i32, duplicator.height as i32);
        let median = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let mut duplicated = Vec::new();
        let mut blitted = Vec::new();
        for _ in 0..20 {
            let start = std::time::Instant::now();
            duplicator.client((0, 0), w as usize, h as usize).unwrap();
            duplicated.push(start.elapsed().as_secs_f64() * 1000.0);
            let start = std::time::Instant::now();
            blit(0, 0, w, h).unwrap();
            blitted.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        let (fast, slow) = (median(duplicated), median(blitted));
        eprintln!(
            "{w}x{h}: duplication {fast:.1} ms p50, gdi blit {slow:.1} ms p50, {:.1}x",
            slow / fast
        );
        // What a recording reads: a 1920x1080 game window, only its rectangle read back.
        let (cw, ch) = (1920.min(w as usize), 1080.min(h as usize));
        let mut window = Vec::new();
        for _ in 0..20 {
            let start = std::time::Instant::now();
            duplicator.client((0, 0), cw, ch).unwrap();
            window.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        eprintln!("{cw}x{ch} window: duplication {:.1} ms p50", median(window));
        assert!(
            fast < slow,
            "duplication ({fast:.1} ms) is not faster than the blit ({slow:.1} ms)"
        );
    }

    /// Duplication and the blit must return the same pixels, not merely similar ones.
    ///
    /// Ignored by default because it reads the live desktop: anything that repaints
    /// between the two captures is a genuine difference and would make this fail for the
    /// right reason at the wrong time. Run it deliberately, on a still screen:
    /// `cargo test --release -- --ignored duplication_and_the_blit`.
    ///
    /// The comparison is against two blits taken either side of the duplication, and the
    /// duplication has to match one of them. That tolerates a single repaint while still
    /// pinning what this module could plausibly get wrong -- the row pitch, the channel
    /// order, and the offset between output coordinates and desktop coordinates.
    #[test]
    #[ignore = "reads the live desktop; run on a still screen"]
    fn duplication_and_the_blit_agree_pixel_for_pixel() {
        let _exclusive = exclusive();
        // The worker does this at startup and the test binary does not, so without it
        // the blit is DPI-virtualized while the duplication is in physical pixels and
        // every byte differs -- which is how this test first failed.
        unsafe {
            SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        }
        let mut duplicator = match Duplicator::new((0, 0)) {
            Ok(d) => d,
            Err(reason) => {
                eprintln!("skipped: {reason}");
                return;
            }
        };
        let (x, y, w, h) = (64i32, 64i32, 512i32, 512i32);
        duplicator.client((x, y), w as usize, h as usize).unwrap();
        let before = blit(x, y, w, h).expect("gdi blit failed");
        let duplicated = duplicator.client((x, y), w as usize, h as usize).unwrap();
        let after = blit(x, y, w, h).expect("gdi blit failed");
        let differing = duplicated
            .iter()
            .zip(&after)
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            duplicated == before || duplicated == after,
            "duplication differs from both blits ({differing} of {} bytes differ from the later \
             one); if the screen was still, the pitch, channel order or origin is wrong",
            duplicated.len()
        );
    }
}
