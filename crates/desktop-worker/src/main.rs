use serde::{Deserialize, Serialize};

#[cfg(windows)]
mod duplication;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    Move { x: f64, y: f64 },
    Button { button: u8, down: bool },
    Key { vk: u16, down: bool },
    Wheel { delta: i32 },
}

/// Area-average one BGRA box down to `size` x `size` RGB, matching torch's `area` mode.
///
/// Output pixel k covers source rows [floor(k*H/size), ceil((k+1)*H/size)) and the
/// matching column span. This is not an approximation of the training resize, it is the
/// same rule: `hoi4_arena.dataset.views` interpolates with mode="area", which is integer
/// binned adaptive average pooling. A box filter over fractional boundaries, or a
/// bilinear filter, would differ by a mean of roughly 11/255 and the policy would see
/// different pixels at deployment than it trained on. Accumulation is f32 and rounding is
/// ties-to-even, because that is what torch does.
///
/// The loop is spread across a few threads because it is the single largest piece of
/// CPU the worker spends per tick -- 17.9 ms for the five views at 3840x2160 -- and
/// every output pixel is an independent reduction over its own source box, so splitting
/// the output rows cannot change a result. It does not: a test asserts the threaded
/// output is byte-identical to the serial one on all five view boxes.
///
/// Doing this on the GPU instead was the obvious idea and is the wrong one. Reaching a
/// GPU means sending the whole 33 MB frame instead of 735 KiB of views, which costs
/// 2.9 ms to compress and 8.3 ms to decompress before any transport at all, and the
/// second machine receives this same protocol over a LAN socket where 33 MB five times
/// a second is not slower but impossible.
pub fn downscale_bgra(src: &[u8], width: usize, box_: [usize; 4], size: usize) -> Vec<u8> {
    let mut out = vec![0u8; size * size * 3];
    let threads = downscale_threads();
    if threads <= 1 || size < threads {
        for (oy, row) in out.chunks_exact_mut(size * 3).enumerate() {
            downscale_row(src, width, box_, size, oy, row);
        }
        return out;
    }
    let stripe = size.div_ceil(threads);
    std::thread::scope(|scope| {
        for (index, chunk) in out.chunks_mut(stripe * size * 3).enumerate() {
            let base = index * stripe;
            scope.spawn(move || {
                for (offset, row) in chunk.chunks_exact_mut(size * 3).enumerate() {
                    downscale_row(src, width, box_, size, base + offset, row);
                }
            });
        }
    });
    out
}

/// How many threads the downscale spreads across.
///
/// Measured at 3840x2160 for all five views: one thread 25.3 ms, two 12.8, four 8.3,
/// twenty-eight 8.9. It stops scaling at four because the loop reads the whole frame
/// rather than computing on it, so past that the threads only contend for the same
/// memory. The cap is also deliberate for a second reason -- the game is running on
/// this machine, and a worker that takes every core is its own kind of dropped frame.
fn downscale_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| (n.get() / 2).clamp(1, 4))
        .unwrap_or(1)
}

/// One output row of the reduction above, written into `out`.
///
/// The source span is narrowed to a slice before the inner loop rather than indexed
/// pixel by pixel out of the flat frame, which lets the bounds check happen once per row
/// instead of three times per source pixel: 1.26x on its own, before any threading.
fn downscale_row(
    src: &[u8],
    width: usize,
    box_: [usize; 4],
    size: usize,
    oy: usize,
    out: &mut [u8],
) {
    let [top, left, bh, bw] = box_;
    let y0 = oy * bh / size;
    let y1 = ((oy + 1) * bh).div_ceil(size);
    for ox in 0..size {
        let x0 = ox * bw / size;
        let x1 = ((ox + 1) * bw).div_ceil(size);
        let (mut sr, mut sg, mut sb) = (0u32, 0u32, 0u32);
        for y in y0..y1 {
            let start = ((top + y) * width + left + x0) * 4;
            for pixel in src[start..start + (x1 - x0) * 4].as_chunks::<4>().0 {
                sb += pixel[0] as u32;
                sg += pixel[1] as u32;
                sr += pixel[2] as u32;
            }
        }
        let n = ((y1 - y0) * (x1 - x0)) as f32;
        let o = ox * 3;
        out[o] = (sr as f32 / n).round_ties_even().clamp(0., 255.) as u8;
        out[o + 1] = (sg as f32 / n).round_ties_even().clamp(0., 255.) as u8;
        out[o + 2] = (sb as f32 / n).round_ties_even().clamp(0., 255.) as u8;
    }
}

/// The global frame plus the four spatially ordered quadrants, in the order the policy
/// expects. Mirrors `hoi4_arena.dataset.quadrants`. The cursor crop is not one of these
/// boxes: it is a native copy, appended after them by `cursor_crop_bgra`.
pub fn view_boxes(width: usize, height: usize) -> [[usize; 4]; 5] {
    let (hh, hw) = (height / 2, width / 2);
    [
        [0, 0, height, width],
        [0, 0, hh, hw],
        [0, hw, hh, width - hw],
        [hh, 0, height - hh, hw],
        [hh, hw, height - hh, width - hw],
    ]
}

/// Native `size` square of RGB centered on the client-pixel pointer.
///
/// The pointer lands on output pixel `(size / 2, size / 2)`. Samples outside the frame
/// are zero, so the pointer stays on that pixel at a screen edge instead of the window
/// sliding. This is a copy, not an average: an in-bounds window is what `downscale_bgra`
/// returns when the box is already `size` on a side. It must match
/// `hoi4_arena.dataset.cursor_crop`, which sees RGB that has already been swizzled.
pub fn cursor_crop_bgra(
    src: &[u8],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    size: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; size * size * 3];
    if size == 0 || width == 0 || height == 0 {
        return out;
    }
    let origin_x = x as i64 - (size / 2) as i64;
    let origin_y = y as i64 - (size / 2) as i64;
    for oy in 0..size {
        let sy = origin_y + oy as i64;
        if sy < 0 || sy >= height as i64 {
            continue;
        }
        for ox in 0..size {
            let sx = origin_x + ox as i64;
            if sx < 0 || sx >= width as i64 {
                continue;
            }
            let i = (sy as usize * width + sx as usize) * 4;
            let o = (oy * size + ox) * 3;
            out[o] = src[i + 2];
            out[o + 1] = src[i + 1];
            out[o + 2] = src[i];
        }
    }
    out
}

/// The arena mod's lines among complete lines of game.log text, without the engine's
/// timestamp prefix, and how many bytes were consumed. A trailing partial line is left
/// for the next read.
fn arena_lines(bytes: &[u8]) -> (Vec<String>, usize) {
    let used = bytes.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
    let lines = String::from_utf8_lossy(&bytes[..used])
        .lines()
        .filter_map(|line| {
            line.split_once("]: ARENA ")
                .map(|(_, rest)| rest.trim().to_string())
        })
        .collect();
    (lines, used)
}

fn valid_event(e: &Event, setup: bool) -> bool {
    match *e {
        Event::Move { x, y } => {
            x.is_finite() && y.is_finite() && (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y)
        }
        Event::Button { button, .. } => button < 3,
        Event::Wheel { delta } => delta.unsigned_abs() <= 1200,
        Event::Key { vk, .. } => {
            // Match input is the demonstration vocabulary: modifiers, arrows, letters,
            // tab, enter, and digits. OS keys, Alt, the console, F12, and the speed keys
            // stay refused, and so do space and escape: space pauses and escape opens the
            // pause menu, which the match loop rejects as game_paused. Speed is
            // operator-declared; a match that can change it falsifies the manifest.
            // Setup may also open the console (grave, 0xc0), to type `observe` on a PC
            // nobody is sitting at.
            matches!(
                vk,
                0x09 | 0x0d | 0x10 | 0x11 | 0x25..=0x28 | 0x30..=0x39 | 0x41..=0x5a
            ) || (setup && matches!(vk, 0x08 | 0x1b | 0x20 | 0xbb | 0xbd | 0xc0))
        }
    }
}

#[cfg(windows)]
mod platform {
    use super::*;
    use std::{
        collections::{BTreeSet, VecDeque},
        io::{self, BufRead, Write},
        mem::{size_of, zeroed},
        ptr::null_mut,
        sync::{
            atomic::{AtomicBool, AtomicUsize, Ordering},
            mpsc, Arc, Mutex, OnceLock,
        },
        thread,
        time::{Duration, Instant},
    };
    use windows_sys::Win32::{
        Foundation::*,
        Graphics::Gdi::*,
        System::{LibraryLoader::GetModuleHandleW, Threading::*},
        UI::{HiDpi::*, Input::KeyboardAndMouse::*, WindowsAndMessaging::*},
    };

    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    static TARGET: AtomicUsize = AtomicUsize::new(0);
    static STOP: AtomicBool = AtomicBool::new(false);
    static OVERFLOW: AtomicBool = AtomicBool::new(false);
    static EVENTS: Mutex<Vec<serde_json::Value>> = Mutex::new(Vec::new());
    static LOG: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());
    fn note(message: &str) {
        eprintln!("{message}");
        if let Ok(mut log) = LOG.lock() {
            if log.len() >= 64 {
                log.pop_front();
            }
            log.push_back(message.to_string());
        }
    }
    fn respond(cmd: &serde_json::Value, result: Result<(serde_json::Value, Vec<u8>), String>) {
        let (mut response, bytes) = match result {
            Ok(value) => value,
            Err(error) => (serde_json::json!({"error": error}), Vec::new()),
        };
        if let Some(id) = cmd.get("id").filter(|id| !id.is_null()) {
            response["id"] = id.clone();
        }
        response["bytes"] = serde_json::json!(bytes.len());
        let mut out = io::stdout().lock();
        if writeln!(out, "{response}").is_ok() {
            let _ = out.write_all(&bytes);
            let _ = out.flush();
        }
    }
    fn disarm(shared: &Arc<Mutex<InputState>>) {
        if let Ok(mut state) = shared.lock() {
            state.held.release();
            state.armed = false;
        }
    }
    fn ns() -> u64 {
        ORIGIN.get_or_init(Instant::now).elapsed().as_nanos() as u64
    }
    fn foreground() -> bool {
        unsafe {
            GetForegroundWindow() as usize == TARGET.load(Ordering::Relaxed)
                && TARGET.load(Ordering::Relaxed) != 0
        }
    }
    /// Where the pointer is in client pixels, including positions outside the window.
    /// Injected moves and a human hand both land here, because both move the OS cursor.
    unsafe fn client_cursor(hwnd: HWND) -> Result<(i32, i32), String> {
        let mut point: POINT = zeroed();
        if GetCursorPos(&mut point) == 0 || ScreenToClient(hwnd, &mut point) == 0 {
            return Err("cursor_unavailable".into());
        }
        Ok((point.x, point.y))
    }
    fn record(e: Event) {
        if !foreground() {
            return;
        }
        if let Ok(mut queue) = EVENTS.lock() {
            if queue.len() >= 100_000 {
                OVERFLOW.store(true, Ordering::Relaxed);
                return;
            }
            queue.push(serde_json::json!({"t_ns":ns(),"event":e}));
        }
    }
    unsafe extern "system" fn keyboard(code: i32, wp: WPARAM, lp: LPARAM) -> LRESULT {
        if code >= 0 {
            let k = &*(lp as *const KBDLLHOOKSTRUCT);
            if k.flags & LLKHF_INJECTED == 0 {
                let down = wp as u32 == WM_KEYDOWN || wp as u32 == WM_SYSKEYDOWN;
                if k.vkCode == VK_F12 as u32 && down {
                    STOP.store(true, Ordering::SeqCst);
                }
                record(Event::Key {
                    vk: match k.vkCode {
                        0xa0 | 0xa1 => 0x10,
                        0xa2 | 0xa3 => 0x11,
                        v => v as u16,
                    },
                    down,
                });
            }
        }
        CallNextHookEx(null_mut(), code, wp, lp)
    }
    unsafe extern "system" fn mouse(code: i32, wp: WPARAM, lp: LPARAM) -> LRESULT {
        if code >= 0 && foreground() {
            let m = &*(lp as *const MSLLHOOKSTRUCT);
            if m.flags & LLMHF_INJECTED == 0 {
                let event = match wp as u32 {
                    WM_MOUSEMOVE => {
                        let hwnd = TARGET.load(Ordering::Relaxed) as HWND;
                        let mut p = m.pt;
                        let mut r: RECT = zeroed();
                        ScreenToClient(hwnd, &mut p);
                        GetClientRect(hwnd, &mut r);
                        if r.right < 2 || r.bottom < 2 {
                            return CallNextHookEx(null_mut(), code, wp, lp);
                        }
                        Event::Move {
                            x: (p.x as f64 / (r.right - 1) as f64).clamp(0., 1.),
                            y: (p.y as f64 / (r.bottom - 1) as f64).clamp(0., 1.),
                        }
                    }
                    WM_LBUTTONDOWN => Event::Button {
                        button: 0,
                        down: true,
                    },
                    WM_LBUTTONUP => Event::Button {
                        button: 0,
                        down: false,
                    },
                    WM_RBUTTONDOWN => Event::Button {
                        button: 1,
                        down: true,
                    },
                    WM_RBUTTONUP => Event::Button {
                        button: 1,
                        down: false,
                    },
                    WM_MBUTTONDOWN => Event::Button {
                        button: 2,
                        down: true,
                    },
                    WM_MBUTTONUP => Event::Button {
                        button: 2,
                        down: false,
                    },
                    WM_MOUSEWHEEL => Event::Wheel {
                        delta: ((m.mouseData >> 16) as i16) as i32,
                    },
                    _ => return CallNextHookEx(null_mut(), code, wp, lp),
                };
                record(event);
            }
        }
        CallNextHookEx(null_mut(), code, wp, lp)
    }
    fn hooks() -> Result<(), String> {
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || unsafe {
            let module = GetModuleHandleW(null_mut());
            let k = SetWindowsHookExW(WH_KEYBOARD_LL, Some(keyboard), module, 0);
            let m = SetWindowsHookExW(WH_MOUSE_LL, Some(mouse), module, 0);
            if k.is_null() || m.is_null() {
                let _ = tx.send(false);
                return;
            }
            let _ = tx.send(true);
            let mut msg: MSG = zeroed();
            while GetMessageW(&mut msg, null_mut(), 0, 0) > 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            UnhookWindowsHookEx(k);
            UnhookWindowsHookEx(m);
        });
        if rx.recv_timeout(Duration::from_secs(3)).unwrap_or(false) {
            Ok(())
        } else {
            Err("input_hooks_unavailable".into())
        }
    }
    unsafe extern "system" fn enumerate(hwnd: HWND, lp: LPARAM) -> BOOL {
        if IsWindowVisible(hwnd) == 0 {
            return 1;
        }
        let mut pid = 0;
        GetWindowThreadProcessId(hwnd, &mut pid);
        let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if process.is_null() {
            return 1;
        }
        let mut buf = [0u16; 32768];
        let mut n = buf.len() as u32;
        let ok = QueryFullProcessImageNameW(process, 0, buf.as_mut_ptr(), &mut n);
        CloseHandle(process);
        if ok != 0
            && String::from_utf16_lossy(&buf[..n as usize])
                .to_lowercase()
                .ends_with("\\hoi4.exe")
        {
            let mut title = [0u16; 512];
            let len = GetWindowTextW(hwnd, title.as_mut_ptr(), 512);
            if len > 0 {
                (*(lp as *mut Vec<(usize, String)>)).push((
                    hwnd as usize,
                    String::from_utf16_lossy(&title[..len as usize]),
                ));
            }
        }
        1
    }
    unsafe fn select() -> Result<HWND, String> {
        let mut windows: Vec<(usize, String)> = Vec::new();
        EnumWindows(Some(enumerate), &mut windows as *mut _ as LPARAM);
        if windows.len() != 1 {
            return Err(format!("expected_one_hoi4_window_found_{}", windows.len()));
        }
        Ok(windows[0].0 as HWND)
    }
    /// The game's client rectangle, from Desktop Duplication if it is available.
    ///
    /// The blit below costs 87.1 ms p50 at 3840x2160, measured by the worker's own
    /// `capture_start_ns` and `t_ns`, and it is the largest single term in a 200 ms tick.
    /// Duplication reads a texture the compositor already holds instead.
    ///
    /// GDI stays as the fallback and that is not politeness. A duplication is lost on a
    /// resolution change, a full-screen transition, a driver reset or a session switch,
    /// and a match must not end because the display mode did. A lost one is dropped here
    /// and rebuilt on the next tick; if the rebuild fails the blit still works. The reply
    /// reports which backend produced the pixels, because that field is already recorded
    /// with every run and a silent downgrade to a 87 ms capture would otherwise look
    /// like the game got slower.
    /// The duplication, and whether it is still worth asking for one.
    ///
    /// Building one costs a D3D11 device and an output enumeration, so a machine that
    /// simply cannot duplicate -- a remote session, an adapter that refuses -- must not
    /// pay for the attempt five times a second forever. A creation failure is final for
    /// the connection. A duplication *lost* at runtime is a different thing: a
    /// resolution change or a driver reset is recoverable and gets a bounded number of
    /// rebuilds before the worker settles for the blit and stops trying.
    enum Screen {
        Untried,
        Active(Box<crate::duplication::Duplicator>),
        Retired,
    }

    /// How many lost duplications to rebuild through before giving up on the fast path.
    const DUPLICATION_REBUILDS: u32 = 3;

    /// One captured client rectangle, and which backend produced it.
    struct Capture {
        bytes: Vec<u8>,
        width: i32,
        height: i32,
        start_ns: u64,
        end_ns: u64,
        backend: &'static str,
    }

    unsafe fn capture(
        screen: &mut Screen,
        rebuilds: &mut u32,
        hwnd: HWND,
    ) -> Result<Capture, String> {
        let start = ns();
        if IsIconic(hwnd) != 0 || !foreground() {
            return Err("game_not_foreground".into());
        }
        let mut r: RECT = zeroed();
        GetClientRect(hwnd, &mut r);
        let (w, h) = (r.right, r.bottom);
        if w <= 0 || h <= 0 || w > 8192 || h > 8192 {
            return Err("invalid_client_geometry".into());
        }
        let mut origin = POINT { x: 0, y: 0 };
        if ClientToScreen(hwnd, &mut origin) == 0 {
            return Err("client_to_screen_failed".into());
        }
        if matches!(screen, Screen::Untried) {
            *screen = match crate::duplication::Duplicator::new((origin.x, origin.y)) {
                Ok(duplicator) => Screen::Active(Box::new(duplicator)),
                Err(reason) => {
                    note(&format!(
                        "desktop duplication unavailable, using gdi: {reason}"
                    ));
                    Screen::Retired
                }
            };
        }
        if let Screen::Active(duplicator) = screen {
            match duplicator.client((origin.x, origin.y), w as usize, h as usize) {
                Ok(bytes) => {
                    // The same post-condition the blit is held to: a frame captured as
                    // focus left the game is a frame of something else.
                    if !foreground() {
                        return Err("capture_failed_or_focus_changed".into());
                    }
                    return Ok(Capture {
                        bytes,
                        width: w,
                        height: h,
                        start_ns: start,
                        end_ns: ns(),
                        backend: "dxgi_bgra",
                    });
                }
                // Nothing presented yet is not a failure. The duplication is healthy
                // and will serve the next tick; this one falls through to the blit
                // rather than spending a rebuild -- a still screen is the screen
                // duplication handles best and must not be the one that retires it.
                Err(crate::duplication::Unavailable::NotReadyYet) => {}
                Err(reason) => {
                    // A geometry miss rebuilds against the output that now holds the
                    // window. It does not spend the retirement budget.
                    if crate::duplication::should_retire(&reason) {
                        *rebuilds += 1;
                    }
                    *screen = if *rebuilds <= DUPLICATION_REBUILDS {
                        note(&format!("desktop duplication lost, rebuilding: {reason}"));
                        Screen::Untried
                    } else {
                        note(&format!(
                            "desktop duplication lost {rebuilds} times, using gdi: {reason}"
                        ));
                        Screen::Retired
                    };
                }
            }
        }
        // A legacy application's own DC can remain DPI-virtualized even when
        // GetClientRect returns physical pixels. Capture the foreground client
        // rectangle from the desktop DC to keep observation/input coordinates equal.
        let dc = GetDC(null_mut());
        let mem = CreateCompatibleDC(dc);
        let bitmap = CreateCompatibleBitmap(dc, w, h);
        if dc.is_null() || mem.is_null() || bitmap.is_null() {
            if !bitmap.is_null() {
                DeleteObject(bitmap);
            }
            if !mem.is_null() {
                DeleteDC(mem);
            }
            if !dc.is_null() {
                ReleaseDC(null_mut(), dc);
            }
            return Err("capture_allocation_failed".into());
        }
        let old = SelectObject(mem, bitmap);
        let ok = BitBlt(
            mem,
            0,
            0,
            w,
            h,
            dc,
            origin.x,
            origin.y,
            SRCCOPY | CAPTUREBLT,
        );
        SelectObject(mem, old);
        let mut info: BITMAPINFO = zeroed();
        info.bmiHeader.biSize = size_of::<BITMAPINFOHEADER>() as u32;
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
        ReleaseDC(null_mut(), dc);
        if lines != h || !foreground() {
            return Err("capture_failed_or_focus_changed".into());
        }
        Ok(Capture {
            bytes,
            width: w,
            height: h,
            start_ns: start,
            end_ns: ns(),
            backend: "gdi_bgra",
        })
    }
    unsafe fn inject(hwnd: HWND, e: &Event) -> Result<(), String> {
        let mut input: INPUT = zeroed();
        match *e {
            Event::Key { vk, down } => {
                input.r#type = INPUT_KEYBOARD;
                input.Anonymous.ki = KEYBDINPUT {
                    wVk: 0,
                    wScan: MapVirtualKeyW(vk as u32, MAPVK_VK_TO_VSC) as u16,
                    dwFlags: KEYEVENTF_SCANCODE
                        | if down { 0 } else { KEYEVENTF_KEYUP }
                        | if (0x25..=0x28).contains(&vk) {
                            KEYEVENTF_EXTENDEDKEY
                        } else {
                            0
                        },
                    time: 0,
                    dwExtraInfo: 0,
                };
            }
            Event::Move { x, y } => {
                let mut r: RECT = zeroed();
                GetClientRect(hwnd, &mut r);
                let mut p = POINT {
                    x: (x * (r.right - 1) as f64).round() as i32,
                    y: (y * (r.bottom - 1) as f64).round() as i32,
                };
                ClientToScreen(hwnd, &mut p);
                let left = GetSystemMetrics(SM_XVIRTUALSCREEN);
                let top = GetSystemMetrics(SM_YVIRTUALSCREEN);
                let w = GetSystemMetrics(SM_CXVIRTUALSCREEN);
                let h = GetSystemMetrics(SM_CYVIRTUALSCREEN);
                input.r#type = INPUT_MOUSE;
                input.Anonymous.mi = MOUSEINPUT {
                    dx: ((p.x - left) as i64 * 65535 / (w - 1) as i64) as i32,
                    dy: ((p.y - top) as i64 * 65535 / (h - 1) as i64) as i32,
                    mouseData: 0,
                    dwFlags: MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
                    time: 0,
                    dwExtraInfo: 0,
                };
            }
            Event::Button { button, down } => {
                input.r#type = INPUT_MOUSE;
                input.Anonymous.mi.dwFlags = match (button, down) {
                    (0, true) => MOUSEEVENTF_LEFTDOWN,
                    (0, false) => MOUSEEVENTF_LEFTUP,
                    (1, true) => MOUSEEVENTF_RIGHTDOWN,
                    (1, false) => MOUSEEVENTF_RIGHTUP,
                    (2, true) => MOUSEEVENTF_MIDDLEDOWN,
                    _ => MOUSEEVENTF_MIDDLEUP,
                };
            }
            Event::Wheel { delta } => {
                input.r#type = INPUT_MOUSE;
                input.Anonymous.mi.dwFlags = MOUSEEVENTF_WHEEL;
                input.Anonymous.mi.mouseData = delta as u32;
            }
        }
        if SendInput(1, &input, size_of::<INPUT>() as i32) != 1 {
            Err("send_input_failed".into())
        } else {
            Ok(())
        }
    }
    struct Held {
        keys: BTreeSet<u16>,
        buttons: BTreeSet<u8>,
        hwnd: usize,
    }
    impl Held {
        fn apply(&mut self, e: &Event) -> Result<(), String> {
            unsafe {
                inject(self.hwnd as HWND, e)?;
            }
            match *e {
                Event::Key { vk, down } => {
                    if down {
                        self.keys.insert(vk);
                    } else {
                        self.keys.remove(&vk);
                    }
                }
                Event::Button { button, down } => {
                    if down {
                        self.buttons.insert(button);
                    } else {
                        self.buttons.remove(&button);
                    }
                }
                _ => {}
            }
            Ok(())
        }
        fn release(&mut self) {
            for vk in std::mem::take(&mut self.keys) {
                unsafe {
                    let _ = inject(self.hwnd as HWND, &Event::Key { vk, down: false });
                }
            }
            for button in std::mem::take(&mut self.buttons) {
                unsafe {
                    let _ = inject(
                        self.hwnd as HWND,
                        &Event::Button {
                            button,
                            down: false,
                        },
                    );
                }
            }
        }
    }
    impl Drop for Held {
        fn drop(&mut self) {
            self.release();
        }
    }
    struct InputState {
        held: Held,
        armed: bool,
        setup: bool,
        last: Instant,
    }
    // The watchdog owns another Arc, so InputState's destructor cannot handle a
    // broken stdout pipe. Release on every return/unwind from the request loop.
    struct ReleaseOnExit(Arc<Mutex<InputState>>);
    impl Drop for ReleaseOnExit {
        fn drop(&mut self) {
            let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
            state.held.release();
            state.armed = false;
        }
    }
    /// Arm, release, apply, and status. These stay off the capture thread so a blit
    /// cannot bunch the 25 ms slots up behind it.
    /// HOI4's game.log under the user's Documents folder, which may be redirected.
    fn game_log_path() -> Result<std::path::PathBuf, String> {
        use windows_sys::Win32::System::Com::CoTaskMemFree;
        use windows_sys::Win32::UI::Shell::{FOLDERID_Documents, SHGetKnownFolderPath};
        unsafe {
            let mut path = null_mut();
            if SHGetKnownFolderPath(&FOLDERID_Documents, 0, null_mut(), &mut path) != 0 {
                return Err("documents_folder_unknown".into());
            }
            let len = (0..).take_while(|&i| *path.add(i) != 0).count();
            let documents = String::from_utf16_lossy(std::slice::from_raw_parts(path, len));
            CoTaskMemFree(path as _);
            Ok(std::path::Path::new(&documents)
                .join("Paradox Interactive")
                .join("Hearts of Iron IV")
                .join("logs")
                .join("game.log"))
        }
    }

    /// New arena lines after `offset`, and the offset to ask from next time. A log
    /// shorter than the offset belongs to a newer game and is read from the start.
    fn read_arena_log(offset: u64) -> Result<(Vec<String>, u64), String> {
        use std::io::{Read, Seek, SeekFrom};
        let path = game_log_path()?;
        let mut file = match std::fs::File::open(&path) {
            Ok(file) => file,
            Err(_) => return Ok((vec![], 0)),
        };
        let len = file.metadata().map_err(|e| e.to_string())?.len();
        let start = if offset > len { 0 } else { offset };
        file.seek(SeekFrom::Start(start))
            .map_err(|e| e.to_string())?;
        let mut bytes = Vec::new();
        file.take(1 << 20)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        let (lines, used) = arena_lines(&bytes);
        Ok((lines, start + used as u64))
    }

    fn fast_op(
        shared: &Arc<Mutex<InputState>>,
        cmd: &serde_json::Value,
    ) -> Result<(serde_json::Value, Vec<u8>), String> {
        let mut state = shared.lock().map_err(|_| "input_lock")?;
        let InputState {
            held,
            armed,
            setup,
            last,
        } = &mut *state;
        match cmd["op"].as_str().unwrap_or("") {
            "arm" => {
                if !foreground() {
                    return Err("game_not_foreground".into());
                }
                *setup = cmd["mode"] == "setup";
                STOP.store(false, Ordering::SeqCst);
                *armed = true;
                *last = Instant::now();
                Ok((serde_json::json!({"armed": true}), vec![]))
            }
            "release" => {
                held.release();
                *armed = false;
                Ok((serde_json::json!({"armed": false}), vec![]))
            }
            // Bring the attached game window to the front, for setup only. A windowed
            // game started by a background process does not take focus, and nobody may be
            // at the second PC to click it. Windows only lets a process that has just sent
            // input change the foreground window, so this taps Alt first; the tap goes to
            // whatever window had focus, before the game has it.
            "focus" => {
                if *armed {
                    return Err("focus_refused_while_armed".into());
                }
                let hwnd = TARGET.load(Ordering::Relaxed) as HWND;
                if hwnd.is_null() {
                    return Err("focus_before_attach".into());
                }
                unsafe {
                    if IsIconic(hwnd) != 0 {
                        ShowWindow(hwnd, SW_RESTORE);
                    }
                    let mut alt: [INPUT; 2] = zeroed();
                    for (i, up) in [false, true].into_iter().enumerate() {
                        alt[i].r#type = INPUT_KEYBOARD;
                        alt[i].Anonymous.ki.wVk = VK_MENU;
                        alt[i].Anonymous.ki.dwFlags = if up { KEYEVENTF_KEYUP } else { 0 };
                    }
                    SendInput(2, alt.as_ptr(), size_of::<INPUT>() as i32);
                    SetForegroundWindow(hwnd);
                }
                thread::sleep(Duration::from_millis(150));
                Ok((serde_json::json!({"foreground": foreground()}), vec![]))
            }
            "events" => {
                let events = std::mem::take(&mut *EVENTS.lock().map_err(|_| "event_lock")?);
                Ok((
                    serde_json::json!({"events": events, "t_ns": ns(), "overflow": OVERFLOW.swap(false, Ordering::Relaxed)}),
                    vec![],
                ))
            }
            "apply" => {
                if !*armed || STOP.load(Ordering::SeqCst) || !foreground() {
                    return Err("input_not_armed_or_focus_lost".into());
                }
                let events: Vec<Event> =
                    serde_json::from_value(cmd["events"].clone()).map_err(|e| e.to_string())?;
                if events.len() > 64 || !events.iter().all(|e| valid_event(e, *setup)) {
                    return Err("invalid_event_batch".into());
                }
                for e in &events {
                    if !foreground() || STOP.load(Ordering::SeqCst) {
                        held.release();
                        *armed = false;
                        return Err("focus_lost_during_batch".into());
                    }
                    held.apply(e)?;
                }
                *last = Instant::now();
                Ok((
                    serde_json::json!({"applied": events.len(), "t_ns": ns()}),
                    vec![],
                ))
            }
            // The arena mod's own lines from HOI4's game.log: surrenders, peace, states
            // changing hands and a weekly count per country. Read-only, one fixed file,
            // and only lines the mod wrote, so it exposes nothing else on the machine.
            "game_log" => {
                let offset = cmd["offset"].as_u64().unwrap_or(0);
                let (lines, offset) = read_arena_log(offset)?;
                Ok((
                    serde_json::json!({"lines": lines, "offset": offset}),
                    vec![],
                ))
            }
            "status" => {
                let log_lines: Vec<String> = LOG
                    .lock()
                    .map(|log| log.iter().cloned().collect())
                    .unwrap_or_default();
                Ok((
                    serde_json::json!({
                        "armed": *armed,
                        "foreground": foreground(),
                        "stopped": STOP.load(Ordering::SeqCst),
                        "held_keys": held.keys,
                        "held_buttons": held.buttons,
                        "t_ns": ns(),
                        "log": log_lines,
                    }),
                    vec![],
                ))
            }
            _ => Err("unknown_operation".into()),
        }
    }

    pub fn run() -> Result<(), String> {
        unsafe {
            SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        }
        ORIGIN.get_or_init(Instant::now);
        hooks()?;
        let shared = Arc::new(Mutex::new(InputState {
            held: Held {
                keys: BTreeSet::new(),
                buttons: BTreeSet::new(),
                hwnd: 0,
            },
            armed: false,
            setup: false,
            last: Instant::now(),
        }));
        let _release_on_exit = ReleaseOnExit(Arc::clone(&shared));
        let safety = Arc::clone(&shared);
        thread::spawn(move || loop {
            thread::sleep(Duration::from_millis(10));
            let mut state = safety.lock().unwrap_or_else(|e| e.into_inner());
            if state.armed
                && (STOP.load(Ordering::SeqCst)
                    || !foreground()
                    || state.last.elapsed() > Duration::from_millis(750))
            {
                state.held.release();
                state.armed = false;
            }
        });
        // Unbounded: a capture in progress must not stop the reader from pulling the
        // next apply off stdin. A bounded send here would freeze the slots behind the blit.
        let (tx, rx) = mpsc::channel();
        let reader_state = Arc::clone(&shared);
        thread::spawn(move || {
            for line in io::stdin().lock().lines() {
                let line = match line {
                    Ok(v) => v,
                    Err(_) => break,
                };
                let cmd: serde_json::Value = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(error) => {
                        respond(&serde_json::Value::Null, Err(error.to_string()));
                        continue;
                    }
                };
                let op = cmd["op"].as_str().unwrap_or("");
                if op == "capture" || op == "attach" {
                    if tx.send(cmd).is_err() {
                        break;
                    }
                    continue;
                }
                let result = fast_op(&reader_state, &cmd);
                if result.is_err() {
                    disarm(&reader_state);
                }
                respond(&cmd, result);
            }
        });
        let mut seq = 0u64;
        // One duplication per process, held across ticks: the device, the output
        // enumeration and the staging texture all cost far more to create than the
        // capture they serve. It stays on this thread. Apply runs on the reader.
        let mut screen = Screen::Untried;
        let mut rebuilds = 0u32;
        while let Ok(cmd) = rx.recv() {
            let result = (|| -> Result<(serde_json::Value, Vec<u8>), String> {
                match cmd["op"].as_str().unwrap_or("") {
                    "attach" => {
                        let hwnd = {
                            let mut state = shared.lock().map_err(|_| "input_lock")?;
                            state.held.release();
                            state.armed = false;
                            let hwnd = unsafe { select()? };
                            state.held.hwnd = hwnd as usize;
                            TARGET.store(hwnd as usize, Ordering::Relaxed);
                            hwnd
                        };
                        // The probe capture is the slow part. Apply can proceed while it runs.
                        // A failed probe is not a GDI frame; saying so hid a game that was
                        // not in front.
                        screen = Screen::Untried;
                        rebuilds = 0;
                        let attached_backend =
                            match unsafe { capture(&mut screen, &mut rebuilds, hwnd) } {
                                Ok(frame) => frame.backend.to_string(),
                                Err(reason) => {
                                    note(&format!("attach probe failed: {reason}"));
                                    "unavailable".to_string()
                                }
                            };
                        Ok((
                            serde_json::json!({"hwnd": hwnd as usize, "foreground": foreground(), "clock_ns": ns(), "backend": attached_backend, "computer": std::env::var("COMPUTERNAME").unwrap_or_default()}),
                            vec![],
                        ))
                    }
                    "capture" => {
                        let hwnd = {
                            let state = shared.lock().map_err(|_| "input_lock")?;
                            state.held.hwnd as HWND
                        };
                        let Capture {
                            bytes: raw,
                            width: w,
                            height: h,
                            start_ns: start,
                            end_ns: end,
                            backend,
                        } = unsafe { capture(&mut screen, &mut rebuilds, hwnd)? };
                        let (uw, uh) = (w as usize, h as usize);
                        let (cx, cy) = unsafe { client_cursor(hwnd)? };
                        let view_size = cmd["views"].as_u64().unwrap_or(0) as usize;
                        if view_size > 1024 {
                            return Err("view_size_too_large".into());
                        }
                        let mut regions: Vec<[usize; 4]> = Vec::new();
                        if let Some(list) = cmd["regions"].as_array() {
                            if list.len() > 64 {
                                return Err("too_many_regions".into());
                            }
                            for item in list {
                                let v: Vec<i64> = serde_json::from_value(item.clone())
                                    .map_err(|e| e.to_string())?;
                                if v.len() != 4 || v.iter().any(|&n| n < 0) {
                                    return Err("invalid_region".into());
                                }
                                let (x, y, rw, rh) =
                                    (v[0] as usize, v[1] as usize, v[2] as usize, v[3] as usize);
                                if rw == 0 || rh == 0 || x + rw > uw || y + rh > uh {
                                    return Err("region_outside_frame".into());
                                }
                                regions.push([y, x, rh, rw]);
                            }
                        }
                        let want_full = if view_size == 0 && regions.is_empty() {
                            true
                        } else {
                            cmd["full"].as_bool().unwrap_or(false)
                        };
                        let mut payload = Vec::new();
                        let full_bytes = if want_full { raw.len() } else { 0 };
                        if want_full {
                            payload.extend_from_slice(&raw);
                        }
                        let mut views_bytes = 0usize;
                        if view_size > 0 {
                            for b in view_boxes(uw, uh) {
                                let v = downscale_bgra(&raw, uw, b, view_size);
                                views_bytes += v.len();
                                payload.extend_from_slice(&v);
                            }
                            // Cropped from the same frame as the other views, on either
                            // backend. prepare_session crops the recorded full frame, so a
                            // separate, later blit would disagree exactly at the pointer.
                            let v = cursor_crop_bgra(&raw, uw, uh, cx, cy, view_size);
                            views_bytes += v.len();
                            payload.extend_from_slice(&v);
                        }
                        let mut region_bytes: Vec<usize> = Vec::new();
                        for r in &regions {
                            let [top, left, rh, rw] = *r;
                            let mut crop = Vec::with_capacity(rw * rh * 4);
                            for y in 0..rh {
                                let i = ((top + y) * uw + left) * 4;
                                crop.extend_from_slice(&raw[i..i + rw * 4]);
                            }
                            region_bytes.push(crop.len());
                            payload.extend_from_slice(&crop);
                        }
                        let encoding = if cmd["encoding"] == "lz4" {
                            "lz4"
                        } else {
                            "raw"
                        };
                        let bytes = if encoding == "lz4" {
                            lz4_flex::block::compress(&payload)
                        } else {
                            payload
                        };
                        seq += 1;
                        let events = std::mem::take(&mut *EVENTS.lock().map_err(|_| "event_lock")?);
                        Ok((
                            serde_json::json!({"seq": seq, "width": w, "height": h, "encoding": encoding, "capture_start_ns": start, "t_ns": end, "events": events, "overflow": OVERFLOW.swap(false, Ordering::Relaxed), "stopped": STOP.load(Ordering::SeqCst), "foreground": foreground(), "cursor": [cx, cy], "full_bytes": full_bytes, "view_size": view_size, "views_bytes": views_bytes, "region_bytes": region_bytes, "backend": backend}),
                            bytes,
                        ))
                    }
                    _ => fast_op(&shared, &cmd),
                }
            })();
            if result.is_err() {
                disarm(&shared);
            }
            respond(&cmd, result);
        }
        shared
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .held
            .release();
        Ok(())
    }
}

fn main() {
    #[cfg(windows)]
    if let Err(e) = platform::run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
    #[cfg(not(windows))]
    {
        eprintln!("The desktop worker requires Windows.");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Deterministic source bytes shared with the Python side of this check.
    fn lcg(n: usize) -> Vec<u8> {
        let mut s: u64 = 12345;
        (0..n)
            .map(|_| {
                s = (1103515245u64.wrapping_mul(s).wrapping_add(12345)) & 0x7FFF_FFFF;
                ((s >> 16) & 0xFF) as u8
            })
            .collect()
    }

    /// The worker's downscale must equal the training resize exactly, not approximately.
    ///
    /// These vectors were produced by `hoi4_arena.dataset.views`' resampler
    /// (torch `F.interpolate(mode="area")`) and the identical constants are asserted from
    /// the Python side in test_worker_downscale_matches_training_resize. If either
    /// implementation drifts, one of the two tests fails. A filter mismatch here would not
    /// crash anything; it would quietly feed the policy different pixels than it trained
    /// on, which is why this is pinned rather than eyeballed.
    #[test]
    fn downscale_matches_the_training_resize() {
        let cases: [(usize, usize, usize, &[u8]); 3] = [
            (
                12,
                8,
                3,
                &[
                    124, 133, 134, 126, 104, 79, 159, 144, 150, 144, 135, 132, 126, 131, 128, 167,
                    142, 133, 112, 146, 125, 119, 129, 154, 148, 117, 118,
                ],
            ),
            (
                7,
                5,
                3,
                &[
                    96, 103, 140, 172, 105, 110, 210, 58, 109, 94, 130, 129, 149, 110, 99, 162,
                    116, 141, 144, 159, 102, 149, 138, 164, 151, 186, 218,
                ],
            ),
            (
                16,
                9,
                4,
                &[
                    116, 167, 123, 160, 96, 116, 145, 123, 130, 154, 106, 105, 144, 137, 177, 162,
                    130, 107, 124, 149, 104, 132, 117, 128, 144, 122, 167, 135, 137, 98, 137, 149,
                    142, 125, 122, 109, 135, 121, 158, 121, 137, 112, 164, 142, 117, 131, 142, 133,
                ],
            ),
        ];
        for (w, h, size, expected) in cases {
            let src = lcg(w * h * 4);
            let got = downscale_bgra(&src, w, [0, 0, h, w], size);
            assert_eq!(got, expected, "{w}x{h} -> {size}x{size}");
        }
    }

    /// Rounding ties must go to even, as torch does. Every channel here averages
    /// exactly x.5, so half-away-from-zero would give 1,3,5 instead of 0,2,4.
    /// The scalar loop the threaded downscale replaced, kept as the thing it must equal.
    fn downscale_serial(src: &[u8], width: usize, box_: [usize; 4], size: usize) -> Vec<u8> {
        let [top, left, bh, bw] = box_;
        let mut out = vec![0u8; size * size * 3];
        for oy in 0..size {
            let y0 = oy * bh / size;
            let y1 = ((oy + 1) * bh).div_ceil(size);
            for ox in 0..size {
                let x0 = ox * bw / size;
                let x1 = ((ox + 1) * bw).div_ceil(size);
                let (mut sr, mut sg, mut sb) = (0u32, 0u32, 0u32);
                for y in y0..y1 {
                    let row = (top + y) * width * 4;
                    for x in x0..x1 {
                        let i = row + (left + x) * 4;
                        sb += src[i] as u32;
                        sg += src[i + 1] as u32;
                        sr += src[i + 2] as u32;
                    }
                }
                let n = ((y1 - y0) * (x1 - x0)) as f32;
                let o = (oy * size + ox) * 3;
                out[o] = (sr as f32 / n).round_ties_even().clamp(0., 255.) as u8;
                out[o + 1] = (sg as f32 / n).round_ties_even().clamp(0., 255.) as u8;
                out[o + 2] = (sb as f32 / n).round_ties_even().clamp(0., 255.) as u8;
            }
        }
        out
    }

    /// Splitting the output rows across threads must not move a single byte.
    ///
    /// Every output pixel reduces its own disjoint source box, so this should hold by
    /// construction -- which is exactly why it is worth asserting, because "should hold
    /// by construction" is how a stripe boundary off by one row gets shipped. The sizes
    /// straddle the thread count in both directions so the serial fallback and the
    /// striped path are both exercised, and the quadrant boxes are included because they
    /// are the ones with a nonzero origin.
    #[test]
    fn threaded_downscale_is_byte_identical_to_the_serial_loop() {
        let (w, h) = (96usize, 64usize);
        let src = lcg(w * h * 4);
        for size in [1usize, 2, 3, 5, 7, 8, 16, 32] {
            for b in view_boxes(w, h) {
                assert_eq!(
                    downscale_bgra(&src, w, b, size),
                    downscale_serial(&src, w, b, size),
                    "size {size} box {b:?} differs between the threaded and serial loops"
                );
            }
        }
    }

    /// The thread count stays inside the range the measurements cover.
    #[test]
    fn downscale_thread_count_is_bounded() {
        let threads = downscale_threads();
        assert!(
            (1..=4).contains(&threads),
            "downscale_threads returned {threads}, outside the measured 1..=4"
        );
    }

    #[test]
    fn downscale_rounds_ties_to_even() {
        let src = [4u8, 2, 0, 0, 5, 3, 1, 0];
        assert_eq!(downscale_bgra(&src, 2, [0, 0, 1, 2], 1), vec![0, 2, 4]);
    }

    /// The four quadrants must arrive in the order the policy's detail encoder expects:
    /// top-left, top-right, bottom-left, bottom-right. Each quadrant here is a distinct
    /// constant, so any permutation changes the bytes.
    #[test]
    fn view_boxes_are_ordered_top_left_top_right_bottom_left_bottom_right() {
        let mut src = vec![0u8; 4 * 4 * 4];
        for (idx, (top, left)) in [(0, 0), (0, 2), (2, 0), (2, 2)].iter().enumerate() {
            let v = 10 * (idx as u8 + 1);
            for y in *top..*top + 2 {
                for x in *left..*left + 2 {
                    let i = (y * 4 + x) * 4;
                    src[i] = v + 2;
                    src[i + 1] = v + 1;
                    src[i + 2] = v;
                }
            }
        }
        let boxes = view_boxes(4, 4);
        assert_eq!(downscale_bgra(&src, 4, boxes[0], 1), vec![25, 26, 27]);
        for (n, expected) in [[10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42]]
            .iter()
            .enumerate()
        {
            assert_eq!(
                downscale_bgra(&src, 4, boxes[n + 1], 1),
                expected.to_vec(),
                "quadrant {n} is out of order"
            );
        }
    }

    /// Same 3x3 image and same RGB bytes as test_cursor_crop_keeps_the_pointer_pixel_centered.
    #[test]
    fn cursor_crop_keeps_the_pointer_pixel_and_pads_outside_with_zero() {
        let rgb = [
            1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27,
        ];
        let mut src = Vec::with_capacity(3 * 3 * 4);
        for px in rgb.chunks(3) {
            src.extend_from_slice(&[px[2], px[1], px[0], 0]);
        }
        assert_eq!(cursor_crop_bgra(&src, 3, 3, 1, 1, 3), rgb);
        assert_eq!(
            cursor_crop_bgra(&src, 3, 3, 0, 0, 3),
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 10, 11, 12, 13, 14,
                15,
            ]
        );
        assert_eq!(cursor_crop_bgra(&src, 3, 3, 9, -4, 3), vec![0u8; 27]);
    }

    /// An in-bounds window is a 1:1 area average, so the crop and the resampler agree.
    #[test]
    fn interior_cursor_crop_matches_downscale_of_that_box() {
        let (w, h, size) = (8usize, 8usize, 4usize);
        let src = lcg(w * h * 4);
        // Center index is size/2 = 2, so cursor (5, 6) opens the box at (3, 4).
        assert_eq!(
            cursor_crop_bgra(&src, w, h, 5, 6, size),
            downscale_bgra(&src, w, [4, 3, size, size], size)
        );
    }

    #[test]
    fn view_boxes_cover_the_frame_without_overlap() {
        let (w, h) = (3840usize, 2160usize);
        let boxes = view_boxes(w, h);
        assert_eq!(boxes[0], [0, 0, h, w]);
        let area: usize = boxes[1..].iter().map(|b| b[2] * b[3]).sum();
        assert_eq!(area, w * h, "quadrants must tile the frame exactly");
        // Odd dimensions must still tile, with the far quadrants taking the extra pixel.
        let odd = view_boxes(7, 5);
        assert_eq!(odd[1..].iter().map(|b| b[2] * b[3]).sum::<usize>(), 35);
    }

    #[test]
    fn blocks_os_and_speed_keys() {
        for vk in [0x5b, 0x5c, 0x12, 0xc0, 0x7b, 0xbb, 0xbd, 0x1b, 0x20] {
            assert!(
                !valid_event(&Event::Key { vk, down: true }, false),
                "vk {vk:#x} should be refused in a match"
            );
        }
        for vk in [0x09, 0x0d, 0x30, 0x39] {
            assert!(
                valid_event(&Event::Key { vk, down: true }, false),
                "vk {vk:#x} is part of the demonstration vocabulary"
            );
        }
    }
    #[test]
    fn arena_lines_skip_other_and_partial_lines() {
        let text =
            b"[1][x][effectbase.cpp:1783]: ARENA week  1:00, 4 January, 1936 BLU states 4\r\n\
[2][x][other.cpp:9]: something else\r\n[3][x][effectbase.cpp:1783]: ARENA capitu";
        let (lines, used) = arena_lines(text);
        assert_eq!(lines, vec!["week  1:00, 4 January, 1936 BLU states 4"]);
        assert_eq!(&text[used..], b"[3][x][effectbase.cpp:1783]: ARENA capitu");
    }
    #[test]
    fn setup_can_open_the_console() {
        let grave = Event::Key {
            vk: 0xc0,
            down: true,
        };
        assert!(valid_event(&grave, true));
        assert!(!valid_event(&grave, false));
    }
    #[test]
    fn coordinates_are_finite_and_bounded() {
        assert!(!valid_event(&Event::Move { x: f64::NAN, y: 0. }, false));
        assert!(!valid_event(&Event::Move { x: 1.1, y: 0. }, false));
        assert!(valid_event(&Event::Move { x: 0.5, y: 1. }, false));
    }
    #[test]
    fn supports_held_inputs() {
        let e = Event::Button {
            button: 0,
            down: true,
        };
        assert_eq!(
            e,
            serde_json::from_str(&serde_json::to_string(&e).unwrap()).unwrap()
        );
        assert!(valid_event(&e, false));
    }
}
