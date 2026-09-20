use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    Move { x: f64, y: f64 },
    Button { button: u8, down: bool },
    Key { vk: u16, down: bool },
    Wheel { delta: i32 },
}

fn valid_event(e: &Event, setup: bool) -> bool {
    match *e {
        Event::Move { x, y } => {
            x.is_finite() && y.is_finite() && (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y)
        }
        Event::Button { button, .. } => button < 3,
        Event::Wheel { delta } => delta.unsigned_abs() <= 1200,
        Event::Key { vk, .. } => {
            // Never allow OS keys, Alt, console, F12 (emergency stop), or speed changes in matches.
            matches!(vk, 0x10 | 0x11 | 0x25..=0x28 | 0x41..=0x5a)
                || (setup
                    && matches!(
                        vk,
                        0x08 | 0x09 | 0x0d | 0x1b | 0x20 | 0x30..=0x39 | 0xbb | 0xbd
                    ))
        }
    }
}

#[cfg(windows)]
mod platform {
    use super::*;
    use std::{
        collections::BTreeSet,
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
    fn ns() -> u64 {
        ORIGIN.get_or_init(Instant::now).elapsed().as_nanos() as u64
    }
    fn foreground() -> bool {
        unsafe {
            GetForegroundWindow() as usize == TARGET.load(Ordering::Relaxed)
                && TARGET.load(Ordering::Relaxed) != 0
        }
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
    unsafe fn capture(hwnd: HWND) -> Result<(Vec<u8>, i32, i32, u64, u64), String> {
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
        Ok((bytes, w, h, start, ns()))
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
    pub fn run() -> Result<(), String> {
        unsafe {
            SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        }
        ORIGIN.get_or_init(Instant::now);
        hooks()?;
        let (tx, rx) = mpsc::sync_channel(8);
        thread::spawn(move || {
            for line in io::stdin().lock().lines() {
                match line {
                    Ok(v) => {
                        if tx.send(v).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
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
        let mut seq = 0u64;
        let mut out = io::BufWriter::new(io::stdout().lock());
        loop {
            let line = match rx.recv_timeout(Duration::from_millis(10)) {
                Ok(v) => v,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(_) => break,
            };
            let result = (|| -> Result<(serde_json::Value, Vec<u8>), String> {
                let mut state = shared.lock().map_err(|_| "input_lock")?;
                let InputState {
                    held,
                    armed,
                    setup,
                    last,
                } = &mut *state;
                let cmd: serde_json::Value =
                    serde_json::from_str(&line).map_err(|e| e.to_string())?;
                match cmd["op"].as_str().unwrap_or("") {
                    "attach" => {
                        held.release();
                        *armed = false;
                        let hwnd = unsafe { select()? };
                        held.hwnd = hwnd as usize;
                        TARGET.store(hwnd as usize, Ordering::Relaxed);
                        Ok((
                            serde_json::json!({"hwnd":hwnd as usize,"foreground":foreground(),"clock_ns":ns(),"backend":"gdi_bgra","computer":std::env::var("COMPUTERNAME").unwrap_or_default()}),
                            vec![],
                        ))
                    }
                    "arm" => {
                        if !foreground() {
                            return Err("game_not_foreground".into());
                        }
                        *setup = cmd["mode"] == "setup";
                        STOP.store(false, Ordering::SeqCst);
                        *armed = true;
                        *last = Instant::now();
                        Ok((serde_json::json!({"armed":true}), vec![]))
                    }
                    "release" => {
                        held.release();
                        *armed = false;
                        Ok((serde_json::json!({"armed":false}), vec![]))
                    }
                    "capture" => {
                        let (bytes, w, h, start, end) = unsafe { capture(held.hwnd as HWND)? };
                        let encoding = if cmd["encoding"] == "lz4" {
                            "lz4"
                        } else {
                            "raw"
                        };
                        let bytes = if encoding == "lz4" {
                            lz4_flex::block::compress(&bytes)
                        } else {
                            bytes
                        };
                        seq += 1;
                        let events = std::mem::take(&mut *EVENTS.lock().map_err(|_| "event_lock")?);
                        Ok((
                            serde_json::json!({"seq":seq,"width":w,"height":h,"encoding":encoding,"capture_start_ns":start,"t_ns":end,"events":events,"overflow":OVERFLOW.swap(false,Ordering::Relaxed),"stopped":STOP.load(Ordering::SeqCst),"foreground":foreground()}),
                            bytes,
                        ))
                    }
                    "events" => {
                        let events = std::mem::take(&mut *EVENTS.lock().map_err(|_| "event_lock")?);
                        Ok((
                            serde_json::json!({"events":events,"t_ns":ns(),"overflow":OVERFLOW.swap(false,Ordering::Relaxed)}),
                            vec![],
                        ))
                    }
                    "apply" => {
                        if !*armed || STOP.load(Ordering::SeqCst) || !foreground() {
                            return Err("input_not_armed_or_focus_lost".into());
                        }
                        let events: Vec<Event> = serde_json::from_value(cmd["events"].clone())
                            .map_err(|e| e.to_string())?;
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
                            serde_json::json!({"applied":events.len(),"t_ns":ns()}),
                            vec![],
                        ))
                    }
                    "status" => Ok((
                        serde_json::json!({"armed":*armed,"foreground":foreground(),"stopped":STOP.load(Ordering::SeqCst),"held_keys":held.keys,"held_buttons":held.buttons,"t_ns":ns()}),
                        vec![],
                    )),
                    _ => Err("unknown_operation".into()),
                }
            })();
            let (mut response, bytes) = match result {
                Ok(v) => v,
                Err(e) => {
                    let mut state = shared.lock().unwrap_or_else(|e| e.into_inner());
                    state.held.release();
                    state.armed = false;
                    (serde_json::json!({"error":e}), vec![])
                }
            };
            response["bytes"] = serde_json::json!(bytes.len());
            writeln!(out, "{}", response).map_err(|e| e.to_string())?;
            out.write_all(&bytes).map_err(|e| e.to_string())?;
            out.flush().map_err(|e| e.to_string())?;
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
    #[test]
    fn blocks_os_and_speed_keys() {
        for vk in [0x5b, 0x5c, 0x12, 0xc0, 0x7b, 0x20, 0xbb, 0xbd] {
            assert!(!valid_event(&Event::Key { vk, down: true }, false));
        }
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
