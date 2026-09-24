//! A recording encoded where it is captured.
//!
//! Before, every frame crossed to the recorder as lz4 BGRA, 8.3 MB raw at 1080p, and was
//! encoded there. Over the network that was the tick: a request, a capture, a transfer and
//! a reply every 200 ms, and anything else on the connection (the camera's own screenshots)
//! pushed frames late. Here the worker owns the clock and the encoder: ffmpeg, started by
//! the worker on the same PC, takes raw frames on its stdin and gives the encoded stream on
//! its stdout, which the worker forwards. Only the video crosses the network.
//!
//! What ffmpeg may be asked to do is fixed here: a few named encoder profiles, each with one
//! quality number, reading a pipe and writing a pipe. Nothing from a request reaches its
//! command line except that name and number, both checked, so the worker still exposes no
//! command and writes no file.

#![cfg(windows)]

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::windows::io::AsRawHandle;
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};
use windows_sys::Win32::System::Threading::CREATE_NO_WINDOW;

/// Frames that may wait for the encoder. At 5 Hz this is a second and a half: far more
/// than a hardware encoder ever needs, so a full queue means ffmpeg has stalled and the
/// tick is dropped (and reported) rather than blocking the clock.
pub const QUEUE: usize = 8;

/// A named encoder: its ffmpeg arguments for a quality, and the quality range.
pub struct Profile {
    pub name: &'static str,
    /// Lowest, default and highest quality number: a QP or CRF, lower is better.
    pub quality: (u32, u32, u32),
    /// Uses the GPU's video encoder, so it may be unavailable on a PC without one.
    pub hardware: bool,
}

pub const PROFILES: [Profile; 4] = [
    // NVENC H.264 in full-resolution colour (High 4:4:4), the slowest preset, constant QP,
    // no B-frames. At QP 14 it keeps more of every frame than x264 at CRF 18 (PSNR 45.47
    // against 44.83 dB, templates scoring 4x closer to lossless) at 2% more bytes, and the
    // training reader decodes it 6% faster (scripts/codec_fidelity.py, second PC, 2026-09-24).
    Profile {
        name: "h264_nvenc",
        quality: (0, 14, 51),
        hardware: true,
    },
    // NVENC HEVC, 4:4:4 (Range Extensions): smaller, but slower to decode.
    Profile {
        name: "hevc_nvenc",
        quality: (0, 16, 51),
        hardware: true,
    },
    // The recordings' own encoder until 2026-09-24, run beside the game instead: CRF 18,
    // 4:4:4, preset faster. Four threads, so it cannot take the game's cores.
    Profile {
        name: "x264",
        quality: (0, 18, 51),
        hardware: false,
    },
    // Lossless, for reference captures that the others are measured against.
    Profile {
        name: "ffv1",
        quality: (0, 0, 0),
        hardware: false,
    },
];

pub fn profile(name: &str) -> Option<&'static Profile> {
    PROFILES.iter().find(|p| p.name == name)
}

/// The encoder arguments of `name` at `quality` (its default when None), for frames
/// arriving `hz` times a second, and the quality used.
pub fn arguments(name: &str, quality: Option<u32>, hz: u32) -> Result<(Vec<String>, u32), String> {
    let profile = profile(name).ok_or("unknown_encoder_profile")?;
    let (low, default, high) = profile.quality;
    let q = quality.unwrap_or(default);
    if !(low..=high).contains(&q) {
        return Err("encoder_quality_out_of_range".into());
    }
    // A keyframe every 20 s: a damaged file loses at most that much, for about 5% more bits.
    let gop = (20 * hz).to_string();
    let q_text = q.to_string();
    let args: Vec<&str> = match name {
        "h264_nvenc" => vec![
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p7",
            "-tune",
            "hq",
            "-profile:v",
            "high444p",
            "-pix_fmt",
            "yuv444p",
            "-rc",
            "constqp",
            "-qp",
            &q_text,
            "-bf",
            "0",
            // Out as soon as encoded, not after a queue of frames.
            "-delay",
            "0",
            "-g",
            &gop,
        ],
        "hevc_nvenc" => vec![
            "-c:v",
            "hevc_nvenc",
            "-preset",
            "p5",
            "-tune",
            "hq",
            "-profile:v",
            "rext",
            "-pix_fmt",
            "yuv444p",
            "-rc",
            "constqp",
            "-qp",
            &q_text,
            "-bf",
            "0",
            "-delay",
            "0",
            "-g",
            &gop,
        ],
        "x264" => vec![
            "-c:v", "libx264", "-preset", "faster", "-crf", &q_text, "-pix_fmt", "yuv444p",
            "-threads", "4", "-g", &gop,
        ],
        "ffv1" => vec![
            "-c:v", "ffv1", "-level", "3", "-slices", "16", "-threads", "8", "-g", &gop,
        ],
        _ => return Err("unknown_encoder_profile".into()),
    };
    Ok((args.into_iter().map(String::from).collect(), q))
}

/// ffmpeg for the encoder: beside the worker, in the compute tools the second PC's
/// deploy puts there, or on PATH.
pub fn find_ffmpeg(exe_dir: &Path, path: Option<&std::ffi::OsStr>) -> Option<PathBuf> {
    [
        exe_dir.join("ffmpeg.exe"),
        exe_dir.join("compute").join("tools").join("ffmpeg.exe"),
    ]
    .into_iter()
    .find(|p| p.is_file())
    .or_else(|| crate::find_on_path("ffmpeg.exe", path))
}

/// Why a frame was not queued.
pub enum Push {
    /// The encoder is behind: the frame would wait too long.
    Full,
    /// The encoder has exited.
    Closed,
}

/// How an encoder ended.
pub struct Finish {
    pub exit: Option<i32>,
    pub frames: u64,
    pub bytes: u64,
    pub errors: Vec<String>,
}

/// A kill-on-close job: the encoder cannot outlive the worker that feeds it.
struct Job(HANDLE);

impl Drop for Job {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.0);
        }
    }
}

// The job handle is used only to be closed.
unsafe impl Send for Job {}

fn kill_on_close(child: &Child) -> Option<Job> {
    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            return None;
        }
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let ok = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &limits as *const _ as *const _,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        ) != 0
            && AssignProcessToJobObject(job, child.as_raw_handle() as HANDLE) != 0;
        if ok {
            Some(Job(job))
        } else {
            CloseHandle(job);
            None
        }
    }
}

pub struct Encoder {
    pub pid: u32,
    frames: Option<SyncSender<Arc<Vec<u8>>>>,
    writer: Option<JoinHandle<u64>>,
    reader: Option<JoinHandle<u64>>,
    child: Child,
    errors: Arc<Mutex<VecDeque<String>>>,
    queued: Arc<std::sync::atomic::AtomicUsize>,
    _job: Option<Job>,
}

impl Encoder {
    /// Start ffmpeg reading `width` x `height` BGRA frames and writing NUT to its stdout,
    /// which `on_data` receives chunk by chunk, on a thread of its own, in order.
    pub fn start(
        ffmpeg: &Path,
        width: usize,
        height: usize,
        hz: u32,
        args: &[String],
        mut on_data: impl FnMut(&[u8]) + Send + 'static,
    ) -> Result<Self, String> {
        let mut child = Command::new(ffmpeg)
            .args(["-hide_banner", "-loglevel", "error", "-nostats"])
            .args(["-f", "rawvideo", "-pixel_format", "bgra"])
            .args(["-video_size", &format!("{width}x{height}")])
            .args(["-framerate", &hz.to_string(), "-i", "pipe:0", "-an"])
            .args(args)
            // NUT writes each packet as it comes, with its timestamp, for any codec; the
            // recorder remuxes it into the Matroska file training reads.
            .args(["-f", "nut", "-flush_packets", "1", "pipe:1"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()
            .map_err(|e| format!("encoder_start_failed: {e}"))?;
        let job = kill_on_close(&child);
        let pid = child.id();
        let mut stdin = child.stdin.take().ok_or("encoder_stdin")?;
        let mut stdout = child.stdout.take().ok_or("encoder_stdout")?;
        let stderr = child.stderr.take().ok_or("encoder_stderr")?;
        let errors = Arc::new(Mutex::new(VecDeque::new()));
        let queued = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (frames, incoming) = mpsc::sync_channel::<Arc<Vec<u8>>>(QUEUE);
        let pending = Arc::clone(&queued);
        let writer = thread::Builder::new()
            .name("encoder-in".into())
            .spawn(move || {
                let mut written = 0u64;
                for frame in incoming {
                    pending.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                    if stdin.write_all(&frame).is_err() {
                        break;
                    }
                    written += 1;
                }
                // Dropping stdin is the end of the input: ffmpeg flushes and exits.
                drop(stdin);
                written
            })
            .map_err(|e| format!("encoder_thread_failed: {e}"))?;
        let reader = thread::Builder::new()
            .name("encoder-out".into())
            .spawn(move || {
                let mut total = 0u64;
                let mut buffer = vec![0u8; 1 << 18];
                while let Ok(n @ 1..) = stdout.read(&mut buffer) {
                    total += n as u64;
                    on_data(&buffer[..n]);
                }
                total
            })
            .map_err(|e| format!("encoder_thread_failed: {e}"))?;
        let sink = Arc::clone(&errors);
        thread::Builder::new()
            .name("encoder-err".into())
            .spawn(move || {
                for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                    if let Ok(mut errors) = sink.lock() {
                        if errors.len() >= 32 {
                            errors.pop_front();
                        }
                        errors.push_back(line);
                    }
                }
            })
            .map_err(|e| format!("encoder_thread_failed: {e}"))?;
        Ok(Self {
            pid,
            frames: Some(frames),
            writer: Some(writer),
            reader: Some(reader),
            child,
            errors,
            queued,
            _job: job,
        })
    }

    /// Queue one BGRA frame, without waiting.
    pub fn push(&self, frame: Arc<Vec<u8>>) -> Result<(), Push> {
        let sender = self.frames.as_ref().ok_or(Push::Closed)?;
        self.queued
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        match sender.try_send(frame) {
            Ok(()) => Ok(()),
            Err(error) => {
                self.queued
                    .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                Err(match error {
                    TrySendError::Full(_) => Push::Full,
                    TrySendError::Disconnected(_) => Push::Closed,
                })
            }
        }
    }

    /// Frames waiting for ffmpeg.
    pub fn queued(&self) -> usize {
        self.queued.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// The encoder's last complaints, if any.
    pub fn errors(&self) -> Vec<String> {
        self.errors
            .lock()
            .map(|e| e.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// End the input, let ffmpeg write out everything it holds, and wait for it: `limit`
    /// at most, after which it is stopped. Every byte it wrote has been handed to
    /// `on_data` by the time this returns.
    pub fn finish(mut self, limit: Duration) -> Finish {
        drop(self.frames.take());
        let frames = self.writer.take().map_or(0, |w| w.join().unwrap_or(0));
        let deadline = Instant::now() + limit;
        let exit = loop {
            match self.child.try_wait() {
                Ok(Some(status)) => break status.code(),
                Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(20)),
                _ => {
                    let _ = self.child.kill();
                    break self.child.wait().ok().and_then(|s| s.code());
                }
            }
        };
        let bytes = self.reader.take().map_or(0, |r| r.join().unwrap_or(0));
        // The stderr reader ends with the process; give it a moment to catch the last line.
        thread::sleep(Duration::from_millis(50));
        let errors = self.errors();
        Finish {
            exit,
            frames,
            bytes,
            errors,
        }
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        // Not finished: a worker that is going away. Stop the encoder with it.
        if self.writer.is_some() {
            let _ = self.child.kill();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_take_only_a_checked_quality() {
        let (args, q) = arguments("h264_nvenc", None, 5).unwrap();
        assert_eq!(q, 14);
        let at = args.iter().position(|a| a == "-qp").unwrap();
        assert_eq!(args[at + 1], "14");
        assert!(args.windows(2).any(|w| w == ["-pix_fmt", "yuv444p"]));
        assert!(args.windows(2).any(|w| w == ["-g", "100"]));
        let (args, _) = arguments("x264", Some(18), 5).unwrap();
        assert!(args.windows(2).any(|w| w == ["-crf", "18"]));
        assert_eq!(
            arguments("x264", Some(52), 5).unwrap_err(),
            "encoder_quality_out_of_range"
        );
        assert_eq!(
            arguments("ffv1", Some(1), 5).unwrap_err(),
            "encoder_quality_out_of_range"
        );
        assert_eq!(
            arguments("-f image2 x.png", None, 5).unwrap_err(),
            "unknown_encoder_profile"
        );
        // No profile ever names an output: the output is always the pipe.
        for p in &PROFILES {
            let (args, _) = arguments(p.name, None, 5).unwrap();
            assert!(args.iter().all(|a| !a.contains(':') || a.starts_with("-")));
        }
    }

    /// A real ffmpeg, if there is one, encodes three frames and hands back a stream.
    #[test]
    fn an_encoder_turns_frames_into_a_stream() {
        let exe = std::env::current_exe().unwrap();
        let Some(ffmpeg) = find_ffmpeg(exe.parent().unwrap(), std::env::var_os("PATH").as_deref())
        else {
            eprintln!("no ffmpeg on this machine; skipped");
            return;
        };
        let (args, _) = arguments("ffv1", None, 5).unwrap();
        let received = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&received);
        let encoder = Encoder::start(&ffmpeg, 64, 32, 5, &args, move |chunk| {
            sink.lock().unwrap().extend_from_slice(chunk)
        })
        .unwrap();
        for i in 0..3u8 {
            let mut sent = false;
            for _ in 0..100 {
                if encoder.push(Arc::new(vec![i * 40; 64 * 32 * 4])).is_ok() {
                    sent = true;
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
            assert!(sent);
        }
        let finish = encoder.finish(Duration::from_secs(20));
        assert_eq!(finish.exit, Some(0), "{:?}", finish.errors);
        assert_eq!(finish.frames, 3);
        let bytes = received.lock().unwrap();
        assert_eq!(finish.bytes as usize, bytes.len());
        // NUT's file id string opens the stream.
        assert!(bytes.starts_with(b"nut/multimedia container"));
    }
}
