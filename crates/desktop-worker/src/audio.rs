//! A game's own sound, for the live view.
//!
//! Windows can capture what one process plays (process loopback, Windows 10 2004 and
//! later) apart from everything else on the PC, so a view carries the game's sound and
//! nothing of whoever uses the PC. The sound goes to the view's ffmpeg as raw PCM on its
//! stdin, paced by the clock: a game that plays nothing still gives silence, so the
//! stream's sound never stops and never drifts from its picture.

#![cfg(windows)]

use std::collections::VecDeque;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Frames a second, and channels of 16-bit samples: what ffmpeg is told to read.
pub const RATE: u64 = 48_000;
pub const CHANNELS: usize = 2;
const BYTES_PER_FRAME: usize = CHANNELS * 2;
/// Captured sound waiting longer than this loses its oldest part, so the sound stays
/// within a quarter of a second of the picture.
const MOST_BEHIND: usize = (RATE / 4) as usize;
const TICK: Duration = Duration::from_millis(10);

/// ffmpeg's arguments for the sound on its stdin, as `audio_input` reads it.
pub fn input_arguments() -> Vec<String> {
    let (rate, channels) = (RATE.to_string(), CHANNELS.to_string());
    [
        "-f",
        "s16le",
        "-ar",
        &rate,
        "-ac",
        &channels,
        "-thread_queue_size",
        "1024",
        "-i",
        "pipe:0",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// How many frames are due `elapsed` after the start, `written` already being out.
pub fn due(elapsed: Duration, written: u64) -> u64 {
    let total = elapsed.as_nanos() * RATE as u128 / 1_000_000_000;
    (total as u64).saturating_sub(written)
}

/// The next `frames` frames of sound: the oldest captured, then silence for what the
/// game did not play. Captured sound too far behind loses its oldest part first.
pub fn take(captured: &mut VecDeque<u8>, frames: usize) -> Vec<u8> {
    let most = (MOST_BEHIND + frames) * BYTES_PER_FRAME;
    if captured.len() > most {
        let extra = captured.len() - most;
        captured.drain(..extra - extra % BYTES_PER_FRAME);
    }
    let mut out = vec![0u8; frames * BYTES_PER_FRAME];
    let have = captured.len().min(out.len());
    for (slot, byte) in out.iter_mut().zip(captured.drain(..have)) {
        *slot = byte;
    }
    out
}

/// A process's sound, captured and, once `attach`ed to its ffmpeg, written there in real
/// time until dropped, or until ffmpeg stops taking it.
pub struct Capture {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    to: Option<mpsc::Sender<Box<dyn Write + Send>>>,
}

impl Capture {
    /// Capture what `pid` and its children play. Fails, so the view goes without sound,
    /// when Windows will not capture it (no sound device, a Windows before 10 2004).
    pub fn start(pid: u32) -> Result<Self, String> {
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = Arc::clone(&stop);
        let (ready, started) = mpsc::channel();
        let (to, from) = mpsc::channel();
        let thread = thread::Builder::new()
            .name("view-audio".into())
            .spawn(move || run(pid, from, &stopping, ready))
            .map_err(|e| format!("audio_thread_failed: {e}"))?;
        match started.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => Ok(Self {
                stop,
                thread: Some(thread),
                to: Some(to),
            }),
            Ok(Err(why)) => Err(why),
            Err(_) => {
                stop.store(true, Ordering::SeqCst);
                Err("audio_start_timeout".into())
            }
        }
    }
}

impl Capture {
    /// Where the sound goes from now on: the view's ffmpeg's stdin. The clock starts here.
    pub fn attach(&mut self, out: impl Write + Send + 'static) {
        if let Some(to) = self.to.take() {
            let _ = to.send(Box::new(out));
        }
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        self.to = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn run(
    pid: u32,
    out: mpsc::Receiver<Box<dyn Write + Send>>,
    stop: &AtomicBool,
    ready: mpsc::Sender<Result<(), String>>,
) {
    let _ = wasapi::initialize_mta();
    let client = match open(pid) {
        Ok(client) => client,
        Err(why) => {
            let _ = ready.send(Err(why));
            return;
        }
    };
    let _ = ready.send(Ok(()));
    let (audio, mut capture) = client;
    // Dropped before attached (ffmpeg failed to start), the sender goes and this ends.
    let Ok(mut out) = out.recv() else {
        let _ = audio.stop_stream();
        return;
    };
    let mut captured = VecDeque::new();
    let mut scratch = vec![0u8; RATE as usize * BYTES_PER_FRAME];
    let (start, mut written) = (Instant::now(), 0u64);
    while !stop.load(Ordering::SeqCst) {
        // A capture that fails (the device went away) leaves silence, not a stopped stream.
        if let Some(reader) = &capture {
            if read(reader, &mut captured, &mut scratch).is_err() {
                capture = None;
            }
        }
        let frames = due(start.elapsed(), written);
        if frames > 0 {
            if out
                .write_all(&take(&mut captured, frames as usize))
                .is_err()
            {
                break;
            }
            written += frames;
        }
        thread::sleep(TICK);
    }
    let _ = audio.stop_stream();
}

type Opened = (wasapi::AudioClient, Option<wasapi::AudioCaptureClient>);

fn open(pid: u32) -> Result<Opened, String> {
    fn fail(what: &'static str) -> impl Fn(wasapi::WasapiError) -> String {
        move |e| format!("audio_{what}: {e}")
    }
    let mut audio = wasapi::AudioClient::new_application_loopback_client(pid, true)
        .map_err(fail("activate"))?;
    let format = wasapi::WaveFormat::new(
        16,
        16,
        &wasapi::SampleType::Int,
        RATE as usize,
        CHANNELS,
        None,
    );
    let mode = wasapi::StreamMode::PollingShared {
        autoconvert: true,
        buffer_duration_hns: 2_000_000,
    };
    audio
        .initialize_client(&format, &wasapi::Direction::Capture, &mode)
        .map_err(fail("initialize"))?;
    let capture = audio.get_audiocaptureclient().map_err(fail("client"))?;
    audio.start_stream().map_err(fail("start"))?;
    Ok((audio, Some(capture)))
}

/// Everything captured since the last read, silent packets as zeros.
fn read(
    reader: &wasapi::AudioCaptureClient,
    captured: &mut VecDeque<u8>,
    scratch: &mut [u8],
) -> Result<(), wasapi::WasapiError> {
    while reader.get_next_packet_size()?.unwrap_or(0) > 0 {
        let (frames, info) = reader.read_from_device(scratch)?;
        let bytes = &mut scratch[..frames as usize * BYTES_PER_FRAME];
        if info.flags.silent {
            bytes.fill(0);
        }
        captured.extend(bytes.iter());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_is_due_follows_the_clock() {
        assert_eq!(due(Duration::from_millis(10), 0), 480);
        assert_eq!(due(Duration::from_secs(1), 47_000), 1_000);
        assert_eq!(due(Duration::from_millis(5), 480), 0);
    }

    #[test]
    fn silence_fills_what_the_game_did_not_play() {
        let mut captured: VecDeque<u8> = [1u8; 8].into_iter().collect();
        let out = take(&mut captured, 4);
        assert_eq!(&out[..8], &[1u8; 8]);
        assert_eq!(&out[8..], &[0u8; 8]);
        assert!(captured.is_empty());
    }

    #[test]
    fn sound_far_behind_loses_its_oldest_part() {
        let frames = MOST_BEHIND + 100;
        let mut captured: VecDeque<u8> = (0..frames * BYTES_PER_FRAME)
            .map(|i| (i / BYTES_PER_FRAME >= 90) as u8)
            .collect();
        let out = take(&mut captured, 10);
        // The 90 oldest frames (zeros here) were dropped; what came out is the newer part.
        assert!(out.iter().all(|&b| b == 1));
        assert_eq!(captured.len(), MOST_BEHIND * BYTES_PER_FRAME);
    }
}

#[cfg(test)]
mod live {
    use super::*;
    use std::process::{Command, Stdio};
    use std::sync::Mutex;

    struct Shared(Arc<Mutex<Vec<u8>>>);

    impl Write for Shared {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Needs a sound device and ffplay on the PATH: a tone at 1% volume, captured from
    /// its process alone.
    #[test]
    #[ignore]
    fn a_process_s_sound_is_captured_and_paced() {
        let mut player = Command::new("ffplay")
            .args(["-nodisp", "-autoexit", "-loglevel", "quiet", "-volume", "1"])
            .args(["-f", "lavfi", "-i", "sine=frequency=440:duration=4"])
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        thread::sleep(Duration::from_millis(500));
        let got = Arc::new(Mutex::new(Vec::new()));
        let mut capture = Capture::start(player.id()).unwrap();
        capture.attach(Shared(Arc::clone(&got)));
        thread::sleep(Duration::from_secs(2));
        drop(capture);
        let _ = player.kill();
        let _ = player.wait();
        let bytes = got.lock().unwrap().clone();
        let frames = bytes.len() / BYTES_PER_FRAME;
        let samples: Vec<i16> = bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        let loudest = samples.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
        println!("{frames} frames in 2 s, loudest {loudest}");
        assert!((90_000..=100_000).contains(&frames), "{frames}");
        assert!(loudest > 0);
    }
}
