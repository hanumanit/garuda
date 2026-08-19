//! Streaming a model's blocks off disk without going through the page cache.
//!
//! For a checkpoint larger than RAM, the page cache is the wrong tool twice over. It
//! decides what to keep by recency, and this workload has none — every block is read
//! exactly once per token, in the same order, so nothing is hotter than anything else.
//! And the bytes it caches on the way past are the very bytes that evict whatever the
//! engine *did* want to keep resident. Measured on Qwen3.8-27B: with everything mapped,
//! the cache settles at ~4 GB of a 12 GB model and a forward pass reads the other 8 GB
//! from disk at 1.2-2.4 GB/s, against a drive that gives 3.9.
//!
//! So the blocks the engine has decided not to keep are read explicitly instead, with
//! `F_NOCACHE` set on the descriptor: large sequential reads into a small ring of
//! buffers this process owns, leaving the cache untouched for the resident half.
//!
//! # The shape of it
//!
//! A pass walks blocks in order, so the ring only needs to be a few deep. Block `l`
//! always lands in slot `l % slots`, which makes the protocol small enough to reason
//! about: a reader thread fills a slot while holding its lock, and the forward pass
//! borrows that slot for as long as it computes the block. A reader that gets too far
//! ahead simply blocks on the slot the pass is still reading, which is the backpressure
//! this needs and costs nothing to arrange.
//!
//! Asking for a block nobody has fetched is not an error: [`BlockStreamer::borrow`]
//! reads it inline. Slower and always correct, which is the right way round.

use crate::core::GarudaError;
use parking_lot::{Condvar, Mutex, MutexGuard};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// One buffer in the ring.
struct SlotState {
    /// Which block these bytes are, once `ready`.
    block: Option<usize>,
    ready: bool,
    buf: Vec<u8>,
}

struct Slot {
    st: Mutex<SlotState>,
    cv: Condvar,
}

/// Reads a checkpoint's blocks into a ring of buffers, ahead of the pass that needs
/// them, bypassing the page cache.
pub struct BlockStreamer {
    file: File,
    /// `spans[l]` is block `l`'s `(start, len)` in the file. An empty span means the
    /// block is resident already and is never streamed.
    spans: Vec<(usize, usize)>,
    slots: Vec<Slot>,
    hits: AtomicU64,
    inline: AtomicU64,
}

impl BlockStreamer {
    /// Buffers in the ring, which is also how many blocks may be read at once.
    ///
    /// Three is what the same measurement said for the page-cache prefetcher: one
    /// reader cannot keep a request queued at the drive, and past three the drive is
    /// already busy.
    pub const SLOTS: usize = 3;

    /// Open `path` for streaming. Fails if the file cannot be opened; `F_NOCACHE` is
    /// best-effort, since a kernel that declines it costs speed and not correctness.
    pub fn open(path: &std::path::Path, spans: Vec<(usize, usize)>) -> Result<Self, GarudaError> {
        let file = File::open(path).map_err(|e| GarudaError::Io(e.to_string()))?;
        // Keep these bytes out of the page cache: they are read once per token and
        // never read back from it, and caching them evicts the blocks the engine is
        // deliberately holding resident.
        unsafe {
            if libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) == -1 {
                tracing::debug!("F_NOCACHE declined; streamed blocks will be cached");
            }
        }
        let widest = spans.iter().map(|&(_, len)| len).max().unwrap_or(0);
        let slots = (0..Self::SLOTS)
            .map(|_| Slot {
                st: Mutex::new(SlotState {
                    block: None,
                    ready: false,
                    // Sized to the widest block once, so a pass never allocates.
                    buf: Vec::with_capacity(widest),
                }),
                cv: Condvar::new(),
            })
            .collect();
        Ok(Self {
            file,
            spans,
            slots,
            hits: AtomicU64::new(0),
            inline: AtomicU64::new(0),
        })
    }

    /// True when this block is streamed rather than held resident.
    pub fn streams(&self, block: usize) -> bool {
        self.spans.get(block).is_some_and(|&(_, len)| len > 0)
    }

    /// Where block `l` starts in the file, which is what turns a weight's file offset
    /// into an offset within the borrowed buffer.
    pub fn base(&self, block: usize) -> usize {
        self.spans.get(block).map_or(0, |&(start, _)| start)
    }

    /// Read block `l` into its slot, if it is not already there.
    ///
    /// Called from a background thread ahead of the pass. Blocks while the pass is
    /// borrowing that slot, which is the backpressure that keeps a reader from
    /// overwriting bytes still in use.
    pub fn fetch(&self, l: usize) {
        if !self.streams(l) {
            return;
        }
        let slot = &self.slots[l % self.slots.len()];
        let mut st = slot.st.lock();
        if st.block == Some(l) && st.ready {
            return;
        }
        self.fill(l, &mut st);
        slot.cv.notify_all();
    }

    /// The bytes of block `l`, for as long as the returned guard is held.
    pub fn borrow(&self, l: usize) -> Guard<'_> {
        let slot = &self.slots[l % self.slots.len()];
        let mut st = slot.st.lock();
        if st.block == Some(l) && st.ready {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            // Whoever held the lock was filling some other block; a reader never runs
            // more than a ring ahead of the pass, so that block is one already used.
            self.inline.fetch_add(1, Ordering::Relaxed);
            self.fill(l, &mut st);
        }
        Guard { st }
    }

    /// Blocks served from a slot a reader had already filled, and blocks the pass had
    /// to read itself. A high inline count means the readers are not keeping up.
    pub fn stats(&self) -> (u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.inline.load(Ordering::Relaxed),
        )
    }

    fn fill(&self, l: usize, st: &mut SlotState) {
        let Some(&(start, len)) = self.spans.get(l) else {
            return;
        };
        st.ready = false;
        st.block = None;
        st.buf.resize(len, 0);
        let mut at = 0usize;
        while at < len {
            match self.file.read_at(&mut st.buf[at..], (start + at) as u64) {
                Ok(0) => break,
                Ok(n) => at += n,
                Err(e) => {
                    tracing::warn!(block = l, error = %e, "streaming read failed");
                    return;
                }
            }
        }
        if at == len {
            st.block = Some(l);
            st.ready = true;
        }
    }
}

/// A borrowed block. The slot stays locked — and so unwritable by the readers — until
/// this is dropped, which is what makes the bytes safe to compute against.
pub struct Guard<'a> {
    st: MutexGuard<'a, SlotState>,
}

impl Guard<'_> {
    pub fn bytes(&self) -> &[u8] {
        &self.st.buf
    }
}

/// Runs a [`BlockStreamer`]'s reads on background threads.
pub struct StreamPrefetcher {
    tx: std::sync::mpsc::SyncSender<usize>,
    depth: usize,
}

impl StreamPrefetcher {
    pub fn new(streamer: Arc<BlockStreamer>, threads: usize) -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel::<usize>(threads.max(1));
        let rx = Arc::new(Mutex::new(rx));
        for t in 0..threads.max(1) {
            let (rx, streamer) = (rx.clone(), streamer.clone());
            std::thread::Builder::new()
                .name(format!("garuda-stream-{t}"))
                .spawn(move || {
                    loop {
                        let block = match rx.lock().recv() {
                            Ok(b) => b,
                            Err(_) => return,
                        };
                        streamer.fetch(block);
                    }
                })
                .expect("spawning a streaming thread");
        }
        Self {
            tx,
            // One less than the ring: the pass is holding one slot itself.
            depth: BlockStreamer::SLOTS.saturating_sub(1).max(1),
        }
    }

    /// Ask for the blocks after `l`. Never blocks: a request that cannot be taken up
    /// now is for a block the pass is about to reach anyway.
    pub fn hint_ahead(&self, l: usize) {
        for ahead in 1..=self.depth {
            let _ = self.tx.try_send(l + ahead);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn file_of(bytes: &[u8], name: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("garuda_stream_{name}"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("weights.bin");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(bytes)
            .unwrap();
        (dir, path)
    }

    /// What a streamed block must be: exactly the bytes at that offset in the file,
    /// however it was fetched — ahead of time or inline.
    #[test]
    fn a_streamed_block_is_the_bytes_the_file_holds() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
        let (dir, path) = file_of(&data, "bytes");

        let spans = vec![(0, 1024), (1024, 1024), (2048, 1024), (3072, 1024)];
        let s = BlockStreamer::open(&path, spans).unwrap();

        // Fetched ahead, then borrowed.
        s.fetch(1);
        assert_eq!(s.borrow(1).bytes(), &data[1024..2048]);

        // Never fetched: read inline, same answer.
        assert_eq!(s.borrow(2).bytes(), &data[2048..3072]);

        // The ring wraps, and a slot that held an older block is refilled.
        s.fetch(3);
        assert_eq!(s.borrow(3).bytes(), &data[3072..4096]);
        assert_eq!(s.borrow(0).bytes(), &data[0..1024]);

        let (hits, inline) = s.stats();
        assert_eq!(hits + inline, 4, "every borrow is accounted for");
        assert!(inline >= 1, "the block nobody fetched was read inline");

        // A block the engine holds resident is not streamed at all.
        let resident = BlockStreamer::open(&path, vec![(0, 0), (1024, 1024)]).unwrap();
        assert!(!resident.streams(0));
        assert!(resident.streams(1));
        assert_eq!(resident.base(1), 1024);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The readers and the pass share the ring without stepping on each other: the
    /// bytes a borrow returns are still the right ones after several laps.
    #[test]
    fn readers_running_ahead_do_not_disturb_a_borrowed_block() {
        let data: Vec<u8> = (0..8192).map(|i| (i % 251) as u8).collect();
        let (dir, path) = file_of(&data, "ring");
        let spans: Vec<(usize, usize)> = (0..8).map(|l| (l * 1024, 1024)).collect();
        let streamer = Arc::new(BlockStreamer::open(&path, spans).unwrap());
        let pf = StreamPrefetcher::new(streamer.clone(), 2);

        for _ in 0..3 {
            for l in 0..8 {
                pf.hint_ahead(l);
                let g = streamer.borrow(l);
                assert_eq!(
                    g.bytes(),
                    &data[l * 1024..(l + 1) * 1024],
                    "block {l} came back wrong"
                );
                std::thread::yield_now();
            }
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
