//! Local-disk implementation of [`StorageBackend`].
//!
//! Paths handed to this backend are joined onto `base_path` and must stay under
//! it: a caller-supplied `../../etc/passwd` is rejected rather than followed.

use crate::core::{GarudaError, StorageBackend};
use std::io::Write;
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Clone)]
pub struct LocalStorageBackend {
    base_path: PathBuf,
}

impl LocalStorageBackend {
    pub fn new<P: Into<PathBuf>>(base_path: P) -> Self {
        Self {
            base_path: base_path.into(),
        }
    }

    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Join `path` under the base, refusing anything that escapes it.
    fn resolve(&self, path: &Path) -> Result<PathBuf, GarudaError> {
        if path.is_absolute() {
            return Err(GarudaError::Storage(format!(
                "absolute path not allowed: {}",
                path.display()
            )));
        }
        if path
            .components()
            .any(|c| matches!(c, Component::ParentDir | Component::Prefix(_)))
        {
            return Err(GarudaError::Storage(format!(
                "path escapes storage root: {}",
                path.display()
            )));
        }
        Ok(self.base_path.join(path))
    }
}

impl StorageBackend for LocalStorageBackend {
    fn read(&self, path: &Path) -> Result<Vec<u8>, GarudaError> {
        let full = self.resolve(path)?;
        std::fs::read(&full)
            .map_err(|e| GarudaError::Storage(format!("read {}: {e}", full.display())))
    }

    fn write(&self, path: &Path, data: &[u8]) -> Result<(), GarudaError> {
        let full = self.resolve(path)?;
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| GarudaError::Storage(format!("create {}: {e}", parent.display())))?;
        }

        // Write to a temp file and rename, so a crash mid-write cannot leave a
        // half-written expert or KV block that a later read would happily parse.
        let tmp = full.with_extension("tmp");
        let mut f = std::fs::File::create(&tmp)
            .map_err(|e| GarudaError::Storage(format!("create {}: {e}", tmp.display())))?;
        f.write_all(data)
            .map_err(|e| GarudaError::Storage(format!("write {}: {e}", tmp.display())))?;
        f.sync_all()
            .map_err(|e| GarudaError::Storage(format!("sync {}: {e}", tmp.display())))?;
        drop(f);

        std::fs::rename(&tmp, &full).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            GarudaError::Storage(format!("rename into {}: {e}", full.display()))
        })
    }

    fn remove(&self, path: &Path) -> Result<(), GarudaError> {
        let full = self.resolve(path)?;
        match std::fs::remove_file(&full) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(GarudaError::Storage(format!(
                "remove {}: {e}",
                full.display()
            ))),
        }
    }

    fn exists(&self, path: &Path) -> bool {
        self.resolve(path).map(|p| p.exists()).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_backend(tag: &str) -> (LocalStorageBackend, PathBuf) {
        let dir = std::env::temp_dir().join(format!("garuda_storage_test_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        (LocalStorageBackend::new(&dir), dir)
    }

    #[test]
    fn write_read_remove_round_trip() {
        let (s, dir) = temp_backend("rt");
        let p = Path::new("nested/block_1.bin");

        assert!(!s.exists(p));
        s.write(p, b"hello").unwrap();
        assert!(s.exists(p));
        assert_eq!(s.read(p).unwrap(), b"hello");

        s.remove(p).unwrap();
        assert!(!s.exists(p));
        // Removing what is already gone is not an error.
        s.remove(p).unwrap();

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn rejects_traversal_and_absolute_paths() {
        let (s, dir) = temp_backend("traversal");

        assert!(s.read(Path::new("../../etc/passwd")).is_err());
        assert!(s.write(Path::new("../escape.bin"), b"x").is_err());
        assert!(s.read(Path::new("/etc/passwd")).is_err());
        assert!(!s.exists(Path::new("../anything")));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn write_leaves_no_temp_file_behind() {
        let (s, dir) = temp_backend("tmp");
        s.write(Path::new("a.bin"), b"data").unwrap();

        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|x| x == "tmp"))
            .collect();
        assert!(leftovers.is_empty(), "temp file was not renamed away");

        let _ = std::fs::remove_dir_all(dir);
    }
}
