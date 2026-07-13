use crate::core::{StorageBackend, GarudaError};
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;
use memmap2::Mmap;

pub struct LocalStorageBackend {
    base_path: PathBuf,
}

impl LocalStorageBackend {
    pub fn new<P: Into<PathBuf>>(base_path: P) -> Self {
        Self {
            base_path: base_path.into(),
        }
    }
}

impl StorageBackend for LocalStorageBackend {
    fn read_file(&self, path: &str) -> Result<Vec<u8>, GarudaError> {
        let full_path = self.base_path.join(path);
        let mut file = File::open(&full_path)
            .map_err(|e| GarudaError::Storage(format!("Failed to open file {}: {}", full_path.display(), e)))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| GarudaError::Storage(format!("Failed to read file {}: {}", full_path.display(), e)))?;
        Ok(buffer)
    }

    fn read_mmap(&self, path: &str) -> Result<Mmap, GarudaError> {
        let full_path = self.base_path.join(path);
        let file = File::open(&full_path)
            .map_err(|e| GarudaError::Storage(format!("Failed to open file {}: {}", full_path.display(), e)))?;
        unsafe {
            Mmap::map(&file)
                .map_err(|e| GarudaError::Storage(format!("Failed to mmap file {}: {}", full_path.display(), e)))
        }
    }

    fn write_file(&self, path: &str, data: &[u8]) -> Result<(), GarudaError> {
        let full_path = self.base_path.join(path);
        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| GarudaError::Storage(format!("Failed to create directories: {}", e)))?;
        }
        let mut file = File::create(&full_path)
            .map_err(|e| GarudaError::Storage(format!("Failed to create file {}: {}", full_path.display(), e)))?;
        file.write_all(data)
            .map_err(|e| GarudaError::Storage(format!("Failed to write to file {}: {}", full_path.display(), e)))?;
        Ok(())
    }
}
