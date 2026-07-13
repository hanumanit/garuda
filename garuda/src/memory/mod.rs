use std::sync::Arc;
use std::path::{Path, PathBuf};
use crate::core::{ExpertId, Expert, GarudaError, Tensor};
use crate::cache::ExpertCache;

pub enum MemoryTier {
    L1RAM,
    L2SSD,
    L3HDD,
}

pub struct MemoryManager {
    pub l1_cache: ExpertCache,
    pub l2_ssd_path: PathBuf,
    pub l3_hdd_path: PathBuf,
}

impl MemoryManager {
    pub fn new(l1_capacity: usize, l2_path: PathBuf, l3_path: PathBuf) -> Self {
        Self {
            l1_cache: ExpertCache::new(l1_capacity),
            l2_ssd_path: l2_path,
            l3_hdd_path: l3_path,
        }
    }

    pub fn get_expert(&self, id: ExpertId) -> Result<Arc<Expert>, GarudaError> {
        // 1. Try L1 RAM
        if let Some(expert) = self.l1_cache.get(id) {
            return Ok(expert);
        }

        // 2. Try L2 SSD
        let ssd_file = self.l2_ssd_path.join(format!("expert_{}.bin", id));
        if ssd_file.exists() {
            let expert = self.load_expert_mmap(id, &ssd_file)?;
            self.l1_cache.insert(id, expert.clone());
            return Ok(expert);
        }

        // 3. Try L3 HDD
        let hdd_file = self.l3_hdd_path.join(format!("expert_{}.bin", id));
        if hdd_file.exists() {
            let expert = self.load_expert_mmap(id, &hdd_file)?;
            // Move/copy to L2 SSD for future warm access
            let ssd_file_target = self.l2_ssd_path.join(format!("expert_{}.bin", id));
            if let Some(parent) = ssd_file_target.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::copy(&hdd_file, &ssd_file_target);
            self.l1_cache.insert(id, expert.clone());
            return Ok(expert);
        }

        // Default: Create/Simulate one
        let simulated_weights = Tensor::zeros(vec![1024]);
        let expert = Arc::new(Expert {
            id,
            weights: simulated_weights,
            hits: 1,
            loaded_at: std::time::Instant::now(),
        });
        self.l1_cache.insert(id, expert.clone());
        Ok(expert)
    }

    fn load_expert_mmap(&self, id: ExpertId, path: &Path) -> Result<Arc<Expert>, GarudaError> {
        let file = std::fs::File::open(path)
            .map_err(|e| GarudaError::Io(format!("Failed to open file {}: {}", path.display(), e)))?;
        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .map_err(|e| GarudaError::Io(format!("Failed to mmap file {}: {}", path.display(), e)))?
        };
        let len = mmap.len() / 4;
        let mut data = vec![0.0; len.min(100)];
        if len > 0 {
            let slice = unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const f32, len.min(100)) };
            data.copy_from_slice(slice);
        }
        let weights = Tensor::new(vec![data.len()], data);
        Ok(Arc::new(Expert {
            id,
            weights,
            hits: 1,
            loaded_at: std::time::Instant::now(),
        }))
    }

    pub fn unload_expert(&self, id: ExpertId) {
        self.l1_cache.remove(id);
    }
}
