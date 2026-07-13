use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;
use garuda::memory::MemoryManager;
use garuda::tokenizer::Tokenizer;
use garuda::moe::MoeEngine;
use garuda::predictor::ExpertPredictor;
use garuda::prefetch::PrefetchEngine;
use garuda::runtime::InferenceRuntime;
use garuda::scheduler::{Scheduler, Request, Priority};

#[tokio::test]
async fn test_inference_pipeline() {
    let l1_capacity = 4;
    let ssd_path = std::env::temp_dir().join("test_ssd_cache");
    let hdd_path = std::env::temp_dir().join("test_hdd_archive");
    let _ = std::fs::create_dir_all(&ssd_path);
    let _ = std::fs::create_dir_all(&hdd_path);

    let memory_manager = Arc::new(MemoryManager::new(l1_capacity, ssd_path, hdd_path));
    let tokenizer = Tokenizer::new();
    let predictor = ExpertPredictor::new(8);
    let prefetch_engine = PrefetchEngine::new(memory_manager.clone(), predictor);
    let moe_engine = Arc::new(MoeEngine::new(garuda::router::RouterType::Mixtral, 8, 2, memory_manager.clone()));
    
    let runtime = Arc::new(InferenceRuntime::new(tokenizer, moe_engine, prefetch_engine));
    
    let text = "Hello world";
    let tokens = runtime.tokenizer.encode(text).unwrap();
    assert!(!tokens.is_empty());
    
    let decoded = runtime.tokenizer.decode(&tokens).unwrap();
    assert!(decoded.contains("Hello"));

    let output = runtime.forward(&tokens).unwrap();
    assert_eq!(output.shape, vec![tokens.len(), 128]);

    let next_token = runtime.sample(&output);
    assert!(next_token < 1000);
}

#[tokio::test]
async fn test_scheduler_queuing() {
    let l1_capacity = 4;
    let ssd_path = std::env::temp_dir().join("test_sched_ssd");
    let hdd_path = std::env::temp_dir().join("test_sched_hdd");
    let _ = std::fs::create_dir_all(&ssd_path);
    let _ = std::fs::create_dir_all(&hdd_path);

    let memory_manager = Arc::new(MemoryManager::new(l1_capacity, ssd_path, hdd_path));
    let tokenizer = Tokenizer::new();
    let predictor = ExpertPredictor::new(8);
    let prefetch_engine = PrefetchEngine::new(memory_manager.clone(), predictor);
    let moe_engine = Arc::new(MoeEngine::new(garuda::router::RouterType::Mixtral, 8, 2, memory_manager.clone()));
    let runtime = Arc::new(InferenceRuntime::new(tokenizer, moe_engine, prefetch_engine));
    
    let scheduler = Arc::new(Scheduler::new(runtime));
    
    let (response_tx, mut response_rx) = mpsc::unbounded_channel();
    let (_cancel_tx, cancel_rx) = oneshot::channel();
    let req = Request {
        id: Uuid::new_v4(),
        user_id: "test_user".to_string(),
        tokens: vec![1, 2, 3],
        priority: Priority::High,
        timeout: std::time::Duration::from_secs(5),
        response_tx,
        cancel_rx,
    };
    
    scheduler.submit_request(req).unwrap();
    
    let mut received_tokens = Vec::new();
    while let Some(res) = response_rx.recv().await {
        if let Ok(tok) = res {
            received_tokens.push(tok);
        }
    }
    
    assert_eq!(received_tokens.len(), 3);
}
