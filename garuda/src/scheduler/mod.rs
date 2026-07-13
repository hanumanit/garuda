use crate::core::{Token, GarudaError};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Duration;
use parking_lot::Mutex;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low,
    Normal,
    High,
}

pub struct Request {
    pub id: Uuid,
    pub user_id: String,
    pub tokens: Vec<Token>,
    pub priority: Priority,
    pub timeout: Duration,
    pub response_tx: mpsc::UnboundedSender<Result<Token, GarudaError>>,
    pub cancel_rx: oneshot::Receiver<()>,
}

pub struct Scheduler {
    request_tx: mpsc::UnboundedSender<Request>,
    rate_limits: Mutex<std::collections::HashMap<String, usize>>,
}

impl Scheduler {
    pub fn new(inference_engine: Arc<crate::runtime::InferenceRuntime>) -> Self {
        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<Request>();
        let rate_limits = Mutex::new(std::collections::HashMap::new());

        tokio::spawn(async move {
            let mut batch = Vec::new();
            
            loop {
                let mut received_any = false;
                
                while let Ok(req) = request_rx.try_recv() {
                    batch.push(req);
                    received_any = true;
                    if batch.len() >= 8 {
                        break;
                    }
                }

                if !received_any && batch.is_empty() {
                    if let Some(req) = request_rx.recv().await {
                        batch.push(req);
                    } else {
                        break;
                    }
                }

                if batch.len() < 8 {
                    tokio::time::sleep(Duration::from_millis(2)).await;
                    while let Ok(req) = request_rx.try_recv() {
                        batch.push(req);
                        if batch.len() >= 8 {
                            break;
                        }
                    }
                }

                batch.sort_by(|a, b| b.priority.cmp(&a.priority));

                let mut active_batch = Vec::new();
                for req in batch.drain(..) {
                    active_batch.push(req);
                }

                if active_batch.is_empty() {
                    continue;
                }

                for mut req in active_batch {
                    if req.cancel_rx.try_recv().is_ok() {
                        let _ = req.response_tx.send(Err(GarudaError::Scheduler("Request cancelled".to_string())));
                        continue;
                    }

                    let engine = inference_engine.clone();
                    tokio::spawn(async move {
                        let result = tokio::time::timeout(req.timeout, async {
                            match engine.forward(&req.tokens) {
                                Ok(_out_tensor) => {
                                    for &t in &req.tokens {
                                        let out_token = (t + 1) % 1000;
                                        if req.response_tx.send(Ok(out_token)).is_err() {
                                            break;
                                        }
                                        tokio::time::sleep(Duration::from_millis(10)).await;
                                    }
                                    Ok(())
                                }
                                Err(e) => Err(e),
                            }
                        }).await;

                        match result {
                            Ok(Ok(())) => {},
                            Ok(Err(e)) => {
                                let _ = req.response_tx.send(Err(e));
                            }
                            Err(_) => {
                                let _ = req.response_tx.send(Err(GarudaError::Timeout));
                            }
                        }
                    });
                }
            }
        });

        Self {
            request_tx,
            rate_limits,
        }
    }

    pub fn submit_request(&self, request: Request) -> Result<(), GarudaError> {
        {
            let mut limits = self.rate_limits.lock();
            let count = limits.entry(request.user_id.clone()).or_insert(0);
            if *count >= 10 {
                return Err(GarudaError::RateLimit);
            }
            *count += 1;
        }

        self.request_tx.send(request)
            .map_err(|e| GarudaError::Scheduler(format!("Failed to submit request: {}", e)))
    }

    pub fn release_rate_limit(&self, user_id: &str) {
        let mut limits = self.rate_limits.lock();
        if let Some(count) = limits.get_mut(user_id) {
            if *count > 0 {
                *count -= 1;
            }
        }
    }
}
