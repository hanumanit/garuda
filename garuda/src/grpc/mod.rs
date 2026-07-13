pub struct GrpcServer;

impl GrpcServer {
    pub fn new() -> Self {
        Self
    }
    
    pub async fn run(&self) -> Result<(), crate::core::GarudaError> {
        Ok(())
    }
}
