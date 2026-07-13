จากที่คุยกันก่อนหน้านี้ (Synology NAS, DS920+, RAM 20GB, SSD Cache 2TB, ต้องการใช้เป็น AI Coding Agent และรองรับ Local LLM) ถ้าจะสร้างโปรเจ็กต์ใหม่ ผมแนะนำว่า อย่าทำ Colibri clone แต่ให้สร้าง Runtime ใหม่ที่รองรับทั้ง Dense และ MoE ตั้งแต่แรก

ผมตั้งชื่อชั่วคราวว่า

Project Garuda
High Performance Rust LLM Runtime with Expert Streaming

⸻

1. เป้าหมาย

สร้าง Runtime สำหรับ Local LLM ที่

* รองรับ GGUF
* รองรับ MoE
* รองรับ Expert Streaming
* ใช้ SSD เป็น Memory Tier
* รองรับหลายผู้ใช้
* ใช้เป็น Backend ของ Coding Agent
* รองรับ MCP
* รองรับ OpenAI API Compatible
* ทำงานบน Linux / Synology / macOS

⸻

2. Architecture

                        REST / OpenAI API
                                │
                      WebSocket Streaming
                                │
                     Session Manager
                                │
                  Request Scheduler
                                │
          ┌──────────────┬──────────────┐
          │              │              │
      Dense Engine   MoE Engine   Embedding Engine
          │              │              │
          └──────────────┴──────────────┘
                         │
                  Inference Runtime
                         │
        ┌────────────────────────────────┐
        │ Router │ Expert Loader │ KV Cache │
        └────────────────────────────────┘
                         │
                 Memory Manager
                         │
      RAM ←→ SSD Cache ←→ HDD/NAS
                         │
                 GGUF / Expert Files

⸻

3. Module Layout

garuda/
core/
runtime/
scheduler/
memory/
storage/
tokenizer/
gguf/
attention/
moe/
router/
cache/
prefetch/
predictor/
simd/
cuda/
api/
grpc/
websocket/
cli/
config/
benchmark/

⸻

4. Runtime Layer

Runtime
↓
Tokenizer
↓
Prompt Cache
↓
KV Cache
↓
Router
↓
Expert Loader
↓
Inference
↓
Sampler
↓
Streaming Output

⸻

5. Memory Manager

แบ่ง Memory เป็น 3 ระดับ

L1 RAM
Hot Experts
KV Cache
---------------------
L2 SSD
Warm Experts
GGUF
---------------------
L3 HDD/NAS
Cold Experts
Archive

ทุก Layer ใช้

mmap()

แทน read()

⸻

6. Expert Loader

Interface

pub trait ExpertLoader {
    fn load(id: ExpertId) -> Arc<Expert>;
    fn unload(id: ExpertId);
    fn prefetch(id: ExpertId);
}

รองรับ

* mmap
* async load
* compression

⸻

7. Router

รับผิดชอบ

Token
↓
Top-K Experts
↓
Schedule
↓
Execute

รองรับ

* Mixtral
* DeepSeek
* Qwen MoE

⸻

8. Prefetch Engine

Predict Expert ก่อนใช้งาน

Current Tokens
↓
Prediction
↓
SSD Read
↓
RAM Cache
↓
Inference

เป้าหมาย

Latency SSD ≈ RAM

⸻

9. Scheduler

รองรับ

Multi User
Priority
Batch Merge
Streaming
Cancellation
Timeout
Rate Limit

⸻

10. Cache

แบ่งหลายชนิด

Prompt Cache
KV Cache
Expert Cache
Embedding Cache
Tokenizer Cache

⸻

11. KV Cache

รองรับ

Paged KV
Shared KV
Sliding Window
Disk Spill

⸻

12. Storage

รองรับ

GGUF
SafeTensor
MoE Experts
Tokenizer
Embedding

⸻

13. API

REST

POST /v1/chat
POST /v1/completion
POST /v1/embed
POST /v1/models

Compatible กับ OpenAI

⸻

14. MCP

รองรับ

Filesystem
Git
Docker
SQL
SSH
Browser
Pathology
Custom MCP

⸻

15. SIMD

ใช้

AVX2
AVX512
NEON
SVE2

ผ่าน

Rust

std::simd

หรือ

portable_simd

⸻

16. Async

ใช้

Tokio

ทุกงาน I/O

Expert Loading
Streaming
WebSocket
API
Background Cache

⸻

17. Thread Model

Main Thread
↓
Scheduler
↓
Inference Pool
↓
IO Pool
↓
Background Cache

ทุก Pool แยกจากกัน

⸻

18. Crates

tokio
axum
dashmap
rayon
memmap2
bytes
serde
serde_json
parking_lot
tracing
thiserror
clap
uuid
crossbeam
lru
blake3
zstd
anyhow

⸻

19. Configuration

[model]
path="/models"
context=32768
gpu=false
threads=8
expert_cache="256GB"
prefetch=true
predictor=true

⸻

20. Benchmark Targets

Metric	Target
Startup	< 1 sec
Load Expert	< 5 ms
Cache Hit	>95%
Token Latency	< 25 ms
Throughput	>300 tok/s CPU
API P99	<100 ms

⸻

21. Coding Standards

unsafe
↓
เฉพาะ SIMD
และ mmap

ส่วนอื่นใช้ Safe Rust ทั้งหมด

⸻

22. Logging

Tracing
OpenTelemetry
Prometheus
Grafana

⸻

23. Roadmap

Phase 1

* GGUF Loader
* Tokenizer
* Dense Inference
* REST API

Phase 2

* KV Cache
* Streaming
* Scheduler

Phase 3

* MoE Runtime
* Expert Streaming
* SSD Cache

Phase 4

* Prefetch Predictor
* Multi-user
* MCP
* Agent Runtime

Phase 5

* CUDA Backend
* ROCm Backend
* Distributed Runtime

⸻

จุดที่อยากเพิ่มจาก Colibri

ผมคิดว่าถ้าจะทำให้เหนือกว่า Colibri ควรออกแบบตั้งแต่ต้นให้เป็น Plugin-based Runtime แทนการผูกทุกอย่างไว้ในตัวโปรแกรม โดยกำหนด Trait หลัก เช่น StorageBackend, ModelLoader, InferenceBackend, SchedulerPolicy และ CachePolicy เพื่อให้สามารถเพิ่ม backend ใหม่ (เช่น CPU, CUDA, Vulkan, Metal หรือแม้แต่ NPU ในอนาคต) ได้โดยไม่ต้องแก้แกนหลักของระบบ

นอกจากนี้ หากเป้าหมายคือการใช้งานกับ AI Coding Agent ของ HANUMANIT ผมแนะนำให้แยก Runtime สำหรับ LLM ออกจาก Agent Runtime อย่างชัดเจน เพื่อให้ LLM Engine สามารถนำไปใช้ซ้ำกับงานอื่น เช่น Pathology AI, RAG และบริการภายในองค์กร โดยไม่ผูกติดกับระบบ Coding Agent เพียงอย่างเดียว ซึ่งจะทำให้โปรเจ็กต์มีความยืดหยุ่นและต่อยอดได้ในระยะยาว.

แนวคิดพื้นฐานจาก https://github.com/JustVugg/colibri