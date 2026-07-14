//! The format-agnostic core every API front end shares.
//!
//! The wire formats (OpenAI, Ollama, Anthropic, llama.cpp, TGI) genuinely differ —
//! request shapes, response shapes, and streaming protocols — so each needs its own
//! adapter. But the part that talks to the *engine* is identical: render a prompt,
//! submit it, and drive the scheduler either to a full reply or a stream of decoded
//! text. That lives here once, so an adapter is only ever "parse the request, call
//! this, format the reply".

use crate::api::SharedState;
use crate::core::{GarudaError, Token};
use crate::runtime::StopReason;
use crate::scheduler::{Handle, Priority, RequestSpec, StreamEvent};
use futures_util::Stream;

/// One normalized streaming event, independent of any wire format.
pub enum Piece {
    /// A run of decoded text (never empty).
    Text(String),
    /// The reply finished, with the reason it stopped.
    Done(StopReason),
    /// The request failed mid-stream.
    Error(GarudaError),
}

/// Flatten chat turns into the flat prompt the tokenizer sees.
pub fn render_chat<'a>(turns: impl IntoIterator<Item = (&'a str, &'a str)>) -> String {
    let mut p = String::new();
    for (role, content) in turns {
        p.push_str(role);
        p.push_str(": ");
        p.push_str(content);
        p.push('\n');
    }
    p.push_str("assistant: ");
    p
}

/// Queue a prompt on the scheduler; the returned handle streams the result and cancels
/// the work when dropped.
pub fn submit(
    state: &SharedState,
    user: &str,
    prompt: Vec<Token>,
    params: crate::runtime::SamplingParams,
    priority: Priority,
) -> Result<Handle, GarudaError> {
    state.scheduler.submit(RequestSpec {
        user_id: user.to_owned(),
        prompt,
        params,
        priority,
        timeout: state.request_timeout,
    })
}

/// A fully collected reply.
pub struct Reply {
    pub text: String,
    pub reason: StopReason,
    /// Number of tokens generated.
    pub tokens: usize,
}

/// Drive a request to completion and decode the whole reply.
pub async fn collect(state: &SharedState, mut handle: Handle) -> Result<Reply, GarudaError> {
    let mut toks = Vec::new();
    let mut reason = None;
    while let Some(ev) = handle.events.recv().await {
        match ev {
            StreamEvent::Token(t) => toks.push(t),
            StreamEvent::Done(r) => {
                reason = Some(r);
                break;
            }
            StreamEvent::Error(e) => return Err(e),
        }
    }
    let reason =
        reason.ok_or_else(|| GarudaError::Scheduler("stream ended without a result".into()))?;
    let text = state.runtime.tokenizer.decode(&toks)?;
    Ok(Reply {
        text,
        reason,
        tokens: toks.len(),
    })
}

/// Drive a request as a stream of decoded [`Piece`]s. Handles the streaming decoder
/// (partial UTF-8, trailing flush) so adapters only map pieces to their wire format.
pub fn pieces(state: SharedState, mut handle: Handle) -> impl Stream<Item = Piece> {
    async_stream::stream! {
        // `handle` moves in; if the client disconnects and the response is dropped, so is
        // the handle, which cancels the request.
        let mut decoder = state.runtime.tokenizer.stream_decoder();
        while let Some(ev) = handle.events.recv().await {
            match ev {
                StreamEvent::Token(t) => {
                    let s = decoder.push(t);
                    if !s.is_empty() {
                        yield Piece::Text(s);
                    }
                }
                StreamEvent::Done(reason) => {
                    let tail = decoder.finish();
                    if !tail.is_empty() {
                        yield Piece::Text(tail);
                    }
                    yield Piece::Done(reason);
                    return;
                }
                StreamEvent::Error(e) => {
                    yield Piece::Error(e);
                    return;
                }
            }
        }
    }
}
