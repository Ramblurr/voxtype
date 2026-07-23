//! ElevenLabs Scribe protocol primitives.
//!
//! This module keeps the provider's wire format, audio framing, request
//! construction, and transcript reconciliation separate from the transport
//! lifecycle. The batch and realtime adapters build on these helpers.

#![allow(
    dead_code,
    reason = "protocol primitives are wired into transports in the following implementation tasks"
)]

use std::{sync::OnceLock, time::Duration};

use base64::Engine as _;
use reqwest::{
    header::{HeaderName, HeaderValue},
    StatusCode,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tokio_tungstenite::tungstenite::{client::IntoClientRequest, http::Request};

use super::{SegmentId, StreamingEvent, Transcriber};
use crate::{
    config::{ElevenLabsConfig, ElevenLabsRegion},
    error::TranscribeError,
};

const REALTIME_MODEL: &str = "scribe_v2_realtime";
const REALTIME_AUDIO_FORMAT: &str = "pcm_16000";
const BATCH_MODEL: &str = "scribe_v2";
const BATCH_FILE_FORMAT: &str = "pcm_s16le_16";
const BATCH_FILE_NAME: &str = "voxtype.pcm";
const BATCH_FILE_MIME: &str = "application/octet-stream";
const SAMPLE_RATE: u32 = 16_000;
const COMMIT_STRATEGY: &str = "vad";
const VAD_SILENCE_THRESHOLD_SECS: &str = "1.5";
const VAD_THRESHOLD: &str = "0.4";
const MIN_SPEECH_DURATION_MS: &str = "100";
const MIN_SILENCE_DURATION_MS: &str = "100";
const FRAME_DURATION_MS: usize = 100;
const SAMPLES_PER_FRAME: usize = SAMPLE_RATE as usize * FRAME_DURATION_MS / 1_000;
const BYTES_PER_FRAME: usize = SAMPLES_PER_FRAME * size_of::<i16>();
const MIN_BATCH_SAMPLES: usize = SAMPLE_RATE as usize / 10;
const BATCH_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_PROVIDER_ERROR_CHARS: usize = 512;
const API_KEY_HEADER: HeaderName = HeaderName::from_static("xi-api-key");

fn api_origin(region: ElevenLabsRegion) -> &'static str {
    match region {
        ElevenLabsRegion::Global => "api.elevenlabs.io",
        ElevenLabsRegion::Us => "api.us.elevenlabs.io",
        ElevenLabsRegion::Eu => "api.eu.residency.elevenlabs.io",
        ElevenLabsRegion::India => "api.in.residency.elevenlabs.io",
        ElevenLabsRegion::Singapore => "api.sg.residency.elevenlabs.io",
    }
}

fn realtime_url(config: &ElevenLabsConfig) -> Result<reqwest::Url, TranscribeError> {
    let endpoint = format!(
        "wss://{}/v1/speech-to-text/realtime",
        api_origin(config.region)
    );
    let mut url = reqwest::Url::parse(&endpoint).map_err(|error| {
        TranscribeError::ConfigError(format!("Invalid ElevenLabs realtime endpoint: {error}"))
    })?;

    {
        let mut query = url.query_pairs_mut();
        query
            .append_pair("model_id", REALTIME_MODEL)
            .append_pair("audio_format", REALTIME_AUDIO_FORMAT)
            .append_pair("commit_strategy", COMMIT_STRATEGY)
            .append_pair("vad_silence_threshold_secs", VAD_SILENCE_THRESHOLD_SECS)
            .append_pair("vad_threshold", VAD_THRESHOLD)
            .append_pair("min_speech_duration_ms", MIN_SPEECH_DURATION_MS)
            .append_pair("min_silence_duration_ms", MIN_SILENCE_DURATION_MS)
            .append_pair("no_verbatim", "false");
        if let Some(language_code) = config.language_code.as_deref() {
            query.append_pair("language_code", language_code);
        }
    }

    Ok(url)
}

fn sensitive_api_key_header(api_key: &str) -> Result<HeaderValue, TranscribeError> {
    let mut value = HeaderValue::from_str(api_key).map_err(|_| {
        TranscribeError::ConfigError(
            "ElevenLabs API key contains bytes that cannot be used in an HTTP header".to_string(),
        )
    })?;
    value.set_sensitive(true);
    Ok(value)
}

fn realtime_request(
    config: &ElevenLabsConfig,
    api_key: &str,
) -> Result<Request<()>, TranscribeError> {
    let url = realtime_url(config)?;
    let mut request = url.as_str().into_client_request().map_err(|error| {
        TranscribeError::ConfigError(format!(
            "Could not construct ElevenLabs realtime request: {error}"
        ))
    })?;
    request
        .headers_mut()
        .insert(API_KEY_HEADER, sensitive_api_key_header(api_key)?);
    Ok(request)
}

pub struct ElevenLabsTranscriber {
    config: ElevenLabsConfig,
    api_key: String,
    batch_client: OnceLock<reqwest::Client>,
}

impl ElevenLabsTranscriber {
    pub fn new(config: &ElevenLabsConfig) -> Result<Self, TranscribeError> {
        let api_key = config.api_key.clone().ok_or_else(|| {
            TranscribeError::ConfigError(
                "ElevenLabs API key required: set [elevenlabs] api_key or ELEVENLABS_API_KEY"
                    .to_string(),
            )
        })?;
        sensitive_api_key_header(&api_key)?;

        let mut config = config.clone();
        config.api_key = None;

        Ok(Self {
            config,
            api_key,
            batch_client: OnceLock::new(),
        })
    }

    fn batch_client(&self) -> Result<&reqwest::Client, TranscribeError> {
        if let Some(client) = self.batch_client.get() {
            return Ok(client);
        }

        let client = build_batch_client()?;
        let _ = self.batch_client.set(client);
        Ok(self
            .batch_client
            .get()
            .expect("ElevenLabs batch client was just initialized"))
    }

    async fn batch_transcribe(&self, samples: &[f32]) -> Result<String, TranscribeError> {
        validate_batch_input(samples)?;

        let client = self.batch_client()?;
        let upload = BatchUpload::new(samples, self.config.language_code.clone());
        let request = build_batch_request(client, &self.config, &self.api_key, upload)?;
        let response = client.execute(request).await.map_err(|error| {
            TranscribeError::NetworkError(format!(
                "ElevenLabs batch request failed: {}",
                redact_and_bound(&error.to_string(), &self.api_key)
            ))
        })?;
        let status = response.status();
        let body = response.text().await.map_err(|error| {
            TranscribeError::NetworkError(format!(
                "Could not read ElevenLabs batch response: {}",
                redact_and_bound(&error.to_string(), &self.api_key)
            ))
        })?;

        if !status.is_success() {
            return Err(map_batch_status(status, &body, &self.api_key));
        }

        parse_batch_response(&body)
    }
}

impl Transcriber for ElevenLabsTranscriber {
    /// Run one synchronous batch transcription against ElevenLabs Scribe.
    ///
    /// Calls made inside Tokio must use a multi-thread runtime because the
    /// bridge uses `block_in_place`. Calls without an ambient runtime use a
    /// private current-thread runtime.
    fn transcribe(&self, samples: &[f32]) -> Result<String, TranscribeError> {
        validate_batch_input(samples)?;

        let run = self.batch_transcribe(samples);
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| handle.block_on(run)),
            Err(_) => {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|error| {
                        TranscribeError::InferenceFailed(format!(
                            "Could not create ElevenLabs batch runtime: {error}"
                        ))
                    })?;
                runtime.block_on(run)
            }
        }
    }
}

fn validate_batch_input(samples: &[f32]) -> Result<(), TranscribeError> {
    if samples.is_empty() {
        return Err(TranscribeError::AudioFormat(
            "Empty audio buffer".to_string(),
        ));
    }
    if samples.len() < MIN_BATCH_SAMPLES {
        return Err(TranscribeError::AudioFormat(format!(
            "ElevenLabs batch transcription requires at least 100 ms of audio ({MIN_BATCH_SAMPLES} samples)"
        )));
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct BatchUpload {
    pcm16le: Vec<u8>,
    language_code: Option<String>,
    model_id: &'static str,
    file_format: &'static str,
    tag_audio_events: bool,
    diarize: bool,
    webhook: bool,
}

impl BatchUpload {
    fn new(samples: &[f32], language_code: Option<String>) -> Self {
        Self {
            pcm16le: f32_to_pcm16le(samples),
            language_code,
            model_id: BATCH_MODEL,
            file_format: BATCH_FILE_FORMAT,
            tag_audio_events: false,
            diarize: false,
            webhook: false,
        }
    }

    fn into_multipart(self) -> Result<reqwest::multipart::Form, TranscribeError> {
        let file = reqwest::multipart::Part::bytes(self.pcm16le)
            .file_name(BATCH_FILE_NAME)
            .mime_str(BATCH_FILE_MIME)
            .map_err(|error| {
                TranscribeError::InferenceFailed(format!(
                    "Could not construct ElevenLabs PCM upload: {error}"
                ))
            })?;

        let mut form = reqwest::multipart::Form::new()
            .part("file", file)
            .text("file_format", self.file_format)
            .text("model_id", self.model_id);
        if let Some(language_code) = self.language_code {
            form = form.text("language_code", language_code);
        }

        Ok(form
            .text("tag_audio_events", self.tag_audio_events.to_string())
            .text("diarize", self.diarize.to_string())
            .text("webhook", self.webhook.to_string()))
    }
}

fn build_batch_client() -> Result<reqwest::Client, TranscribeError> {
    reqwest::Client::builder()
        // The custom xi-api-key header is not covered by reqwest's
        // cross-origin redirect sanitizer. Never forward it away from the
        // fixed regional production endpoint.
        .redirect(reqwest::redirect::Policy::none())
        .timeout(BATCH_REQUEST_TIMEOUT)
        .build()
        .map_err(|error| {
            TranscribeError::InferenceFailed(format!(
                "Could not initialize ElevenLabs HTTP client: {error}"
            ))
        })
}

fn batch_url(config: &ElevenLabsConfig) -> Result<reqwest::Url, TranscribeError> {
    let endpoint = format!("https://{}/v1/speech-to-text", api_origin(config.region));
    reqwest::Url::parse(&endpoint).map_err(|error| {
        TranscribeError::ConfigError(format!("Invalid ElevenLabs batch endpoint: {error}"))
    })
}

fn build_batch_request(
    client: &reqwest::Client,
    config: &ElevenLabsConfig,
    api_key: &str,
    upload: BatchUpload,
) -> Result<reqwest::Request, TranscribeError> {
    client
        .post(batch_url(config)?)
        .header(API_KEY_HEADER, sensitive_api_key_header(api_key)?)
        .multipart(upload.into_multipart()?)
        .build()
        .map_err(|error| {
            TranscribeError::InferenceFailed(format!(
                "Could not construct ElevenLabs batch request: {error}"
            ))
        })
}

fn parse_batch_response(body: &str) -> Result<String, TranscribeError> {
    let response: serde_json::Value = serde_json::from_str(body).map_err(|error| {
        TranscribeError::InferenceFailed(format!(
            "Could not parse ElevenLabs batch response: {error}"
        ))
    })?;

    response
        .get("text")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| {
            TranscribeError::InferenceFailed("ElevenLabs batch response missing text".to_string())
        })
}

struct BatchErrorInfo {
    provider_status: Option<String>,
    message: String,
}

fn batch_error_info(body: &str) -> BatchErrorInfo {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body) else {
        return BatchErrorInfo {
            provider_status: None,
            message: body.to_string(),
        };
    };

    let provider_status = value
        .pointer("/detail/status")
        .or_else(|| value.get("status"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let message = value
        .pointer("/detail/message")
        .or_else(|| value.get("message"))
        .or_else(|| value.get("error"))
        .or_else(|| value.get("detail"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or(body)
        .to_string();

    BatchErrorInfo {
        provider_status,
        message,
    }
}

fn map_batch_status(status: StatusCode, body: &str, api_key: &str) -> TranscribeError {
    let error_info = batch_error_info(body);
    let provider_status = error_info.provider_status.as_deref();
    let message_hint = error_info.message.to_ascii_lowercase();
    let unaccepted_terms = provider_status == Some("unaccepted_terms")
        || message_hint.contains("terms of service")
        || message_hint.contains("terms have not been accepted")
        || message_hint.contains("accept the terms");
    let authentication_failure = status == StatusCode::UNAUTHORIZED
        || matches!(
            provider_status,
            Some("auth_error" | "invalid_api_key" | "authentication_error")
        )
        || (status == StatusCode::FORBIDDEN
            && (message_hint.contains("api key")
                || message_hint.contains("api-key")
                || message_hint.contains("authentication")));

    let message = if matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN)
        && unaccepted_terms
    {
        "ElevenLabs Scribe terms have not been accepted: accept them in the ElevenLabs dashboard"
            .to_string()
    } else if authentication_failure {
        "ElevenLabs authentication failed: check [elevenlabs] api_key or ELEVENLABS_API_KEY"
            .to_string()
    } else if status == StatusCode::FORBIDDEN {
        "ElevenLabs access denied: check API key permissions and Scribe access".to_string()
    } else if status == StatusCode::TOO_MANY_REQUESTS {
        "ElevenLabs rate limit exceeded: wait and try again".to_string()
    } else {
        let detail = redact_and_bound(error_info.message.trim(), api_key);
        if detail.is_empty() {
            format!("ElevenLabs batch request failed (HTTP {status})")
        } else {
            format!("ElevenLabs batch request failed (HTTP {status}): {detail}")
        }
    };

    TranscribeError::InferenceFailed(message)
}

fn f32_to_pcm16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16
}

fn f32_to_pcm16le(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * size_of::<i16>());
    for &sample in samples {
        bytes.extend_from_slice(&f32_to_pcm16(sample).to_le_bytes());
    }
    bytes
}

#[derive(Debug, Default)]
struct PcmFrameAccumulator {
    pending: Vec<u8>,
}

impl PcmFrameAccumulator {
    fn push(&mut self, samples: &[f32]) -> Vec<Vec<u8>> {
        self.pending.extend(f32_to_pcm16le(samples));

        let complete_bytes = self.pending.len() / BYTES_PER_FRAME * BYTES_PER_FRAME;
        let frames = self.pending[..complete_bytes]
            .chunks_exact(BYTES_PER_FRAME)
            .map(<[u8]>::to_vec)
            .collect();
        self.pending.drain(..complete_bytes);
        frames
    }

    fn flush(&mut self) -> Option<Vec<u8>> {
        (!self.pending.is_empty()).then(|| std::mem::take(&mut self.pending))
    }
}

#[derive(Debug, Serialize)]
struct InputAudioChunk {
    message_type: &'static str,
    audio_base_64: String,
    commit: bool,
    sample_rate: u32,
}

fn serialize_audio_chunk(pcm16le: &[u8]) -> Result<String, TranscribeError> {
    serialize_input_audio_chunk(pcm16le, false)
}

fn serialize_final_commit() -> Result<String, TranscribeError> {
    serialize_input_audio_chunk(&[], true)
}

fn serialize_input_audio_chunk(pcm16le: &[u8], commit: bool) -> Result<String, TranscribeError> {
    let message = InputAudioChunk {
        message_type: "input_audio_chunk",
        audio_base_64: base64::engine::general_purpose::STANDARD.encode(pcm16le),
        commit,
        sample_rate: SAMPLE_RATE,
    };
    serde_json::to_string(&message).map_err(|error| {
        TranscribeError::InferenceFailed(format!(
            "Could not serialize ElevenLabs audio message: {error}"
        ))
    })
}

#[derive(Debug, PartialEq, Eq)]
enum WireMessage {
    SessionStarted { session_id: String },
    Partial { text: String },
    Committed { text: String },
    CommittedWithTimestamps { text: String },
    ProviderError { kind: String, message: String },
    Unknown,
}

#[derive(Debug, Deserialize)]
struct MessageEnvelope {
    message_type: String,
}

#[derive(Debug, Deserialize)]
struct SessionStartedPayload {
    session_id: String,
}

#[derive(Debug, Deserialize)]
struct TranscriptPayload {
    text: String,
}

#[derive(Debug, Deserialize)]
struct ProviderErrorPayload {
    error: String,
}

const PROVIDER_ERROR_EVENTS: &[&str] = &[
    "error",
    "auth_error",
    "quota_exceeded",
    "commit_throttled",
    "unaccepted_terms",
    "rate_limited",
    "queue_overflow",
    "resource_exhausted",
    "session_time_limit_exceeded",
    "input_error",
    "chunk_size_exceeded",
    "insufficient_audio_activity",
    "transcriber_error",
];

fn parse_payload<T: DeserializeOwned>(
    value: &serde_json::Value,
    message_type: &str,
) -> Result<T, TranscribeError> {
    serde_json::from_value(value.clone()).map_err(|error| {
        TranscribeError::InferenceFailed(format!(
            "Malformed ElevenLabs {message_type} message: {error}"
        ))
    })
}

fn parse_wire_message(payload: &str, api_key: &str) -> Result<WireMessage, TranscribeError> {
    let value: serde_json::Value = serde_json::from_str(payload).map_err(|error| {
        TranscribeError::InferenceFailed(format!("Malformed ElevenLabs realtime message: {error}"))
    })?;
    let envelope: MessageEnvelope = parse_payload(&value, "realtime")?;

    match envelope.message_type.as_str() {
        "session_started" => {
            let parsed: SessionStartedPayload = parse_payload(&value, "session_started")?;
            Ok(WireMessage::SessionStarted {
                session_id: parsed.session_id,
            })
        }
        "partial_transcript" => {
            let parsed: TranscriptPayload = parse_payload(&value, "partial_transcript")?;
            Ok(WireMessage::Partial { text: parsed.text })
        }
        "committed_transcript" => {
            let parsed: TranscriptPayload = parse_payload(&value, "committed_transcript")?;
            Ok(WireMessage::Committed { text: parsed.text })
        }
        "committed_transcript_with_timestamps" => {
            let parsed: TranscriptPayload =
                parse_payload(&value, "committed_transcript_with_timestamps")?;
            Ok(WireMessage::CommittedWithTimestamps { text: parsed.text })
        }
        message_type if PROVIDER_ERROR_EVENTS.contains(&message_type) => {
            let parsed: ProviderErrorPayload = parse_payload(&value, message_type)?;
            Ok(WireMessage::ProviderError {
                kind: message_type.to_string(),
                message: redact_and_bound(&parsed.error, api_key),
            })
        }
        _ => {
            // Do not log or retain the provider-controlled event name because a
            // malformed response could echo credentials or audio data in it.
            tracing::debug!("Ignoring unknown ElevenLabs realtime event");
            Ok(WireMessage::Unknown)
        }
    }
}

fn redact_api_key(text: &str, api_key: &str) -> String {
    if api_key.is_empty() {
        text.to_string()
    } else {
        text.replace(api_key, "[REDACTED]")
    }
}

fn redact_and_bound(text: &str, api_key: &str) -> String {
    let redacted = redact_api_key(text, api_key);
    if redacted.chars().count() <= MAX_PROVIDER_ERROR_CHARS {
        return redacted;
    }

    let mut bounded: String = redacted.chars().take(MAX_PROVIDER_ERROR_CHARS).collect();
    bounded.push_str("...");
    bounded
}

#[derive(Debug)]
struct TranscriptReconciler {
    type_partials: bool,
    typed_partial: String,
    segment_id: SegmentId,
}

impl TranscriptReconciler {
    fn new(type_partials: bool) -> Self {
        Self {
            type_partials,
            typed_partial: String::new(),
            segment_id: 0,
        }
    }

    fn process(&mut self, message: WireMessage) -> Vec<StreamingEvent> {
        match message {
            WireMessage::Partial { text } => self.process_partial(text),
            WireMessage::Committed { text } => vec![self.process_commit(text)],
            // The service sends this after `committed_transcript` when timestamps
            // are requested. The preceding event already finalized the segment.
            WireMessage::CommittedWithTimestamps { .. }
            | WireMessage::SessionStarted { .. }
            | WireMessage::Unknown => Vec::new(),
            WireMessage::ProviderError { kind, message } => {
                vec![StreamingEvent::Error(TranscribeError::InferenceFailed(
                    format!("ElevenLabs realtime {kind}: {message}"),
                ))]
            }
        }
    }

    fn process_partial(&mut self, snapshot: String) -> Vec<StreamingEvent> {
        if !self.type_partials || !snapshot.starts_with(&self.typed_partial) {
            return Vec::new();
        }

        let suffix = snapshot[self.typed_partial.len()..].to_string();
        if suffix.is_empty() {
            return Vec::new();
        }

        self.typed_partial = snapshot;
        vec![StreamingEvent::Partial {
            text: suffix,
            segment_id: self.segment_id,
        }]
    }

    fn process_commit(&mut self, committed: String) -> StreamingEvent {
        let segment_id = self.segment_id;
        let event = if committed.starts_with(&self.typed_partial) {
            StreamingEvent::Final {
                text: committed[self.typed_partial.len()..].to_string(),
                segment_id,
            }
        } else {
            let common_chars = common_prefix_char_count(&self.typed_partial, &committed);
            StreamingEvent::Replace {
                backspace: self.typed_partial.chars().count() - common_chars,
                text: committed.chars().skip(common_chars).collect(),
                segment_id,
            }
        };

        self.typed_partial.clear();
        self.segment_id += 1;
        event
    }
}

fn common_prefix_char_count(left: &str, right: &str) -> usize {
    left.chars()
        .zip(right.chars())
        .take_while(|(left, right)| left == right)
        .count()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    const API_KEY: &str = "elevenlabs-secret-key";

    #[derive(Debug, PartialEq, Eq)]
    enum EventView {
        Partial(String, SegmentId),
        Final(String, SegmentId),
        Replace(usize, String, SegmentId),
        Error(String),
    }

    fn event_views(events: Vec<StreamingEvent>) -> Vec<EventView> {
        events
            .into_iter()
            .map(|event| match event {
                StreamingEvent::Partial { text, segment_id } => {
                    EventView::Partial(text, segment_id)
                }
                StreamingEvent::Final { text, segment_id } => EventView::Final(text, segment_id),
                StreamingEvent::Replace {
                    backspace,
                    text,
                    segment_id,
                } => EventView::Replace(backspace, text, segment_id),
                StreamingEvent::Error(error) => EventView::Error(error.to_string()),
                StreamingEvent::Ended => panic!("reconciler does not emit Ended"),
            })
            .collect()
    }

    fn process_all(
        reconciler: &mut TranscriptReconciler,
        messages: impl IntoIterator<Item = WireMessage>,
    ) -> Vec<EventView> {
        event_views(
            messages
                .into_iter()
                .flat_map(|message| reconciler.process(message))
                .collect(),
        )
    }

    #[test]
    fn converts_f32_samples_to_clipped_pcm16le() {
        let samples = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let actual = f32_to_pcm16le(&samples);
        let expected_samples = [-32767i16, -32767, -16384, 0, 16384, 32767, 32767];
        let expected: Vec<u8> = expected_samples
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect();

        assert_eq!(actual, expected);
        assert_eq!(&actual[4..6], &(-16384i16).to_le_bytes());
        assert_eq!(&actual[8..10], &16384i16.to_le_bytes());
    }

    #[test]
    fn accumulates_irregular_chunks_into_100ms_frames_and_flushes_tail() {
        let samples: Vec<f32> = (0..(SAMPLES_PER_FRAME * 2 + 137))
            .map(|index| (index as f32 % 101.0 - 50.0) / 50.0)
            .collect();
        let expected = f32_to_pcm16le(&samples);
        let chunk_sizes = [17, 1_701, 33, 999, 480, 107];
        assert_eq!(chunk_sizes.iter().sum::<usize>(), samples.len());

        let mut accumulator = PcmFrameAccumulator::default();
        let mut frames = Vec::new();
        let mut offset = 0;
        for chunk_size in chunk_sizes {
            frames.extend(accumulator.push(&samples[offset..offset + chunk_size]));
            offset += chunk_size;
        }
        let tail = accumulator.flush().expect("short tail is retained");

        assert_eq!(
            frames.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![BYTES_PER_FRAME, BYTES_PER_FRAME]
        );
        assert_eq!(tail.len(), 137 * size_of::<i16>());
        let reconstructed: Vec<u8> = frames.into_iter().flatten().chain(tail).collect();
        assert_eq!(reconstructed, expected);
        assert!(accumulator.flush().is_none());
    }

    #[test]
    fn serializes_normal_audio_and_final_commit_messages() {
        let audio: serde_json::Value =
            serde_json::from_str(&serialize_audio_chunk(&[0, 1, 2, 3]).unwrap()).unwrap();
        assert_eq!(
            audio,
            serde_json::json!({
                "message_type": "input_audio_chunk",
                "audio_base_64": "AAECAw==",
                "commit": false,
                "sample_rate": 16000
            })
        );

        let commit: serde_json::Value =
            serde_json::from_str(&serialize_final_commit().unwrap()).unwrap();
        assert_eq!(
            commit,
            serde_json::json!({
                "message_type": "input_audio_chunk",
                "audio_base_64": "",
                "commit": true,
                "sample_rate": 16000
            })
        );
    }

    #[test]
    fn constructs_sensitive_regional_requests_without_putting_key_in_uri() {
        let cases = [
            (ElevenLabsRegion::Global, "api.elevenlabs.io"),
            (ElevenLabsRegion::Us, "api.us.elevenlabs.io"),
            (ElevenLabsRegion::Eu, "api.eu.residency.elevenlabs.io"),
            (ElevenLabsRegion::India, "api.in.residency.elevenlabs.io"),
            (
                ElevenLabsRegion::Singapore,
                "api.sg.residency.elevenlabs.io",
            ),
        ];

        for (region, expected_host) in cases {
            let config = ElevenLabsConfig {
                region,
                language_code: Some("en".to_string()),
                ..ElevenLabsConfig::default()
            };
            let request = realtime_request(&config, API_KEY).unwrap();
            let uri = request.uri().to_string();
            let url = reqwest::Url::parse(&uri).unwrap();
            let query: BTreeMap<_, _> = url.query_pairs().into_owned().collect();
            let expected_query = BTreeMap::from([
                ("audio_format".to_string(), "pcm_16000".to_string()),
                ("commit_strategy".to_string(), "vad".to_string()),
                ("language_code".to_string(), "en".to_string()),
                ("min_silence_duration_ms".to_string(), "100".to_string()),
                ("min_speech_duration_ms".to_string(), "100".to_string()),
                ("model_id".to_string(), "scribe_v2_realtime".to_string()),
                ("no_verbatim".to_string(), "false".to_string()),
                ("vad_silence_threshold_secs".to_string(), "1.5".to_string()),
                ("vad_threshold".to_string(), "0.4".to_string()),
            ]);

            assert_eq!(url.scheme(), "wss");
            assert_eq!(url.host_str(), Some(expected_host));
            assert_eq!(url.path(), "/v1/speech-to-text/realtime");
            assert_eq!(query, expected_query);
            assert!(!uri.contains(API_KEY));
            let header = request.headers().get(&API_KEY_HEADER).unwrap();
            assert_eq!(header, API_KEY);
            assert!(header.is_sensitive());
        }
    }

    #[test]
    fn omits_absent_language_and_rejects_invalid_header_bytes() {
        let config = ElevenLabsConfig::default();
        let request = realtime_request(&config, API_KEY).unwrap();
        assert!(!request.uri().to_string().contains("language_code"));

        let error = realtime_request(&config, "invalid\nkey").unwrap_err();
        let surfaced = error.to_string();
        assert!(surfaced.contains("cannot be used in an HTTP header"));
        assert!(!surfaced.contains("invalid"));
    }

    #[test]
    fn parses_documented_realtime_message_fixtures() {
        let cases = [
            (
                r#"{
                    "message_type": "session_started",
                    "session_id": "session-123",
                    "config": {
                        "sample_rate": 16000,
                        "audio_format": "pcm_16000",
                        "language_code": "en",
                        "commit_strategy": "vad",
                        "vad_silence_threshold_secs": 1.5,
                        "vad_threshold": 0.4,
                        "min_speech_duration_ms": 100,
                        "min_silence_duration_ms": 100,
                        "model_id": "scribe_v2_realtime",
                        "enable_logging": true,
                        "include_timestamps": false,
                        "include_language_detection": false,
                        "keyterms": [],
                        "no_verbatim": false
                    }
                }"#,
                WireMessage::SessionStarted {
                    session_id: "session-123".to_string(),
                },
            ),
            (
                r#"{"message_type":"partial_transcript","text":"hello wor"}"#,
                WireMessage::Partial {
                    text: "hello wor".to_string(),
                },
            ),
            (
                r#"{"message_type":"committed_transcript","text":"hello world"}"#,
                WireMessage::Committed {
                    text: "hello world".to_string(),
                },
            ),
            (
                r#"{
                    "message_type": "committed_transcript_with_timestamps",
                    "text": "hello world",
                    "language_code": "en",
                    "words": [{
                        "text": "hello",
                        "start": 0.0,
                        "end": 0.4,
                        "type": "word",
                        "speaker_id": "speaker_0",
                        "logprob": -0.1,
                        "characters": ["h", "e", "l", "l", "o"]
                    }, {
                        "text": " ",
                        "start": 0.4,
                        "end": 0.45,
                        "type": "spacing",
                        "speaker_id": "speaker_0",
                        "logprob": 0.0,
                        "characters": [" "]
                    }, {
                        "text": "world",
                        "start": 0.45,
                        "end": 0.9,
                        "type": "word",
                        "speaker_id": "speaker_0",
                        "logprob": -0.2,
                        "characters": ["w", "o", "r", "l", "d"]
                    }]
                }"#,
                WireMessage::CommittedWithTimestamps {
                    text: "hello world".to_string(),
                },
            ),
            (
                r#"{"message_type":"auth_error","error":"invalid key elevenlabs-secret-key"}"#,
                WireMessage::ProviderError {
                    kind: "auth_error".to_string(),
                    message: "invalid key [REDACTED]".to_string(),
                },
            ),
            (
                r#"{"message_type":"future-elevenlabs-secret-key-event","value":42}"#,
                WireMessage::Unknown,
            ),
        ];

        for (fixture, expected) in cases {
            assert_eq!(parse_wire_message(fixture, API_KEY).unwrap(), expected);
        }
    }

    #[test]
    fn rejects_malformed_realtime_json_without_echoing_payload() {
        let fixture = r#"{"message_type":"partial_transcript","text":"elevenlabs-secret-key""#;
        let error = parse_wire_message(fixture, API_KEY).unwrap_err();
        let surfaced = error.to_string();

        assert!(surfaced.contains("Malformed ElevenLabs realtime message"));
        assert!(!surfaced.contains(API_KEY));
    }

    #[test]
    fn redacts_all_key_occurrences_and_bounds_provider_errors() {
        let long_error = format!("{API_KEY} middle {API_KEY} {}", "x".repeat(600));
        let surfaced = redact_and_bound(&long_error, API_KEY);

        assert!(!surfaced.contains(API_KEY));
        assert_eq!(surfaced.matches("[REDACTED]").count(), 2);
        assert_eq!(surfaced.chars().count(), MAX_PROVIDER_ERROR_CHARS + 3);
        assert!(surfaced.ends_with("..."));
    }

    #[test]
    fn reconciles_partial_extensions_repeats_divergence_and_unicode() {
        struct Case {
            name: &'static str,
            snapshots: &'static [&'static str],
            expected: Vec<EventView>,
        }
        let cases = [
            Case {
                name: "extensions",
                snapshots: &["hel", "hello"],
                expected: vec![
                    EventView::Partial("hel".to_string(), 0),
                    EventView::Partial("lo".to_string(), 0),
                ],
            },
            Case {
                name: "repeats",
                snapshots: &["hello", "hello"],
                expected: vec![EventView::Partial("hello".to_string(), 0)],
            },
            Case {
                name: "divergence then stable extension",
                snapshots: &["hello", "hullo", "hello world"],
                expected: vec![
                    EventView::Partial("hello".to_string(), 0),
                    EventView::Partial(" world".to_string(), 0),
                ],
            },
            Case {
                name: "unicode extension",
                snapshots: &["hé", "héllo"],
                expected: vec![
                    EventView::Partial("hé".to_string(), 0),
                    EventView::Partial("llo".to_string(), 0),
                ],
            },
        ];

        for case in cases {
            let mut reconciler = TranscriptReconciler::new(true);
            let messages = case.snapshots.iter().map(|text| WireMessage::Partial {
                text: (*text).to_string(),
            });
            assert_eq!(
                process_all(&mut reconciler, messages),
                case.expected,
                "{}",
                case.name
            );
        }
    }

    #[test]
    fn reconciles_exact_extended_corrected_and_unicode_commits() {
        struct Case {
            name: &'static str,
            partial: &'static str,
            committed: &'static str,
            expected_commit: EventView,
        }
        let cases = [
            Case {
                name: "extension",
                partial: "hello",
                committed: "hello world",
                expected_commit: EventView::Final(" world".to_string(), 0),
            },
            Case {
                name: "exact",
                partial: "hello",
                committed: "hello",
                expected_commit: EventView::Final(String::new(), 0),
            },
            Case {
                name: "correction",
                partial: "hello wurld",
                committed: "hello world",
                expected_commit: EventView::Replace(4, "orld".to_string(), 0),
            },
            Case {
                name: "shorter correction",
                partial: "hello world",
                committed: "hello",
                expected_commit: EventView::Replace(6, String::new(), 0),
            },
            Case {
                name: "unicode scalar correction",
                partial: "naïf",
                committed: "naïve",
                expected_commit: EventView::Replace(1, "ve".to_string(), 0),
            },
        ];

        for case in cases {
            let mut reconciler = TranscriptReconciler::new(true);
            let events = process_all(
                &mut reconciler,
                [
                    WireMessage::Partial {
                        text: case.partial.to_string(),
                    },
                    WireMessage::Committed {
                        text: case.committed.to_string(),
                    },
                ],
            );
            assert_eq!(
                events,
                vec![
                    EventView::Partial(case.partial.to_string(), 0),
                    case.expected_commit
                ],
                "{}",
                case.name
            );
            assert_eq!(reconciler.typed_partial, "", "{}", case.name);
            assert_eq!(reconciler.segment_id, 1, "{}", case.name);
        }
    }

    #[test]
    fn committed_only_mode_ignores_partials_and_uses_sequential_segments() {
        let mut reconciler = TranscriptReconciler::new(false);
        let events = process_all(
            &mut reconciler,
            [
                WireMessage::Partial {
                    text: "first dra".to_string(),
                },
                WireMessage::Committed {
                    text: "first draft".to_string(),
                },
                WireMessage::Partial {
                    text: "second dra".to_string(),
                },
                WireMessage::Committed {
                    text: "second draft".to_string(),
                },
            ],
        );

        assert_eq!(
            events,
            vec![
                EventView::Final("first draft".to_string(), 0),
                EventView::Final("second draft".to_string(), 1),
            ]
        );
        assert_eq!(reconciler.segment_id, 2);
    }

    #[test]
    fn partial_mode_clears_state_between_sequential_vad_segments() {
        let mut reconciler = TranscriptReconciler::new(true);
        let events = process_all(
            &mut reconciler,
            [
                WireMessage::Partial {
                    text: "first".to_string(),
                },
                WireMessage::Committed {
                    text: "first".to_string(),
                },
                WireMessage::Partial {
                    text: "second".to_string(),
                },
                WireMessage::Committed {
                    text: "second segment".to_string(),
                },
            ],
        );

        assert_eq!(
            events,
            vec![
                EventView::Partial("first".to_string(), 0),
                EventView::Final(String::new(), 0),
                EventView::Partial("second".to_string(), 1),
                EventView::Final(" segment".to_string(), 1),
            ]
        );
    }

    #[test]
    fn ignores_timestamp_followup_after_committed_transcript() {
        let mut reconciler = TranscriptReconciler::new(false);
        let events = process_all(
            &mut reconciler,
            [
                WireMessage::Committed {
                    text: "one transcript".to_string(),
                },
                WireMessage::CommittedWithTimestamps {
                    text: "one transcript".to_string(),
                },
            ],
        );

        assert_eq!(
            events,
            vec![EventView::Final("one transcript".to_string(), 0)]
        );
        assert_eq!(reconciler.segment_id, 1);
    }

    #[test]
    fn turns_redacted_provider_errors_into_streaming_errors() {
        let parsed = parse_wire_message(
            r#"{"message_type":"auth_error","error":"bad elevenlabs-secret-key"}"#,
            API_KEY,
        )
        .unwrap();
        let mut reconciler = TranscriptReconciler::new(false);
        let events = process_all(&mut reconciler, [parsed]);

        assert_eq!(
            events,
            vec![EventView::Error(
                "Transcription failed: ElevenLabs realtime auth_error: bad [REDACTED]".to_string()
            )]
        );
    }

    fn batch_config(region: ElevenLabsRegion) -> ElevenLabsConfig {
        ElevenLabsConfig {
            api_key: Some(API_KEY.to_string()),
            region,
            language_code: Some("en".to_string()),
            streaming: false,
            type_partials: false,
        }
    }

    fn batch_transcriber() -> ElevenLabsTranscriber {
        ElevenLabsTranscriber::new(&batch_config(ElevenLabsRegion::Global))
            .expect("valid test transcriber")
    }

    #[test]
    fn rejects_empty_and_short_batch_input_before_requesting() {
        let transcriber = batch_transcriber();

        let empty_error = transcriber.transcribe(&[]).unwrap_err().to_string();
        assert!(empty_error.contains("Empty audio buffer"));

        let short_error = transcriber
            .transcribe(&vec![0.0; MIN_BATCH_SAMPLES - 1])
            .unwrap_err()
            .to_string();
        assert!(short_error.contains("at least 100 ms"));
        assert!(short_error.contains("1600 samples"));
        assert!(transcriber.batch_client.get().is_none());
    }

    #[test]
    fn builds_pcm_batch_upload_with_fixed_fields() {
        let samples = [-2.0, -0.5, 0.0, 0.5, 2.0];
        let expected_pcm: Vec<u8> = [-32767i16, -16384, 0, 16384, 32767]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect();

        assert_eq!(
            BatchUpload::new(&samples, Some("en".to_string())),
            BatchUpload {
                pcm16le: expected_pcm,
                language_code: Some("en".to_string()),
                model_id: "scribe_v2",
                file_format: "pcm_s16le_16",
                tag_audio_events: false,
                diarize: false,
                webhook: false,
            }
        );
    }

    #[test]
    fn builds_sensitive_regional_batch_requests() {
        let cases = [
            (ElevenLabsRegion::Global, "api.elevenlabs.io"),
            (ElevenLabsRegion::Us, "api.us.elevenlabs.io"),
            (ElevenLabsRegion::Eu, "api.eu.residency.elevenlabs.io"),
            (ElevenLabsRegion::India, "api.in.residency.elevenlabs.io"),
            (
                ElevenLabsRegion::Singapore,
                "api.sg.residency.elevenlabs.io",
            ),
        ];
        let client = build_batch_client().unwrap();

        for (region, expected_host) in cases {
            let config = batch_config(region);
            let upload = BatchUpload::new(&[0.0; MIN_BATCH_SAMPLES], config.language_code.clone());
            let request = build_batch_request(&client, &config, API_KEY, upload).unwrap();
            let uri = request.url().as_str();

            assert_eq!(request.method(), reqwest::Method::POST);
            assert_eq!(request.url().scheme(), "https");
            assert_eq!(request.url().host_str(), Some(expected_host));
            assert_eq!(request.url().path(), "/v1/speech-to-text");
            assert!(request.url().query().is_none());
            assert!(!uri.contains(API_KEY));
            let header = request.headers().get(&API_KEY_HEADER).unwrap();
            assert_eq!(header, API_KEY);
            assert!(header.is_sensitive());
            assert!(request
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("multipart/form-data; boundary="));
        }
    }

    #[test]
    fn lazily_builds_and_reuses_batch_client() {
        let transcriber = batch_transcriber();
        assert!(transcriber.batch_client.get().is_none());

        let first = transcriber.batch_client().unwrap() as *const reqwest::Client;
        let second = transcriber.batch_client().unwrap() as *const reqwest::Client;

        assert_eq!(first, second);
        assert!(transcriber.batch_client.get().is_some());
        assert_eq!(BATCH_REQUEST_TIMEOUT, Duration::from_secs(120));
    }

    #[test]
    fn parses_documented_batch_response() {
        let fixture = r#"{
            "language_code": "en",
            "language_probability": 0.98,
            "text": "Hello world!",
            "words": [{
                "end": 0.5,
                "logprob": -0.124,
                "speaker_id": "speaker_1",
                "start": 0,
                "text": "Hello",
                "type": "word"
            }]
        }"#;

        assert_eq!(parse_batch_response(fixture).unwrap(), "Hello world!");
    }

    #[test]
    fn rejects_malformed_or_textless_batch_responses() {
        let malformed = parse_batch_response(r#"{"text":"unfinished""#)
            .unwrap_err()
            .to_string();
        let missing = parse_batch_response(r#"{"language_code":"en","words":[]}"#)
            .unwrap_err()
            .to_string();

        assert!(malformed.contains("Could not parse ElevenLabs batch response"));
        assert!(missing.contains("missing text"));
        assert!(!malformed.contains("unfinished"));
    }

    #[test]
    fn maps_batch_http_statuses_to_remediation() {
        let cases = [
            (
                StatusCode::UNAUTHORIZED,
                r#"{"detail":{"status":"invalid_api_key","message":"Invalid xi-api-key"}}"#,
                "Transcription failed: ElevenLabs authentication failed: check [elevenlabs] api_key or ELEVENLABS_API_KEY",
            ),
            (
                StatusCode::FORBIDDEN,
                r#"{"detail":{"status":"unaccepted_terms","message":"Forbidden"}}"#,
                "Transcription failed: ElevenLabs Scribe terms have not been accepted: accept them in the ElevenLabs dashboard",
            ),
            (
                StatusCode::FORBIDDEN,
                r#"{"detail":{"status":"forbidden","message":"API key terminated"}}"#,
                "Transcription failed: ElevenLabs authentication failed: check [elevenlabs] api_key or ELEVENLABS_API_KEY",
            ),
            (
                StatusCode::TOO_MANY_REQUESTS,
                r#"{"detail":{"status":"rate_limited","message":"Too many requests"}}"#,
                "Transcription failed: ElevenLabs rate limit exceeded: wait and try again",
            ),
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                r#"{"detail":{"status":"server_error","message":"temporary provider failure"}}"#,
                "Transcription failed: ElevenLabs batch request failed (HTTP 500 Internal Server Error): temporary provider failure",
            ),
        ];

        for (status, body, expected) in cases {
            assert_eq!(
                map_batch_status(status, body, API_KEY).to_string(),
                expected
            );
        }
    }

    #[test]
    fn redacts_and_bounds_api_key_in_batch_provider_errors() {
        let body = serde_json::json!({
            "detail": {
                "status": "server_error",
                "message": format!("failed for {API_KEY}: {}", "x".repeat(600)),
            }
        })
        .to_string();
        let surfaced = map_batch_status(StatusCode::BAD_GATEWAY, &body, API_KEY).to_string();

        assert!(!surfaced.contains(API_KEY));
        assert!(surfaced.contains("[REDACTED]"));
        assert!(surfaced.ends_with("..."));
        assert!(surfaced.chars().count() <= MAX_PROVIDER_ERROR_CHARS + 100);
    }
}
