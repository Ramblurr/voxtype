//! ElevenLabs Scribe protocol primitives.
//!
//! This module keeps the provider's wire format, audio framing, request
//! construction, and transcript reconciliation separate from the transport
//! lifecycle. The batch and realtime adapters build on these helpers.

#![allow(
    dead_code,
    reason = "protocol primitives are wired into transports in the following implementation tasks"
)]

use base64::Engine as _;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tokio_tungstenite::tungstenite::{
    client::IntoClientRequest,
    http::{header::HeaderName, HeaderValue, Request},
};

use super::{SegmentId, StreamingEvent};
use crate::{
    config::{ElevenLabsConfig, ElevenLabsRegion},
    error::TranscribeError,
};

const REALTIME_MODEL: &str = "scribe_v2_realtime";
const REALTIME_AUDIO_FORMAT: &str = "pcm_16000";
const SAMPLE_RATE: u32 = 16_000;
const COMMIT_STRATEGY: &str = "vad";
const VAD_SILENCE_THRESHOLD_SECS: &str = "1.5";
const VAD_THRESHOLD: &str = "0.4";
const MIN_SPEECH_DURATION_MS: &str = "100";
const MIN_SILENCE_DURATION_MS: &str = "100";
const FRAME_DURATION_MS: usize = 100;
const SAMPLES_PER_FRAME: usize = SAMPLE_RATE as usize * FRAME_DURATION_MS / 1_000;
const BYTES_PER_FRAME: usize = SAMPLES_PER_FRAME * size_of::<i16>();
const MAX_PROVIDER_ERROR_CHARS: usize = 512;
const API_KEY_HEADER: HeaderName = HeaderName::from_static("xi-api-key");

fn realtime_origin(region: ElevenLabsRegion) -> &'static str {
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
        realtime_origin(config.region)
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
}
