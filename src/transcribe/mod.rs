//! Speech-to-text transcription module
//!
//! Provides transcription via:
//! - Local whisper.cpp inference (whisper-rs crate)
//! - Remote OpenAI-compatible Whisper API (whisper.cpp server, OpenAI, etc.)
//! - CLI subprocess using whisper-cli (fallback for glibc 2.42+ compatibility)
//! - Subprocess isolation for GPU memory release
//! - Optionally ElevenLabs Scribe cloud transcription (when `elevenlabs` feature is enabled)
//! - Optionally NVIDIA Parakeet via ONNX Runtime (when `parakeet` feature is enabled)
//! - Optionally Moonshine via ONNX Runtime (when `moonshine` feature is enabled)
//! - Optionally SenseVoice via ONNX Runtime (when `sensevoice` feature is enabled)
//! - Optionally Paraformer via ONNX Runtime (when `paraformer` feature is enabled)
//! - Optionally Dolphin via ONNX Runtime (when `dolphin` feature is enabled)
//! - Optionally Omnilingual via ONNX Runtime (when `omnilingual` feature is enabled)

pub mod cli;
#[cfg(feature = "elevenlabs")]
mod elevenlabs;
#[cfg(feature = "parakeet")]
pub mod parakeet_streaming;
pub mod remote;
#[cfg(feature = "soniox")]
pub mod soniox;
pub mod streaming;
pub mod subprocess;
pub mod whisper;
pub mod worker;

pub use streaming::{SegmentId, StreamHandle, StreamingEvent, StreamingTranscriber};

/// Shared log-mel filterbank feature extraction for ONNX-based ASR engines
#[cfg(any(
    feature = "sensevoice",
    feature = "paraformer",
    feature = "dolphin",
    feature = "omnilingual",
    feature = "cohere",
))]
pub mod fbank;

/// Shared GPU execution-provider registration for ONNX-based engines.
#[cfg(feature = "onnx-common")]
pub mod onnx_ep;

/// Shared CTC greedy decoder for CTC-based ASR engines
#[cfg(any(
    feature = "sensevoice",
    feature = "paraformer",
    feature = "dolphin",
    feature = "omnilingual",
    feature = "cohere",
))]
pub mod ctc;

#[cfg(feature = "parakeet")]
pub mod parakeet;

#[cfg(feature = "moonshine")]
pub mod moonshine;

#[cfg(feature = "sensevoice")]
pub mod sensevoice;

#[cfg(feature = "paraformer")]
pub mod paraformer;

#[cfg(feature = "dolphin")]
pub mod dolphin;

#[cfg(feature = "omnilingual")]
pub mod omnilingual;

/// Cohere Transcribe backend (proof-of-concept, not wired into factory/CLI/config).
/// See `src/transcribe/cohere.rs` for usage.
#[cfg(feature = "cohere")]
pub mod cohere;

/// Cohere-specific log-mel feature extractor (NeMo conventions, 128 mels).
#[cfg(feature = "cohere")]
pub mod cohere_fbank;

use crate::config::{Config, TranscriptionEngine, WhisperConfig, WhisperMode};
use crate::error::TranscribeError;
use crate::setup::gpu;

/// A timed segment from transcription (word or sentence level)
#[derive(Debug, Clone)]
pub struct TimedSegment {
    pub text: String,
    /// Start time in seconds relative to the audio input
    pub start_secs: f32,
    /// End time in seconds relative to the audio input
    pub end_secs: f32,
}

/// Trait for speech-to-text implementations
pub trait Transcriber: Send + Sync {
    /// Transcribe audio samples to text
    /// Input: f32 samples, mono, 16kHz
    fn transcribe(&self, samples: &[f32]) -> Result<String, TranscribeError>;

    /// Transcribe with word-level timestamps.
    /// Default implementation falls back to transcribe() with a single segment.
    fn transcribe_timed(&self, samples: &[f32]) -> Result<Vec<TimedSegment>, TranscribeError> {
        let text = self.transcribe(samples)?;
        let duration = samples.len() as f32 / 16000.0;
        if text.is_empty() {
            Ok(vec![])
        } else {
            Ok(vec![TimedSegment {
                text,
                start_secs: 0.0,
                end_secs: duration,
            }])
        }
    }

    /// Prepare for transcription (optional, called when recording starts)
    ///
    /// For subprocess-based transcribers, this spawns the worker process
    /// and begins loading the model while the user is still speaking.
    /// This hides model loading latency behind recording time.
    ///
    /// Default implementation does nothing (for transcribers that don't
    /// benefit from preparation, like those with preloaded models).
    fn prepare(&self) {
        // Default: no-op
    }

    /// Streaming-capable view of this transcriber, if it supports streaming.
    ///
    /// Returns `None` by default. Streaming-capable backends override this to
    /// return `Some(self)` (or some other implementor of [`StreamingTranscriber`]).
    /// The daemon consults this when `[transcribe] streaming = true` is set in
    /// config to decide between batch and streaming pipelines.
    fn as_streaming(&self) -> Option<&dyn StreamingTranscriber> {
        None
    }

    /// Two-letter language code detected (or selected) for the most recent
    /// transcription, if the backend tracks it.
    ///
    /// This is used by output methods that benefit from a layout hint
    /// (notably [`crate::output::eitype::EitypeOutput`] and
    /// [`crate::output::dotool::DotoolOutput`]). It is set by backends with
    /// language auto-detection or explicit single-language mode; backends
    /// without language awareness return `None`.
    ///
    /// The default implementation returns `None`. Backends override this when
    /// they track the language used for the previous call to
    /// [`Self::transcribe`].
    fn last_detected_language(&self) -> Option<String> {
        None
    }
}

/// Factory function to create transcriber based on configured engine
pub fn create_transcriber(config: &Config) -> Result<Box<dyn Transcriber>, TranscribeError> {
    match config.engine {
        TranscriptionEngine::Whisper => create_whisper_transcriber(&config.whisper),
        #[cfg(feature = "parakeet")]
        TranscriptionEngine::Parakeet => {
            let parakeet_config = config.parakeet.as_ref().ok_or_else(|| {
                TranscribeError::InitFailed(
                    "Parakeet engine selected but [parakeet] config section is missing".to_string(),
                )
            })?;
            if parakeet_config.streaming {
                Ok(Box::new(
                    parakeet_streaming::ParakeetStreamingTranscriber::new(parakeet_config)?,
                ))
            } else {
                Ok(Box::new(parakeet::ParakeetTranscriber::new(
                    parakeet_config,
                )?))
            }
        }
        #[cfg(not(feature = "parakeet"))]
        TranscriptionEngine::Parakeet => Err(TranscribeError::InitFailed(
            "Parakeet engine requested but voxtype was not compiled with --features parakeet"
                .to_string(),
        )),
        #[cfg(feature = "moonshine")]
        TranscriptionEngine::Moonshine => {
            let moonshine_config = config.moonshine.as_ref().ok_or_else(|| {
                TranscribeError::InitFailed(
                    "Moonshine engine selected but [moonshine] config section is missing"
                        .to_string(),
                )
            })?;
            Ok(Box::new(moonshine::MoonshineTranscriber::new(
                moonshine_config,
            )?))
        }
        #[cfg(not(feature = "moonshine"))]
        TranscriptionEngine::Moonshine => Err(TranscribeError::InitFailed(
            "Moonshine engine requested but voxtype was not compiled with --features moonshine"
                .to_string(),
        )),
        #[cfg(feature = "sensevoice")]
        TranscriptionEngine::SenseVoice => {
            let sensevoice_config = config.sensevoice.as_ref().ok_or_else(|| {
                TranscribeError::InitFailed(
                    "SenseVoice engine selected but [sensevoice] config section is missing"
                        .to_string(),
                )
            })?;
            Ok(Box::new(sensevoice::SenseVoiceTranscriber::new(
                sensevoice_config,
            )?))
        }
        #[cfg(not(feature = "sensevoice"))]
        TranscriptionEngine::SenseVoice => Err(TranscribeError::InitFailed(
            "SenseVoice engine requested but voxtype was not compiled with --features sensevoice"
                .to_string(),
        )),
        #[cfg(feature = "paraformer")]
        TranscriptionEngine::Paraformer => {
            let cfg = config.paraformer.as_ref().ok_or_else(|| {
                TranscribeError::InitFailed(
                    "Paraformer engine selected but [paraformer] config section is missing"
                        .to_string(),
                )
            })?;
            Ok(Box::new(paraformer::ParaformerTranscriber::new(cfg)?))
        }
        #[cfg(not(feature = "paraformer"))]
        TranscriptionEngine::Paraformer => Err(TranscribeError::InitFailed(
            "Paraformer engine requested but voxtype was not compiled with --features paraformer"
                .to_string(),
        )),
        #[cfg(feature = "dolphin")]
        TranscriptionEngine::Dolphin => {
            let cfg = config.dolphin.as_ref().ok_or_else(|| {
                TranscribeError::InitFailed(
                    "Dolphin engine selected but [dolphin] config section is missing".to_string(),
                )
            })?;
            Ok(Box::new(dolphin::DolphinTranscriber::new(cfg)?))
        }
        #[cfg(not(feature = "dolphin"))]
        TranscriptionEngine::Dolphin => Err(TranscribeError::InitFailed(
            "Dolphin engine requested but voxtype was not compiled with --features dolphin"
                .to_string(),
        )),
        #[cfg(feature = "omnilingual")]
        TranscriptionEngine::Omnilingual => {
            let cfg = config.omnilingual.as_ref().ok_or_else(|| {
                TranscribeError::InitFailed(
                    "Omnilingual engine selected but [omnilingual] config section is missing"
                        .to_string(),
                )
            })?;
            Ok(Box::new(omnilingual::OmnilingualTranscriber::new(cfg)?))
        }
        #[cfg(not(feature = "omnilingual"))]
        TranscriptionEngine::Omnilingual => Err(TranscribeError::InitFailed(
            "Omnilingual engine requested but voxtype was not compiled with --features omnilingual"
                .to_string(),
        )),
        #[cfg(feature = "cohere")]
        TranscriptionEngine::Cohere => {
            let cfg = config.cohere.as_ref().ok_or_else(|| {
                TranscribeError::InitFailed(
                    "Cohere engine selected but [cohere] config section is missing".to_string(),
                )
            })?;
            Ok(Box::new(cohere::CohereTranscriber::new(cfg)?))
        }
        #[cfg(not(feature = "cohere"))]
        TranscriptionEngine::Cohere => Err(TranscribeError::InitFailed(
            "Cohere engine requested but voxtype was not compiled with --features cohere"
                .to_string(),
        )),
        #[cfg(feature = "soniox")]
        TranscriptionEngine::Soniox => {
            let cfg = config.soniox.as_ref().ok_or_else(|| {
                TranscribeError::InitFailed(
                    "Soniox engine selected but [soniox] config section is missing".to_string(),
                )
            })?;
            Ok(Box::new(soniox::SonioxTranscriber::new(cfg.clone())?))
        }
        #[cfg(not(feature = "soniox"))]
        TranscriptionEngine::Soniox => Err(TranscribeError::InitFailed(
            "Soniox engine requested but voxtype was not compiled with --features soniox"
                .to_string(),
        )),
        #[cfg(feature = "elevenlabs")]
        TranscriptionEngine::ElevenLabs => {
            let provider_config = config.elevenlabs.clone().unwrap_or_default();
            Ok(Box::new(elevenlabs::ElevenLabsTranscriber::new(
                &provider_config,
            )?))
        }
        #[cfg(not(feature = "elevenlabs"))]
        TranscriptionEngine::ElevenLabs => Err(TranscribeError::InitFailed(
            "ElevenLabs engine requested but voxtype was not compiled with --features elevenlabs"
                .to_string(),
        )),
    }
}

pub(crate) fn validate_elevenlabs_realtime_output(
    output: &crate::config::OutputConfig,
    profile_post_process: bool,
) -> Result<(), TranscribeError> {
    let incompatible_setting = if output.mode != crate::config::OutputMode::Type {
        Some("output.mode")
    } else if output.auto_submit {
        Some("output.auto_submit")
    } else if output.append_text.is_some() {
        Some("output.append_text")
    } else if output.pre_recording_command.is_some() {
        Some("output.pre_recording_command")
    } else if output.pre_output_command.is_some() {
        Some("output.pre_output_command")
    } else if output.post_output_command.is_some() {
        Some("output.post_output_command")
    } else if output.post_process.is_some() {
        Some("output.post_process")
    } else if profile_post_process {
        Some("profile.post_process_command")
    } else {
        None
    };

    if let Some(setting) = incompatible_setting {
        return Err(TranscribeError::ConfigError(format!(
            "ElevenLabs realtime transcription does not support {setting}; \
             set [elevenlabs] streaming = false to use batch transcription"
        )));
    }

    Ok(())
}

/// Factory function to create Whisper transcriber (local or remote)
pub fn create_whisper_transcriber(
    config: &WhisperConfig,
) -> Result<Box<dyn Transcriber>, TranscribeError> {
    create_transcriber_with_config_path(config, None)
}

/// Factory function to create transcriber with optional config path
/// The config path is passed to subprocess transcriber for isolated GPU execution
pub fn create_transcriber_with_config_path(
    config: &WhisperConfig,
    config_path: Option<std::path::PathBuf>,
) -> Result<Box<dyn Transcriber>, TranscribeError> {
    // Apply GPU selection from VOXTYPE_VULKAN_DEVICE environment variable
    // This sets VK_LOADER_DRIVERS_SELECT to filter Vulkan drivers
    if let Some(vendor) = gpu::apply_gpu_selection() {
        tracing::info!(
            "GPU selection: {} (via VOXTYPE_VULKAN_DEVICE)",
            vendor.display_name()
        );
    }

    match config.effective_mode() {
        WhisperMode::Local => {
            if config.gpu_isolation {
                tracing::info!(
                    "Using subprocess-isolated whisper transcription (gpu_isolation=true)"
                );
                Ok(Box::new(subprocess::SubprocessTranscriber::new(
                    config,
                    config_path,
                )?))
            } else {
                tracing::info!("Using local whisper transcription mode");
                Ok(Box::new(whisper::WhisperTranscriber::new(config)?))
            }
        }
        WhisperMode::Remote => {
            tracing::info!("Using remote whisper transcription mode");
            Ok(Box::new(remote::RemoteTranscriber::new(config)?))
        }
        WhisperMode::Cli => {
            tracing::info!("Using whisper-cli subprocess backend");
            Ok(Box::new(cli::CliTranscriber::new(config)?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn factory_error(config: &Config) -> TranscribeError {
        match create_transcriber(config) {
            Ok(_) => panic!("expected transcriber construction to fail"),
            Err(error) => error,
        }
    }

    #[cfg(not(feature = "elevenlabs"))]
    #[test]
    fn elevenlabs_factory_reports_feature_disabled_remediation() {
        let config = Config {
            engine: TranscriptionEngine::ElevenLabs,
            ..Config::default()
        };

        let error = factory_error(&config).to_string();

        assert!(error.contains("--features elevenlabs"));
    }

    #[cfg(feature = "elevenlabs")]
    mod elevenlabs {
        use super::*;
        use crate::config::{ElevenLabsConfig, OutputMode, PostProcessConfig};

        const API_KEY: &str = "test-elevenlabs-key";

        fn config(streaming: bool) -> Config {
            Config {
                engine: TranscriptionEngine::ElevenLabs,
                elevenlabs: Some(ElevenLabsConfig {
                    api_key: Some(API_KEY.to_string()),
                    streaming,
                    ..ElevenLabsConfig::default()
                }),
                ..Config::default()
            }
        }

        #[test]
        fn factory_materializes_defaults_before_reporting_missing_api_key() {
            let config = Config {
                engine: TranscriptionEngine::ElevenLabs,
                elevenlabs: None,
                ..Config::default()
            };

            let error = factory_error(&config).to_string();

            assert!(error.contains(
                "ElevenLabs API key required: set [elevenlabs] api_key or ELEVENLABS_API_KEY"
            ));
            assert!(!error.contains("config section is missing"));
        }

        #[test]
        fn factory_rejects_api_keys_that_are_invalid_http_headers() {
            let invalid_key = "secret\nheader";
            let mut config = config(true);
            config.elevenlabs.as_mut().unwrap().api_key = Some(invalid_key.to_string());

            let error = factory_error(&config).to_string();

            assert!(error.contains("cannot be used in an HTTP header"));
            assert!(!error.contains(invalid_key));
        }

        #[test]
        fn factory_exposes_realtime_only_when_streaming_is_enabled() {
            let realtime =
                create_transcriber(&config(true)).expect("realtime factory construction");
            let batch = create_transcriber(&config(false)).expect("batch factory construction");

            assert!(realtime.as_streaming().is_some());
            assert!(batch.as_streaming().is_none());
        }

        #[test]
        fn realtime_preflight_rejects_unsafe_incremental_output_settings() {
            fn clipboard(output: &mut crate::config::OutputConfig) {
                output.mode = OutputMode::Clipboard;
            }
            fn auto_submit(output: &mut crate::config::OutputConfig) {
                output.auto_submit = true;
            }
            fn append_text(output: &mut crate::config::OutputConfig) {
                output.append_text = Some(" ".to_string());
            }
            fn pre_recording_hook(output: &mut crate::config::OutputConfig) {
                output.pre_recording_command = Some("prepare-output".to_string());
            }
            fn pre_output_hook(output: &mut crate::config::OutputConfig) {
                output.pre_output_command = Some("prepare-output".to_string());
            }
            fn post_output_hook(output: &mut crate::config::OutputConfig) {
                output.post_output_command = Some("restore-output".to_string());
            }
            fn post_process(output: &mut crate::config::OutputConfig) {
                output.post_process = Some(PostProcessConfig {
                    command: "clean-transcript".to_string(),
                    timeout_ms: 30_000,
                    trim: true,
                    fallback_on_empty: true,
                });
            }

            type UnsafeSetting = (&'static str, fn(&mut crate::config::OutputConfig));
            let cases: [UnsafeSetting; 7] = [
                ("output.mode", clipboard),
                ("auto_submit", auto_submit),
                ("append_text", append_text),
                ("pre_recording_command", pre_recording_hook),
                ("pre_output_command", pre_output_hook),
                ("post_output_command", post_output_hook),
                ("post_process", post_process),
            ];

            for (setting, make_unsafe) in cases {
                let mut output = crate::config::OutputConfig::default();
                make_unsafe(&mut output);

                let error = validate_elevenlabs_realtime_output(&output, false)
                    .expect_err("unsafe realtime output must be rejected")
                    .to_string();

                assert!(
                    error.contains(setting),
                    "error for {setting} did not name the unsafe setting: {error}"
                );
                assert!(
                    error.contains("streaming = false"),
                    "error for {setting} omitted the batch-mode remedy: {error}"
                );
            }

            let error =
                validate_elevenlabs_realtime_output(&crate::config::OutputConfig::default(), true)
                    .expect_err("profile post-processing must be rejected")
                    .to_string();
            assert!(error.contains("profile.post_process_command"));
            assert!(error.contains("streaming = false"));
        }

        #[test]
        fn factory_leaves_realtime_output_validation_to_daemon_batch_context() {
            let mut config = config(true);
            config.output.mode = OutputMode::Clipboard;
            config.output.auto_submit = true;
            config.output.post_process = Some(PostProcessConfig {
                command: "clean-transcript".to_string(),
                timeout_ms: 30_000,
                trim: true,
                fallback_on_empty: true,
            });

            let transcriber =
                create_transcriber(&config).expect("batch-only callers can construct the provider");

            assert!(transcriber.as_streaming().is_some());
        }

        #[test]
        fn batch_factory_allows_settings_rejected_for_realtime() {
            let mut config = config(false);
            config.output.mode = OutputMode::Clipboard;
            config.output.auto_submit = true;
            config.output.append_text = Some(" ".to_string());
            config.output.pre_recording_command = Some("prepare-output".to_string());
            config.output.pre_output_command = Some("prepare-output".to_string());
            config.output.post_output_command = Some("restore-output".to_string());
            config.output.post_process = Some(PostProcessConfig {
                command: "clean-transcript".to_string(),
                timeout_ms: 30_000,
                trim: true,
                fallback_on_empty: true,
            });

            let transcriber = create_transcriber(&config).expect("batch factory construction");

            assert!(transcriber.as_streaming().is_none());
        }
    }
}
