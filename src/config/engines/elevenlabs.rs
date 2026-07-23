//! ElevenLabs Scribe engine configuration.

use serde::{Deserialize, Deserializer, Serialize};

use super::super::default_true;

/// ElevenLabs API region.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    Deserialize,
    Serialize,
    PartialEq,
    Eq,
    strum::Display,
    strum::EnumString,
)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum ElevenLabsRegion {
    /// Global API endpoint.
    #[default]
    Global,
    /// United States API endpoint.
    Us,
    /// European Union residency endpoint.
    Eu,
    /// India residency endpoint.
    India,
    /// Singapore residency endpoint.
    Singapore,
}

/// ElevenLabs Scribe cloud transcription configuration.
/// Requires: cargo build --features elevenlabs
#[derive(Clone, Deserialize, Serialize)]
pub struct ElevenLabsConfig {
    /// API key sent in the sensitive `xi-api-key` header.
    #[serde(default)]
    pub api_key: Option<String>,

    /// API endpoint region. Default: `global`.
    #[serde(default)]
    pub region: ElevenLabsRegion,

    /// Optional ISO 639-1 or ISO 639-3 language hint.
    #[serde(default, deserialize_with = "deserialize_language_code")]
    pub language_code: Option<String>,

    /// Use the realtime WebSocket API for dictation. Default: true.
    #[serde(default = "default_true")]
    pub streaming: bool,

    /// Type stable partial extensions before commit. Default: false.
    #[serde(default)]
    pub type_partials: bool,
}

impl ElevenLabsConfig {
    /// Set a normalized language hint, treating whitespace-only input as unset.
    pub fn set_language_code(&mut self, value: &str) {
        self.language_code = normalize_language_code(value);
    }
}

impl Default for ElevenLabsConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            region: ElevenLabsRegion::Global,
            language_code: None,
            streaming: true,
            type_partials: false,
        }
    }
}

impl std::fmt::Debug for ElevenLabsConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ElevenLabsConfig")
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("region", &self.region)
            .field("language_code", &self.language_code)
            .field("streaming", &self.streaming)
            .field("type_partials", &self.type_partials)
            .finish()
    }
}

fn normalize_language_code(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn deserialize_language_code<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<String>::deserialize(deserializer)?;
    Ok(value.as_deref().and_then(normalize_language_code))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_configuration_contract() {
        let config = ElevenLabsConfig::default();
        assert_eq!(
            (
                config.api_key,
                config.region,
                config.language_code,
                config.streaming,
                config.type_partials,
            ),
            (None, ElevenLabsRegion::Global, None, true, false)
        );

        let deserialized: ElevenLabsConfig = toml::from_str("").unwrap();
        assert_eq!(
            (
                deserialized.api_key,
                deserialized.region,
                deserialized.language_code,
                deserialized.streaming,
                deserialized.type_partials,
            ),
            (None, ElevenLabsRegion::Global, None, true, false)
        );
    }

    #[test]
    fn parses_every_region_string() {
        let cases = [
            ("global", ElevenLabsRegion::Global),
            ("us", ElevenLabsRegion::Us),
            ("eu", ElevenLabsRegion::Eu),
            ("india", ElevenLabsRegion::India),
            ("singapore", ElevenLabsRegion::Singapore),
        ];

        for (value, expected) in cases {
            assert_eq!(value.parse::<ElevenLabsRegion>().unwrap(), expected);
            let parsed: ElevenLabsConfig =
                toml::from_str(&format!("region = \"{value}\"")).unwrap();
            assert_eq!(parsed.region, expected);
        }
    }

    #[test]
    fn rejects_invalid_region_string() {
        assert!("mars".parse::<ElevenLabsRegion>().is_err());
        assert!(toml::from_str::<ElevenLabsConfig>("region = \"mars\"").is_err());
    }

    #[test]
    fn trims_language_code_and_treats_empty_as_unset() {
        let config: ElevenLabsConfig = toml::from_str("language_code = \"  en  \"").unwrap();
        assert_eq!(config.language_code.as_deref(), Some("en"));

        let config: ElevenLabsConfig = toml::from_str("language_code = \"   \"").unwrap();
        assert_eq!(config.language_code, None);
    }

    #[test]
    fn debug_output_redacts_api_key() {
        let config = ElevenLabsConfig {
            api_key: Some("super-secret-key".to_string()),
            ..ElevenLabsConfig::default()
        };
        let debug = format!("{config:?}");
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("super-secret-key"));
    }
}
