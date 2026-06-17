//! MPRIS media player control via direct D-Bus.
//!
//! Pauses playing players when recording starts and resumes only the
//! players we actually paused. The previous implementation shelled out
//! to `playerctl`, which had two failure modes that hit real users:
//!
//!   * `playerctl -l` silently filters out some MPRIS-compliant players
//!     (e.g. `cliamp`) even when they expose a complete MPRIS interface
//!     on the bus, so voxtype never tried to pause them.
//!   * The resume path called `playerctl --player <stored-name> play`
//!     and ignored the exit code. If the player's bus name had gone
//!     away during the dictation window (Chromium's PID-suffixed names
//!     are particularly fragile), the resume silently no-opped and the
//!     user's music stayed paused. See Omarchy issue #6029.
//!
//! Talking D-Bus directly via zbus fixes both: we enumerate all owned
//! names matching `org.mpris.MediaPlayer2.*` ourselves and surface real
//! errors on resume.

/// PulseAudio/PipeWire stream volume captured before media ducking.
#[derive(Debug, Clone)]
pub struct DuckedMediaStream {
    index: u32,
    volumes: Vec<String>,
}

#[cfg(target_os = "linux")]
mod imp {
    use super::DuckedMediaStream;
    use serde_json::Value;
    use tokio::process::Command;
    use tracing::{debug, warn};
    use zbus::{fdo::DBusProxy, Connection, Proxy};

    const MPRIS_PREFIX: &str = "org.mpris.MediaPlayer2.";
    const MPRIS_PATH: &str = "/org/mpris/MediaPlayer2";
    const MPRIS_IFACE: &str = "org.mpris.MediaPlayer2.Player";

    /// Pause all currently playing MPRIS media players.
    /// Returns the bus names of players that were paused so they can be resumed.
    /// Suffixes in `ignored` are matched against the part after the MPRIS prefix
    /// either exactly or as a `<entry>.<instance>` prefix (so `"chromium"`
    /// matches `chromium.instance1872063`).
    pub async fn pause_playing_players(ignored: &[String]) -> Vec<String> {
        let conn = match Connection::session().await {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to connect to session bus: {e}");
                return Vec::new();
            }
        };

        let players = match list_mpris_players(&conn, ignored).await {
            Ok(p) => p,
            Err(e) => {
                warn!("Failed to enumerate MPRIS players: {e}");
                return Vec::new();
            }
        };

        if players.is_empty() {
            debug!("No MPRIS players found");
            return Vec::new();
        }

        let mut paused = Vec::new();
        for bus_name in players {
            match player_status(&conn, &bus_name).await {
                Ok(status) if status == "Playing" => {
                    debug!(player = %bus_name, "Pausing media player");
                    match call_player(&conn, &bus_name, "Pause").await {
                        Ok(()) => paused.push(bus_name),
                        Err(e) => warn!(player = %bus_name, "Pause failed: {e}"),
                    }
                }
                Ok(status) => {
                    debug!(player = %bus_name, %status, "Skipping non-playing player")
                }
                Err(e) => debug!(player = %bus_name, "Status query failed: {e}"),
            }
        }

        if !paused.is_empty() {
            debug!("Paused {} media player(s)", paused.len());
        }
        paused
    }

    /// Resume previously-paused MPRIS media players.
    pub async fn resume_players(players: Vec<String>) {
        if players.is_empty() {
            return;
        }
        let conn = match Connection::session().await {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to connect to session bus for resume: {e}");
                return;
            }
        };

        let mut resumed = 0usize;
        for bus_name in &players {
            debug!(player = %bus_name, "Resuming media player");
            match call_player(&conn, bus_name, "Play").await {
                Ok(()) => resumed += 1,
                Err(e) => warn!(
                    player = %bus_name,
                    "Resume failed (player may have exited during dictation): {e}"
                ),
            }
        }

        debug!("Resumed {}/{} media player(s)", resumed, players.len());
    }

    /// Lower active sink-input volumes and return their original volumes.
    ///
    /// This intentionally does not use MPRIS transport controls. It can be
    /// enabled alongside `pause_media`, but when both are enabled the pause
    /// feature keeps its existing start/stop behavior and ducking only manages
    /// stream volume.
    pub async fn duck_playing_audio(volume_percent: u8) -> Vec<DuckedMediaStream> {
        let streams = match list_active_sink_inputs().await {
            Ok(streams) => streams,
            Err(e) => {
                warn!("Failed to enumerate audio streams for media ducking: {e}");
                return Vec::new();
            }
        };

        if streams.is_empty() {
            debug!("No active audio streams found for media ducking");
            return Vec::new();
        }

        let target = format!("{}%", volume_percent.min(150));
        let mut ducked = Vec::new();
        for stream in streams {
            debug!(stream = stream.index, volume = %target, "Ducking media stream");
            match Command::new("pactl")
                .arg("set-sink-input-volume")
                .arg(stream.index.to_string())
                .arg(&target)
                .status()
                .await
            {
                Ok(status) if status.success() => ducked.push(stream),
                Ok(status) => warn!(
                    stream = stream.index,
                    "Media ducking failed with status: {status}"
                ),
                Err(e) => warn!(
                    stream = stream.index,
                    "Failed to run pactl for media ducking: {e}"
                ),
            }
        }

        ducked
    }

    /// Restore stream volumes captured by `duck_playing_audio`.
    pub async fn restore_ducked_audio(streams: Vec<DuckedMediaStream>) {
        if streams.is_empty() {
            return;
        }

        let total = streams.len();
        let mut restored = 0usize;
        for stream in streams {
            debug!(stream = stream.index, "Restoring ducked media stream");
            match Command::new("pactl")
                .arg("set-sink-input-volume")
                .arg(stream.index.to_string())
                .args(&stream.volumes)
                .status()
                .await
            {
                Ok(status) if status.success() => restored += 1,
                Ok(status) => warn!(
                    stream = stream.index,
                    "Media duck restore failed with status: {status}"
                ),
                Err(e) => warn!(
                    stream = stream.index,
                    "Failed to run pactl for media duck restore: {e}"
                ),
            }
        }

        debug!("Restored {restored}/{total} ducked media stream(s)");
    }

    async fn list_active_sink_inputs() -> Result<Vec<DuckedMediaStream>, String> {
        let output = Command::new("pactl")
            .args(["-f", "json", "list", "sink-inputs"])
            .output()
            .await
            .map_err(|e| e.to_string())?;

        if !output.status.success() {
            return Err(format!("pactl exited with {}", output.status));
        }

        let value: Value = serde_json::from_slice(&output.stdout).map_err(|e| e.to_string())?;
        Ok(parse_sink_inputs(&value))
    }

    fn parse_sink_inputs(value: &Value) -> Vec<DuckedMediaStream> {
        let Some(streams) = value.as_array() else {
            return Vec::new();
        };

        streams
            .iter()
            .filter(|stream| {
                !stream.get("mute").and_then(Value::as_bool).unwrap_or(false)
                    && !stream
                        .get("corked")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
            })
            .filter_map(|stream| {
                let index = value_as_u32(stream.get("index")?)?;
                let volumes = stream_volumes(stream);
                if volumes.is_empty() {
                    None
                } else {
                    Some(DuckedMediaStream { index, volumes })
                }
            })
            .collect()
    }

    fn value_as_u32(value: &Value) -> Option<u32> {
        value
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .or_else(|| value.as_str()?.parse().ok())
    }

    fn stream_volumes(stream: &Value) -> Vec<String> {
        let Some(volume) = stream.get("volume").and_then(Value::as_object) else {
            return Vec::new();
        };
        let mut channels: Vec<_> = volume.iter().collect();
        channels.sort_by(|a, b| a.0.cmp(b.0));
        channels
            .into_iter()
            .filter_map(|(_, channel)| {
                channel
                    .get("value_percent")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
            .collect()
    }

    async fn list_mpris_players(
        conn: &Connection,
        ignored: &[String],
    ) -> zbus::Result<Vec<String>> {
        let dbus = DBusProxy::new(conn).await?;
        let names = dbus.list_names().await?;
        let mut out = Vec::new();
        for n in names {
            let s: &str = n.as_str();
            if !s.starts_with(MPRIS_PREFIX) {
                continue;
            }
            let suffix = &s[MPRIS_PREFIX.len()..];
            // Skip playerctld's aggregator: pausing it would double-fire
            // pause across every underlying player.
            if suffix == "playerctld" {
                continue;
            }
            if ignored.iter().any(|ig| {
                suffix == ig
                    || suffix.starts_with(ig) && suffix.as_bytes().get(ig.len()) == Some(&b'.')
            }) {
                debug!(player = %suffix, "Ignored by config");
                continue;
            }
            out.push(s.to_string());
        }
        Ok(out)
    }

    async fn player_status(conn: &Connection, bus_name: &str) -> zbus::Result<String> {
        let proxy = Proxy::new(conn, bus_name, MPRIS_PATH, MPRIS_IFACE).await?;
        proxy.get_property::<String>("PlaybackStatus").await
    }

    async fn call_player(
        conn: &Connection,
        bus_name: &str,
        method: &'static str,
    ) -> zbus::Result<()> {
        let proxy = Proxy::new(conn, bus_name, MPRIS_PATH, MPRIS_IFACE).await?;
        proxy.call::<_, _, ()>(method, &()).await
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[tokio::test]
        async fn list_skips_non_mpris_and_playerctld() {
            // We can't easily fake a real session bus, but we can at least
            // exercise the filter logic by running against the live bus and
            // confirming playerctld and ignored entries never appear.
            let Ok(conn) = Connection::session().await else {
                eprintln!("skip: no session bus");
                return;
            };
            let players = list_mpris_players(&conn, &["chromium".to_string()])
                .await
                .unwrap_or_default();
            for p in players {
                assert!(p.starts_with(MPRIS_PREFIX));
                let suffix = &p[MPRIS_PREFIX.len()..];
                assert_ne!(suffix, "playerctld");
                assert!(
                    !(suffix == "chromium" || suffix.starts_with("chromium.")),
                    "ignored prefix leaked: {p}"
                );
            }
        }

        #[test]
        fn parses_active_sink_inputs_with_channel_volumes() {
            let value: Value = serde_json::json!([
                {
                    "index": 42,
                    "mute": false,
                    "corked": false,
                    "volume": {
                        "front-right": { "value_percent": "80%" },
                        "front-left": { "value_percent": "75%" }
                    }
                },
                {
                    "index": 43,
                    "mute": true,
                    "corked": false,
                    "volume": {
                        "mono": { "value_percent": "100%" }
                    }
                }
            ]);

            let streams = parse_sink_inputs(&value);
            assert_eq!(streams.len(), 1);
            assert_eq!(streams[0].index, 42);
            assert_eq!(streams[0].volumes, vec!["75%", "80%"]);
        }
    }
}

#[cfg(target_os = "linux")]
pub use imp::{duck_playing_audio, pause_playing_players, restore_ducked_audio, resume_players};

// On non-Linux targets MPRIS doesn't apply. Keep the public API stable
// so the daemon doesn't need to cfg-gate every call site.
#[cfg(not(target_os = "linux"))]
pub async fn pause_playing_players(_ignored: &[String]) -> Vec<String> {
    Vec::new()
}

#[cfg(not(target_os = "linux"))]
pub async fn resume_players(_players: Vec<String>) {}

#[cfg(not(target_os = "linux"))]
pub async fn duck_playing_audio(_volume_percent: u8) -> Vec<DuckedMediaStream> {
    Vec::new()
}

#[cfg(not(target_os = "linux"))]
pub async fn restore_ducked_audio(_streams: Vec<DuckedMediaStream>) {}
