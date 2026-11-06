//! Streaming output session
//!
//! Drives incremental cursor output for streaming transcribers. Backends may
//! emit committed segments only or opt into provisional partial typing. A
//! provisional revision removes only the current partial tail; a final
//! replacement commits the retained prefix plus its corrected suffix.
//!
//! The session tracks cursor state in Unicode scalar values so cancellation and
//! correction emit the same number of BackSpace keys as visible characters.
//! BackSpace output is best effort during cancellation, but a failed transcript
//! update is fatal because the provider and cursor can no longer be kept in sync.
//!
//! Whole-transcript post-processing is intentionally bypassed during streaming.
//! Running it against isolated partial or final deltas would produce output that
//! does not match the cumulative text already visible at the cursor. Output hooks
//! still wrap each typing burst through `output_with_fallback`.

use crate::error::OutputError;
use crate::output::{output_with_fallback, OutputOptions, TextOutput};
use crate::text::TextProcessor;
use std::{future::Future, process::Stdio};
use tokio::process::Command;

fn streaming_output_options<'a>(
    pre_output_command: Option<&'a str>,
    post_output_command: Option<&'a str>,
) -> OutputOptions<'a> {
    OutputOptions {
        pre_output_command,
        post_output_command,
        // Streaming runs while the hotkey may remain held. The modifier-release
        // guard applies only to one-shot output.
        wait_for_modifier_release: false,
        modifier_release_timeout: std::time::Duration::from_millis(0),
    }
}

/// A streaming output session: types finalized segments incrementally,
/// tracks typed-character count, and supports cancel-rewind.
///
/// One session corresponds to one streaming utterance (one hotkey
/// press). The session does not own the output chain — the daemon
/// passes it in by reference for each `commit_segment` call so the
/// existing fallback chain and configuration are reused unchanged.
pub struct StreamingSession {
    /// Concatenated finalized segments committed so far.
    finalized_text: String,
    /// Total Unicode scalar values typed to the output. Counts what was
    /// actually sent to `output_with_fallback`, not what the
    /// post-processor returned (since the post-processor is allowed to
    /// reformat). For accurate rewinding we count *typed* output.
    typed_chars: usize,
    /// Typed but not yet finalized text for the current segment.
    partial: String,
    /// Configured delay between correction BackSpace key events.
    type_delay_ms: u32,
    /// Whether a configured post-output hook was already attempted by an
    /// output burst. `output_with_fallback` runs it on both success and error.
    post_output_hook_attempted: bool,
}

impl StreamingSession {
    /// Create a new empty session with explicit correction pacing.
    pub fn new(type_delay_ms: u32) -> Self {
        Self {
            finalized_text: String::new(),
            typed_chars: 0,
            partial: String::new(),
            type_delay_ms,
            post_output_hook_attempted: false,
        }
    }

    /// Type an incremental partial delta at the cursor.
    ///
    /// `parakeet-rs::ParakeetUnified::transcribe_chunk` returns *only the
    /// newly-decoded text from that chunk's inference* — not a running
    /// cumulative transcript. So each `Partial` event carries just the
    /// new tail, and the right operation is to append it directly. No
    /// LCP reconciliation needed because the partials are deltas by
    /// construction; concatenating them in order rebuilds the
    /// transcript.
    ///
    /// `self.partial` accumulates the typed-but-not-yet-finalized tail
    /// of the current segment so `commit_segment` can know what's
    /// already at the cursor. `typed_chars` is bumped by the actual
    /// scalar count for cancel-rewind accounting.
    pub async fn type_partial_delta(
        &mut self,
        chain: &[Box<dyn TextOutput>],
        new_partial: String,
        pre_output_command: Option<&str>,
        post_output_command: Option<&str>,
    ) -> Result<(), OutputError> {
        if new_partial.is_empty() {
            return Ok(());
        }

        let opts = streaming_output_options(pre_output_command, post_output_command);
        self.post_output_hook_attempted |= post_output_command.is_some();
        output_with_fallback(chain, &new_partial, opts).await?;

        self.typed_chars += new_partial.chars().count();
        self.partial.push_str(&new_partial);
        Ok(())
    }

    /// Revise the current provisional tail without finalizing it.
    pub async fn revise_partial(
        &mut self,
        chain: &[Box<dyn TextOutput>],
        backspace: usize,
        text: &str,
        pre_output_command: Option<&str>,
        post_output_command: Option<&str>,
    ) -> Result<(), OutputError> {
        self.revise_partial_with_backspaces(
            chain,
            backspace,
            text,
            pre_output_command,
            post_output_command,
            emit_backspaces,
        )
        .await
    }

    async fn revise_partial_with_backspaces<F, Fut>(
        &mut self,
        chain: &[Box<dyn TextOutput>],
        backspace: usize,
        text: &str,
        pre_output_command: Option<&str>,
        post_output_command: Option<&str>,
        emit: F,
    ) -> Result<(), OutputError>
    where
        F: FnOnce(usize, u32) -> Fut,
        Fut: Future<Output = usize>,
    {
        let count = backspace.min(self.partial.chars().count());
        if count > 0 {
            let emitted = emit(count, self.type_delay_ms).await;
            if emitted == 0 {
                return Err(OutputError::AllMethodsFailed);
            }

            let retained_chars = self.partial.chars().count().saturating_sub(emitted);
            self.partial = self.partial.chars().take(retained_chars).collect();
            self.typed_chars = self.typed_chars.saturating_sub(emitted);
        }

        if text.is_empty() {
            return Ok(());
        }

        let opts = streaming_output_options(pre_output_command, post_output_command);
        self.post_output_hook_attempted |= post_output_command.is_some();
        output_with_fallback(chain, text, opts).await?;
        self.partial.push_str(text);
        self.typed_chars += text.chars().count();
        Ok(())
    }

    /// Clear the partial buffer (e.g., when a Final supersedes it).
    pub fn clear_partial(&mut self) {
        self.partial.clear();
    }

    /// File-output counterpart to `type_partial_delta`: accumulate the
    /// delta into the partial buffer without typing anything. File-mode
    /// sessions have no cursor to type into, but the delta still has to
    /// be held until the next commit — partials are deltas, not a
    /// cumulative transcript (see `type_partial_delta`), so dropping
    /// this would silently lose whatever text was only ever seen as an
    /// unfinalized partial.
    pub fn observe_partial_delta(&mut self, new_partial: &str) {
        self.partial.push_str(new_partial);
    }

    /// Current partial buffer.
    pub fn partial(&self) -> &str {
        &self.partial
    }

    /// All finalized text committed so far.
    pub fn finalized_text(&self) -> &str {
        &self.finalized_text
    }

    /// Number of Unicode scalar values typed to the output. Used by the
    /// daemon to populate `State::Streaming.typed_chars`.
    pub fn typed_chars(&self) -> usize {
        self.typed_chars
    }

    /// Whether an output burst already attempted the configured post-output
    /// hook. Terminal cleanup uses this to avoid a duplicate invocation.
    pub(crate) fn post_output_hook_attempted(&self) -> bool {
        self.post_output_hook_attempted
    }

    /// Type a finalized segment to the output, optionally running it
    /// through the daemon's [`TextProcessor`] (replacements, spoken
    /// punctuation) first — a segment becoming final is the one point in
    /// a streaming session where text stops being provisional, so this is
    /// where the batch pipeline's processing applies (#669). Partials are
    /// never processed: they get revised by backend-supplied backspace
    /// counts that reference the raw text, and transforming them would
    /// desynchronize that accounting.
    ///
    /// On output error, the session's internal state is **not**
    /// updated; the caller can retry or surface the error.
    ///
    /// The `pre_output_command` and `post_output_command` hooks fire
    /// *once per segment*, just like end-of-utterance batch output. This
    /// is intentional: hooks like compositor submap toggles need to
    /// wrap each typing burst, and streaming segments are short bursts.
    pub async fn commit_segment(
        &mut self,
        chain: &[Box<dyn TextOutput>],
        text: &str,
        text_processor: Option<&TextProcessor>,
        pre_output_command: Option<&str>,
        post_output_command: Option<&str>,
    ) -> Result<(), OutputError> {
        let processed;
        let text = match text_processor {
            Some(tp) => {
                processed = tp.process(text);
                processed.as_str()
            }
            None => text,
        };
        if text.is_empty() {
            self.finalized_text.push_str(&self.partial);
            self.clear_partial();
            return Ok(());
        }

        // Like `transcribe_chunk`, `ParakeetUnified::flush` returns only
        // the newly-emitted tail buffered when the stream closed — it is
        // a delta, not a cumulative transcript. So type it directly,
        // same as a partial.
        //
        // post_process is intentionally bypassed during streaming.
        // Per-segment cleanup would run only against the final tail
        // (not the cumulative transcript visible at the cursor) and
        // produce inconsistent output. Users who rely on post_process
        // should disable streaming for now.
        let opts = streaming_output_options(pre_output_command, post_output_command);
        self.post_output_hook_attempted |= post_output_command.is_some();
        output_with_fallback(chain, text, opts).await?;

        self.typed_chars += text.chars().count();
        // Treat the partial-stream-so-far plus this final tail as the
        // committed text for cancel-rewind context.
        let finalized_tail = format!("{}{}", self.partial, text);
        self.finalized_text.push_str(&finalized_tail);
        self.clear_partial();
        Ok(())
    }

    /// Fold any still-pending `partial` into `finalized_text` without a
    /// trailing Final segment. For file-output sessions, `commit_segment_silent`
    /// only runs when the engine happens to emit a `Final` event — but the
    /// sliding-window engine can legitimately confirm an entire utterance as
    /// a single `Partial` mid-session (e.g. a short phrase transcribed
    /// identically on back-to-back ticks) and then have nothing left for
    /// `final_flush` to send as `Final` at end-of-stream. Without this, that
    /// text sits in `partial` forever and the session ends with an empty
    /// `finalized_text`, silently discarding a correct transcription. Call
    /// this once, at stream end, before reading `finalized_text()`.
    pub fn finalize_pending_partial(&mut self) {
        if !self.partial.is_empty() {
            self.finalized_text.push_str(&self.partial);
            self.partial.clear();
        }
    }

    /// File-output counterpart to `commit_segment`: same finalized_text
    /// bookkeeping, but skips `output_with_fallback` entirely — there's
    /// no cursor/focused window to type into when the session's target
    /// is a file.
    pub fn commit_segment_silent(&mut self, text: &str) {
        if text.is_empty() {
            self.clear_partial();
            return;
        }
        let finalized_tail = format!("{}{}", self.partial, text);
        self.finalized_text.push_str(&finalized_tail);
        self.clear_partial();
    }

    /// Backspace `backspace` chars then commit `text`. Used by streaming
    /// backends that revise the previously-typed partial tail when
    /// finalizing (e.g. Soniox punctuation flips).
    ///
    /// The partial buffer is truncated by `backspace` scalars, then
    /// `text` is appended to both cursor and finalized_text. Net effect:
    /// the cursor ends up with the truncated partial + new text, matching
    /// what the daemon's commit_segment would do for a plain Final.
    ///
    /// Like `commit_segment`, the committed `text` runs through the
    /// [`TextProcessor`] when one is given — it is becoming final here —
    /// while `backspace` stays untouched: it counts raw chars of the
    /// previously-typed partial, which was never processed.
    pub async fn replace_and_commit(
        &mut self,
        chain: &[Box<dyn TextOutput>],
        backspace: usize,
        text: &str,
        text_processor: Option<&TextProcessor>,
        pre_output_command: Option<&str>,
        post_output_command: Option<&str>,
    ) -> Result<(), OutputError> {
        self.replace_and_commit_with_backspaces(
            chain,
            backspace,
            text,
            text_processor,
            pre_output_command,
            post_output_command,
            emit_backspaces,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn replace_and_commit_with_backspaces<F, Fut>(
        &mut self,
        chain: &[Box<dyn TextOutput>],
        backspace: usize,
        text: &str,
        text_processor: Option<&TextProcessor>,
        pre_output_command: Option<&str>,
        post_output_command: Option<&str>,
        emit: F,
    ) -> Result<(), OutputError>
    where
        F: FnOnce(usize, u32) -> Fut,
        Fut: Future<Output = usize>,
    {
        // Final corrections can edit only the current provisional segment.
        // Even a malformed or stale event must never erase finalized text.
        let count = backspace.min(self.partial.chars().count());
        if count > 0 {
            let emitted = emit(count, self.type_delay_ms).await;
            if emitted == 0 {
                return Err(OutputError::AllMethodsFailed);
            }

            let retained_chars = self.partial.chars().count().saturating_sub(emitted);
            self.partial = self.partial.chars().take(retained_chars).collect();
            self.typed_chars = self.typed_chars.saturating_sub(emitted);
        }

        let processed;
        let text = match text_processor {
            Some(tp) => {
                processed = tp.process(text);
                processed.as_str()
            }
            None => text,
        };
        if !text.is_empty() {
            let opts = streaming_output_options(pre_output_command, post_output_command);
            self.post_output_hook_attempted |= post_output_command.is_some();
            output_with_fallback(chain, text, opts).await?;
            self.typed_chars += text.chars().count();
            self.partial.push_str(text);
        }

        // The retained prefix is part of the committed transcript even when
        // the provider's replacement suffix is empty.
        self.finalized_text.push_str(&self.partial);
        self.clear_partial();
        Ok(())
    }

    /// File-output counterpart to `replace_and_commit`: no real cursor to
    /// send backspace keystrokes to, but `backspace` still means the same
    /// thing — trim that many trailing chars off the accumulated
    /// `partial` (a plain string edit, not a keystroke) before folding
    /// the (now-truncated) partial tail plus `text` into `finalized_text`.
    ///
    /// Originally shipped ignoring `backspace` entirely under the
    /// assumption that "no cursor" meant "nothing to revise" — wrong:
    /// `backspace` describes how much of the already-accumulated string
    /// is stale, independent of whether removing it is a keystroke or a
    /// string truncation. Found live: revision mode's file-output
    /// corrections landed as literal concatenations ("Open blue button.
    /// bubbles." instead of "Open blue bubbles.") because the wrong
    /// tail was never actually removed.
    pub fn replace_and_commit_silent(&mut self, backspace: usize, text: &str) {
        if backspace > 0 {
            let new_partial_len = self.partial.chars().count().saturating_sub(backspace);
            self.partial = self.partial.chars().take(new_partial_len).collect();
        }
        if !text.is_empty() {
            let finalized_tail = format!("{}{}", self.partial, text);
            self.finalized_text.push_str(&finalized_tail);
        }
        self.clear_partial();
    }

    /// Best-effort rewind: emit `typed_chars` BackSpace key events via
    /// wtype, falling back to dotool then ydotool. Returns `Ok(())`
    /// even if no backspace backend is available, since the user has
    /// already cancelled and we should not propagate further errors.
    ///
    /// Resets the session's typed-chars counter to zero on success.
    /// The session can be re-used after rewind, though the daemon
    /// typically discards it.
    pub async fn rewind(&mut self) -> Result<(), OutputError> {
        let count = self.typed_chars;
        if count == 0 {
            return Ok(());
        }

        if emit_backspaces(count, self.type_delay_ms).await > 0 {
            self.typed_chars = 0;
            return Ok(());
        }

        tracing::warn!(
            "Streaming cancel: could not rewind {} typed chars (no backspace-capable backend)",
            count
        );
        // Keep typed_chars set so a subsequent retry could work; report
        // a soft error to the daemon.
        Err(OutputError::AllMethodsFailed)
    }
}

/// Backspace `count` chars using the first available method.
/// Returns the actual number of backspaces emitted. Correction keys use the
/// same configured pacing as ordinary streaming text output.
async fn emit_backspaces(count: usize, type_delay_ms: u32) -> usize {
    if count == 0 {
        return 0;
    }
    if try_wtype_backspaces(count, type_delay_ms).await {
        return count;
    }
    if try_dotool_backspaces(count, type_delay_ms).await {
        return count;
    }
    if try_ydotool_backspaces(count, type_delay_ms).await {
        return count;
    }
    0
}

async fn try_wtype_backspaces(count: usize, type_delay_ms: u32) -> bool {
    let mut cmd = Command::new("wtype");
    cmd.args(build_wtype_backspace_args(count, type_delay_ms));
    cmd.stdout(Stdio::null()).stderr(Stdio::null());
    matches!(cmd.status().await, Ok(s) if s.success())
}

fn build_wtype_backspace_args(count: usize, type_delay_ms: u32) -> Vec<String> {
    let mut args = Vec::with_capacity(count * 4);
    for _ in 0..count {
        if type_delay_ms > 0 {
            args.push("-s".to_string());
            args.push(type_delay_ms.to_string());
        }
        args.push("-k".to_string());
        args.push("BackSpace".to_string());
    }
    args
}

async fn try_dotool_backspaces(count: usize, type_delay_ms: u32) -> bool {
    // Prefer `dotoolc` whenever dotoold is actually accepting input.
    // Spawning raw `dotool` creates a *new* uinput keyboard per call;
    // KDE Plasma can drop events on the typing keyboard while these
    // ephemeral keyboards rapidly appear and disappear. Routing through
    // dotoolc reuses dotoold's persistent keyboard.
    use tokio::io::AsyncWriteExt;
    let (binary, env_pipe) = match crate::output::dotool::DotoolOutput::live_daemon_pipe_path() {
        Some(p) => ("dotoolc", Some(p)),
        None => ("dotool", None),
    };
    let mut cmd = Command::new(binary);
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(p) = env_pipe {
        cmd.env("DOTOOL_PIPE", p);
    }
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(_) => return false,
    };
    if let Some(mut stdin) = child.stdin.take() {
        let buf = build_dotool_backspace_commands(count, type_delay_ms);
        if stdin.write_all(buf.as_bytes()).await.is_err() {
            return false;
        }
        drop(stdin);
    }
    matches!(child.wait().await, Ok(s) if s.success())
}

fn build_dotool_backspace_commands(count: usize, type_delay_ms: u32) -> String {
    if count == 0 {
        return String::new();
    }

    let mut commands = String::with_capacity(count * "key backspace\n".len() + 32);
    if type_delay_ms > 0 {
        commands.push_str(&format!("keydelay {type_delay_ms}\n"));
        commands.push_str(&format!("keyhold {type_delay_ms}\n"));
    }
    for _ in 0..count {
        commands.push_str("key backspace\n");
    }
    commands
}

async fn try_ydotool_backspaces(count: usize, type_delay_ms: u32) -> bool {
    let mut cmd = Command::new("ydotool");
    cmd.args(build_ydotool_backspace_args(count, type_delay_ms));
    cmd.stdout(Stdio::null()).stderr(Stdio::null());
    matches!(cmd.status().await, Ok(s) if s.success())
}

fn build_ydotool_backspace_args(count: usize, type_delay_ms: u32) -> Vec<String> {
    if count == 0 {
        return Vec::new();
    }
    let mut args = Vec::with_capacity(3 + count * 2);
    args.push("key".to_string());
    if type_delay_ms > 0 {
        args.push("-d".to_string());
        args.push(type_delay_ms.to_string());
    }
    for _ in 0..count {
        args.push("14:1".to_string());
        args.push("14:0".to_string());
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::Mutex;

    /// In-memory output that records every typed string. Used to
    /// verify session bookkeeping without spawning subprocesses.
    struct RecordingOutput {
        log: Mutex<Vec<String>>,
    }

    impl RecordingOutput {
        fn new() -> Self {
            Self {
                log: Mutex::new(Vec::new()),
            }
        }

        fn typed(&self) -> Vec<String> {
            self.log.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl TextOutput for RecordingOutput {
        async fn output(&self, text: &str) -> Result<(), OutputError> {
            self.log.lock().unwrap().push(text.to_string());
            Ok(())
        }
        async fn is_available(&self) -> bool {
            true
        }
        fn name(&self) -> &'static str {
            "recording-test"
        }
    }

    fn chain_with(out: std::sync::Arc<RecordingOutput>) -> Vec<Box<dyn TextOutput>> {
        struct Wrap(std::sync::Arc<RecordingOutput>);
        #[async_trait]
        impl TextOutput for Wrap {
            async fn output(&self, text: &str) -> Result<(), OutputError> {
                self.0.output(text).await
            }
            async fn is_available(&self) -> bool {
                self.0.is_available().await
            }
            fn name(&self) -> &'static str {
                self.0.name()
            }
        }
        vec![Box::new(Wrap(out))]
    }

    #[tokio::test]
    async fn commits_segment_and_tracks_typed_chars() {
        let rec = std::sync::Arc::new(RecordingOutput::new());
        let chain = chain_with(rec.clone());
        let mut session = StreamingSession::new(0);

        session
            .commit_segment(&chain, "hello", None, None, None)
            .await
            .unwrap();
        assert_eq!(rec.typed(), vec!["hello".to_string()]);
        assert_eq!(session.typed_chars(), 5);
        assert_eq!(session.finalized_text(), "hello");

        session
            .commit_segment(&chain, " world", None, None, None)
            .await
            .unwrap();
        assert_eq!(session.typed_chars(), 11);
        assert_eq!(session.finalized_text(), "hello world");
    }

    /// #669: a segment becoming final is where streaming text stops being
    /// provisional, so the daemon's replacements and spoken punctuation
    /// apply there — what reaches the cursor is the processed text, and
    /// typed_chars/finalized_text account for what was actually typed.
    #[tokio::test]
    async fn final_segments_apply_text_processing() {
        let config = crate::config::TextConfig {
            spoken_punctuation: true,
            replacements: [("vox type".to_string(), "voxtype".to_string())]
                .into_iter()
                .collect(),
            ..Default::default()
        };
        let tp = TextProcessor::new(&config);

        let rec = std::sync::Arc::new(RecordingOutput::new());
        let chain = chain_with(rec.clone());
        let mut session = StreamingSession::new(0);
        session
            .commit_segment(&chain, "I use vox type period", Some(&tp), None, None)
            .await
            .unwrap();
        assert_eq!(rec.typed(), vec!["I use voxtype.".to_string()]);
        assert_eq!(session.finalized_text(), "I use voxtype.");
        assert_eq!(session.typed_chars(), "I use voxtype.".chars().count());

        // Replace commits are finals too: the correction text is processed,
        // while the backspace count still refers to the raw partial tail.
        let mut session = StreamingSession::new(0);
        session.observe_partial_delta("raw partial");
        session
            .replace_and_commit(&chain, 8, "vox type comma", Some(&tp), None, None)
            .await
            .unwrap();
        assert_eq!(rec.typed().last().unwrap(), "voxtype,");
    }

    #[tokio::test]
    async fn typed_chars_counts_unicode_scalars_not_bytes() {
        // Three CJK chars = 3 scalars but 9 UTF-8 bytes. Cancel rewind
        // must use scalars to send one BackSpace per visible char.
        let rec = std::sync::Arc::new(RecordingOutput::new());
        let chain = chain_with(rec.clone());
        let mut session = StreamingSession::new(0);
        session
            .commit_segment(&chain, "你好世", None, None, None)
            .await
            .unwrap();
        assert_eq!(session.typed_chars(), 3);
    }

    #[tokio::test]
    async fn empty_segment_is_noop() {
        let rec = std::sync::Arc::new(RecordingOutput::new());
        let chain = chain_with(rec.clone());
        let mut session = StreamingSession::new(0);
        session
            .commit_segment(&chain, "", None, None, None)
            .await
            .unwrap();
        assert!(rec.typed().is_empty());
        assert_eq!(session.typed_chars(), 0);
    }

    #[tokio::test]
    async fn empty_final_commits_typed_partial_without_duplicate_output() {
        let rec = std::sync::Arc::new(RecordingOutput::new());
        let chain = chain_with(rec.clone());
        let mut session = StreamingSession::new(0);

        session
            .type_partial_delta(&chain, "hello".into(), None, None)
            .await
            .unwrap();
        session
            .commit_segment(&chain, "", None, None, None)
            .await
            .unwrap();

        assert_eq!(rec.typed(), vec!["hello".to_string()]);
        assert_eq!(session.finalized_text(), "hello");
        assert_eq!(session.typed_chars(), 5);
        assert_eq!(session.partial(), "");

        session
            .type_partial_delta(&chain, "next".into(), None, None)
            .await
            .unwrap();
        assert_eq!(session.partial(), "next");
    }

    #[test]
    fn finalize_pending_partial_folds_leftover_partial() {
        // Reproduces the empty-transcript bug: the sliding-window engine
        // can confirm an entire short utterance as a single Partial
        // mid-session (see `on_tick`'s growing-mode branch) and then have
        // nothing left for `final_flush` to send as a terminal Final. A
        // file-output session that only ever calls `commit_segment_silent`
        // on a Final event would strand that text in `partial` forever and
        // write an empty file despite a correct transcription.
        let mut session = StreamingSession::new(0);
        session.observe_partial_delta("turn on Sophie's room.");
        assert_eq!(session.finalized_text(), "");

        session.finalize_pending_partial();

        assert_eq!(session.finalized_text(), "turn on Sophie's room.");
        assert_eq!(session.partial(), "");
    }

    #[test]
    fn finalize_pending_partial_is_noop_when_already_final() {
        let mut session = StreamingSession::new(0);
        session.commit_segment_silent("hello");
        session.finalize_pending_partial();
        assert_eq!(session.finalized_text(), "hello");
    }

    #[test]
    fn replace_and_commit_silent_trims_the_wrong_tail_before_appending() {
        // Reproduces a real bug found live: revision mode's file-output
        // corrections landed as literal concatenations instead of actual
        // corrections, because replace_and_commit_silent ignored its
        // `backspace` argument entirely. "Open blue button." (partial)
        // then a Replace{backspace: 8, text: " bubbles."} correcting
        // "button." (7 chars) + its leading space (1) must produce
        // "Open blue bubbles.", not "Open blue button. bubbles.".
        let mut session = StreamingSession::new(0);
        session.observe_partial_delta("Open blue button.");
        session.replace_and_commit_silent(8, " bubbles.");
        assert_eq!(session.finalized_text(), "Open blue bubbles.");
    }

    #[test]
    fn replace_and_commit_silent_zero_backspace_is_a_plain_append() {
        let mut session = StreamingSession::new(0);
        session.observe_partial_delta("hello");
        session.replace_and_commit_silent(0, " world");
        assert_eq!(session.finalized_text(), "hello world");
    }

    #[test]
    fn replace_and_commit_silent_backspace_capped_at_partial_length() {
        // A backspace count larger than what's actually pending in
        // `partial` should not panic or reach into already-finalized
        // text — same invariant `replace_and_commit` (the live-typing
        // counterpart) relies on: corrections only ever touch the
        // still-open partial tail, never truly-finalized content.
        let mut session = StreamingSession::new(0);
        session.commit_segment_silent("hello");
        session.observe_partial_delta(" wor");
        session.replace_and_commit_silent(999, "world");
        assert_eq!(session.finalized_text(), "helloworld");
    }

    #[tokio::test]
    async fn rewind_with_zero_chars_is_ok() {
        let mut session = StreamingSession::new(0);
        // Should succeed without spawning anything.
        session.rewind().await.unwrap();
        assert_eq!(session.typed_chars(), 0);
    }

    #[tokio::test]
    async fn successful_partial_revision_changes_only_provisional_state() {
        let rec = std::sync::Arc::new(RecordingOutput::new());
        let chain = chain_with(rec.clone());
        let mut session = StreamingSession::new(17);
        session
            .commit_segment(&chain, "done ", None, None, None)
            .await
            .unwrap();
        session
            .type_partial_delta(&chain, "hello wurld".into(), None, None)
            .await
            .unwrap();

        session
            .revise_partial_with_backspaces(
                &chain,
                4,
                "orld",
                None,
                None,
                |count, _delay| async move { count },
            )
            .await
            .unwrap();

        assert_eq!(
            (
                session.partial(),
                session.finalized_text(),
                session.typed_chars(),
                rec.typed(),
            ),
            (
                "hello world",
                "done ",
                16,
                vec![
                    "done ".to_string(),
                    "hello wurld".to_string(),
                    "orld".to_string(),
                ],
            )
        );
    }

    #[tokio::test]
    async fn empty_final_commits_revised_partial_without_duplicate_output() {
        let rec = std::sync::Arc::new(RecordingOutput::new());
        let chain = chain_with(rec.clone());
        let mut session = StreamingSession::new(17);
        session
            .type_partial_delta(&chain, "This is difficult.".into(), None, None)
            .await
            .unwrap();
        session
            .revise_partial_with_backspaces(
                &chain,
                1,
                " because",
                None,
                None,
                |count, _delay| async move { count },
            )
            .await
            .unwrap();
        session
            .commit_segment(&chain, "", None, None, None)
            .await
            .unwrap();

        assert_eq!(
            (
                session.partial(),
                session.finalized_text(),
                session.typed_chars()
            ),
            ("", "This is difficult because", 25)
        );
        assert_eq!(
            rec.typed(),
            vec!["This is difficult.".to_string(), " because".to_string(),]
        );
    }

    #[tokio::test]
    async fn partial_revision_cannot_backspace_finalized_text() {
        let rec = std::sync::Arc::new(RecordingOutput::new());
        let chain = chain_with(rec);
        let emitted = std::sync::Arc::new(Mutex::new(Vec::new()));
        let mut session = StreamingSession::new(17);
        session
            .commit_segment(&chain, "done ", None, None, None)
            .await
            .unwrap();
        session
            .type_partial_delta(&chain, "hello".into(), None, None)
            .await
            .unwrap();

        let captured = emitted.clone();
        session
            .revise_partial_with_backspaces(
                &chain,
                99,
                "hullo",
                None,
                None,
                move |count, delay| async move {
                    captured.lock().unwrap().push((count, delay));
                    count
                },
            )
            .await
            .unwrap();

        assert_eq!(*emitted.lock().unwrap(), vec![(5, 17)]);
        assert_eq!(
            (
                session.partial(),
                session.finalized_text(),
                session.typed_chars()
            ),
            ("hullo", "done ", 10)
        );
    }

    #[tokio::test]
    async fn unavailable_backspaces_leave_revision_state_unchanged() {
        let rec = std::sync::Arc::new(RecordingOutput::new());
        let chain = chain_with(rec.clone());
        let mut session = StreamingSession::new(17);
        session
            .type_partial_delta(&chain, "hello wurld".into(), None, None)
            .await
            .unwrap();

        let error = session
            .revise_partial_with_backspaces(&chain, 4, "orld", None, None, |_count, _delay| async {
                0
            })
            .await
            .unwrap_err();

        assert!(matches!(error, OutputError::AllMethodsFailed));
        assert_eq!(
            (
                session.partial(),
                session.finalized_text(),
                session.typed_chars(),
                rec.typed(),
            ),
            ("hello wurld", "", 11, vec!["hello wurld".to_string()])
        );
    }

    struct FailingOutput;

    #[async_trait]
    impl TextOutput for FailingOutput {
        async fn output(&self, _text: &str) -> Result<(), OutputError> {
            Err(OutputError::InjectionFailed("test failure".to_string()))
        }

        async fn is_available(&self) -> bool {
            true
        }

        fn name(&self) -> &'static str {
            "failing-test"
        }
    }

    #[tokio::test]
    async fn replacement_output_failure_keeps_post_backspace_state() {
        let rec = std::sync::Arc::new(RecordingOutput::new());
        let initial_chain = chain_with(rec.clone());
        let failing_chain: Vec<Box<dyn TextOutput>> = vec![Box::new(FailingOutput)];
        let mut session = StreamingSession::new(17);
        session
            .type_partial_delta(&initial_chain, "hello wurld".into(), None, None)
            .await
            .unwrap();

        let error = session
            .revise_partial_with_backspaces(
                &failing_chain,
                4,
                "orld",
                None,
                None,
                |count, _delay| async move { count },
            )
            .await
            .unwrap_err();

        assert!(matches!(error, OutputError::AllMethodsFailed));
        assert_eq!(
            (
                session.partial(),
                session.finalized_text(),
                session.typed_chars(),
                rec.typed(),
            ),
            ("hello w", "", 7, vec!["hello wurld".to_string()])
        );
    }

    #[tokio::test]
    async fn empty_replacement_suffix_finalizes_the_retained_partial() {
        let rec = std::sync::Arc::new(RecordingOutput::new());
        let chain = chain_with(rec.clone());
        let emitted = std::sync::Arc::new(Mutex::new(Vec::new()));
        let mut session = StreamingSession::new(17);
        session
            .type_partial_delta(&chain, "hello world".into(), None, None)
            .await
            .unwrap();

        let captured = emitted.clone();
        session
            .replace_and_commit_with_backspaces(
                &chain,
                6,
                "",
                None,
                None,
                None,
                move |count, delay| async move {
                    captured.lock().unwrap().push((count, delay));
                    count
                },
            )
            .await
            .unwrap();

        assert_eq!(*emitted.lock().unwrap(), vec![(6, 17)]);
        assert_eq!(
            (
                session.partial(),
                session.finalized_text(),
                session.typed_chars(),
                rec.typed(),
            ),
            ("", "hello", 5, vec!["hello world".to_string()],)
        );
    }

    #[tokio::test]
    async fn final_replacement_cannot_backspace_finalized_text() {
        let rec = std::sync::Arc::new(RecordingOutput::new());
        let chain = chain_with(rec.clone());
        let emitted = std::sync::Arc::new(Mutex::new(Vec::new()));
        let mut session = StreamingSession::new(17);
        session
            .commit_segment(&chain, "done ", None, None, None)
            .await
            .unwrap();
        session
            .type_partial_delta(&chain, "hello".into(), None, None)
            .await
            .unwrap();

        let captured = emitted.clone();
        session
            .replace_and_commit_with_backspaces(
                &chain,
                99,
                "",
                None,
                None,
                None,
                move |count, delay| async move {
                    captured.lock().unwrap().push((count, delay));
                    count
                },
            )
            .await
            .unwrap();

        assert_eq!(*emitted.lock().unwrap(), vec![(5, 17)]);
        assert_eq!(
            (
                session.partial(),
                session.finalized_text(),
                session.typed_chars(),
                rec.typed(),
            ),
            (
                "",
                "done ",
                5,
                vec!["done ".to_string(), "hello".to_string()],
            )
        );
    }

    #[test]
    fn dotool_backspaces_use_key_pacing_without_mutating_zero_count_state() {
        assert_eq!(
            build_dotool_backspace_commands(3, 17),
            "keydelay 17\nkeyhold 17\nkey backspace\nkey backspace\nkey backspace\n"
        );
        assert_eq!(
            build_dotool_backspace_commands(2, 0),
            "key backspace\nkey backspace\n"
        );
        assert_eq!(build_dotool_backspace_commands(0, 17), "");
        assert!(!build_dotool_backspace_commands(3, 17).contains("typedelay"));
        assert!(!build_dotool_backspace_commands(3, 17).contains("typehold"));
    }

    #[test]
    fn wtype_backspaces_sleep_before_each_key_unless_delay_is_zero() {
        assert_eq!(
            build_wtype_backspace_args(2, 17),
            vec!["-s", "17", "-k", "BackSpace", "-s", "17", "-k", "BackSpace"]
        );
        assert_eq!(
            build_wtype_backspace_args(2, 0),
            vec!["-k", "BackSpace", "-k", "BackSpace"]
        );
        assert!(build_wtype_backspace_args(0, 17).is_empty());
    }

    #[test]
    fn ydotool_backspaces_use_configured_delay_unless_zero() {
        assert_eq!(
            build_ydotool_backspace_args(2, 17),
            vec!["key", "-d", "17", "14:1", "14:0", "14:1", "14:0"]
        );
        assert_eq!(
            build_ydotool_backspace_args(2, 0),
            vec!["key", "14:1", "14:0", "14:1", "14:0"]
        );
        assert!(build_ydotool_backspace_args(0, 17).is_empty());
    }
}
