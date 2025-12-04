"""
Starting script for AI voice assistant with memory integration.

This script:
1. Listens to speech (STT)
2. Identifies the speaker
3. Stores transcripts in memory
4. Sends to AI (Ollama)
5. Stores AI responses in memory
6. Speaks responses (TTS)
"""

from __future__ import annotations

import sys
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import modules
try:
    from modules.whisper_live import WhisperLiveStreamer, TranscriptEvent
    WHISPER_AVAILABLE = True
    LiveSpeechStreamer = None  # Not needed when Whisper is available
except ImportError:
    WHISPER_AVAILABLE = False
    from modules.live_speech import LiveSpeechStreamer, TranscriptEvent
    WhisperLiveStreamer = None  # Not available

from modules.voice_identity import VoiceIdentifier, IdentificationResult
from modules.ollama_ai import OllamaAISession, create_ollama_clients
from modules.memory import create_memory_manager, process_turn, get_context

# Import sounddevice (same as live_speech module uses)
try:
    import sounddevice as sd  # type: ignore
    import numpy as np  # type: ignore
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False

# Try to import TTS libraries that can generate audio
TTS_AVAILABLE = False
TTS_BACKEND = None

try:
    from gtts import gTTS  # type: ignore
    TTS_AVAILABLE = True
    TTS_BACKEND = "gtts"
except ImportError:
    try:
        import pyttsx3  # type: ignore
        TTS_AVAILABLE = True
        TTS_BACKEND = "pyttsx3"
    except ImportError:
        pass


class TTSEngine:
    """
    TTS engine using sounddevice (same infrastructure as live_speech module).
    Uses sounddevice to play audio, similar to how live_speech uses it to capture audio.
    """

    def __init__(self, use_gtts: bool = True):
        """
        Initialize TTS engine.
        
        Args:
            use_gtts: If True, use gTTS (requires internet). If False, use pyttsx3 (offline).
        """
        if not SD_AVAILABLE:
            raise RuntimeError(
                "sounddevice is required for TTS (same as live_speech uses). "
                "Install it with: pip install sounddevice"
            )
        
        self.use_gtts = use_gtts
        self.backend = None
        self.sample_rate = 22050  # Standard audio sample rate
        self._tts_lock = threading.Lock()  # Lock for pyttsx3 to prevent concurrent calls
        
        if use_gtts:
            if TTS_BACKEND == "gtts":
                self.backend = "gtts"
                print("TTS: Using gTTS + sounddevice (like live_speech module)")
            else:
                raise RuntimeError(
                    "gTTS not available. Install it with: pip install gtts"
                )
        else:
            if TTS_BACKEND == "pyttsx3":
                try:
                    # Test that pyttsx3 works, but we'll create fresh instances in speak()
                    # to avoid "run loop already started" errors when called from multiple threads
                    test_engine = pyttsx3.init()
                    test_engine.stop()  # Clean up test engine
                    self.backend = "pyttsx3"
                    print("TTS: Using pyttsx3 (offline) - will create fresh engine for each speak call")
                except Exception as e:
                    raise RuntimeError(f"Failed to initialize pyttsx3: {e}")
            else:
                raise RuntimeError(
                    "No TTS backend available. Install gtts or pyttsx3. "
                    "Run: pip install gtts (or pip install pyttsx3)"
                )

    def speak(self, text: str) -> None:
        """
        Speak the given text using sounddevice (same as live_speech infrastructure).
        
        Args:
            text: Text to speak
        """
        if not text.strip():
            return

        print(f"[TTS] Speaking: {text[:50]}..." if len(text) > 50 else f"[TTS] Speaking: {text}")

        if self.backend == "gtts":
            try:
                import tempfile
                import os
                import librosa  # type: ignore

                # Generate audio with gTTS
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
                    tts = gTTS(text=text, lang='en', slow=False)
                    tts.save(tmp.name)
                    tmp_path = tmp.name

                # Load audio file using librosa (same ecosystem as live_speech)
                audio_data, sr = librosa.load(tmp_path, sr=self.sample_rate)
                
                # Play audio using sounddevice (same as live_speech uses for input)
                sd.play(audio_data, samplerate=sr)
                sd.wait()  # Wait until playback is finished

                # Cleanup
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            except ImportError as e:
                print(f"[TTS ERROR] Missing dependency: {e}")
                print("Install with: pip install librosa soundfile")
            except Exception as e:
                print(f"[TTS ERROR] {e}")
        elif self.backend == "pyttsx3":
            # pyttsx3: Use lock to prevent concurrent calls and create fresh engine each time
            # This avoids "run loop already started" errors when called from multiple threads
            with self._tts_lock:
                try:
                    # Create a fresh engine instance for this call
                    # pyttsx3 can't reuse engines after runAndWait(), so we create new ones
                    engine = pyttsx3.init()
                    # Configure voice settings
                    voices = engine.getProperty('voices')
                    if voices:
                        for voice in voices:
                            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                                engine.setProperty('voice', voice.id)
                                break
                        else:
                            engine.setProperty('voice', voices[0].id)
                    engine.setProperty('rate', 150)
                    engine.setProperty('volume', 0.9)
                    
                    # Use the fresh engine for this call
                    engine.say(text)
                    engine.runAndWait()
                    # Engine is automatically cleaned up after runAndWait() completes
                except Exception as e:
                    print(f"[TTS ERROR] {e}")
                    import traceback
                    traceback.print_exc()


class AIVoiceSession:
    """Main session that coordinates STT, voice ID, memory, AI, and TTS."""

    def __init__(
        self,
        model: str = "P2Wdisabled/lyra:7b",
        session_id: str = "default-session",
        memory_storage_dir: Path | str = Path("memory_data"),
        voice_db_path: Optional[Path | str] = None,
        use_tts: bool = True,
        tts_use_gtts: bool = False,
        system_prompt: Optional[str] = None,
        whisper_model_size: str = "large-v3-turbo",
        whisper_device: str = "cpu",
        whisper_compute_type: str = "int8",
    ):
        """
        Initialize the AI voice session.

        Args:
            model: Ollama model name
            session_id: Session identifier for memory
            memory_storage_dir: Directory for memory storage
            voice_db_path: Path to voice database (optional)
            use_tts: Whether to use text-to-speech
            tts_use_gtts: Use gTTS instead of pyttsx3
            system_prompt: Optional system prompt for AI
        """
        self.session_id = session_id
        self.model = model

        # Initialize TTS (using sounddevice like live_speech module)
        self.tts: Optional[TTSEngine] = None
        if use_tts:
            if not SD_AVAILABLE:
                print("WARNING: sounddevice not available (required for TTS, same as live_speech).")
                print("  Install with: pip install sounddevice")
            elif not TTS_AVAILABLE:
                print("WARNING: TTS library not available. Install gtts or pyttsx3:")
                print("  pip install gtts librosa soundfile  (for gTTS + sounddevice)")
                print("  OR: pip install pyttsx3  (for offline TTS)")
            else:
                try:
                    self.tts = TTSEngine(use_gtts=tts_use_gtts)
                    print("TTS initialized successfully (using sounddevice like live_speech)")
                except Exception as e:
                    print(f"WARNING: Failed to initialize TTS: {e}")
                    print("AI responses will be text-only")
        else:
            print("TTS disabled - AI responses will be text-only")

        # Initialize voice identifier
        self.voice_identifier: Optional[VoiceIdentifier] = None
        self.voice_db_path = Path(voice_db_path) if voice_db_path else Path("voice_database.json")
        try:
            if self.voice_db_path.exists() and self.voice_db_path.stat().st_size > 10:
                self.voice_identifier = VoiceIdentifier.load_database(self.voice_db_path)
                known_speakers = list(self.voice_identifier.known_speakers)
                if known_speakers:
                    print(f"Loaded voice database from {self.voice_db_path}")
                    print(f"Known speakers: {', '.join(known_speakers)}")
                else:
                    print(f"Voice database exists but is empty.")
                    print(f"To register a speaker, run: python start_ai_session.py --register-speaker <name>")
            else:
                self.voice_identifier = VoiceIdentifier()
                print("Created new voice identifier")
                print(f"To register a speaker, run: python start_ai_session.py --register-speaker <name>")
        except Exception as e:
            print(f"Warning: Voice identification not available: {e}")
            self.voice_identifier = VoiceIdentifier()

        # Initialize Ollama clients and memory
        print("Initializing Ollama and memory system...")
        llm, emb = create_ollama_clients(model=model)
        self.memory_manager = create_memory_manager(
            llm_client=llm,
            embedding_client=emb,
            storage_dir=memory_storage_dir,
            system_instructions=[system_prompt] if system_prompt else None,
        )
        self.ai_session = OllamaAISession(model=model, system_prompt=system_prompt)

        # Initialize speech streamer (use Whisper if available)
        if WHISPER_AVAILABLE and WhisperLiveStreamer is not None:
            self.streamer = WhisperLiveStreamer(
                model_size=whisper_model_size,
                device=whisper_device,
                compute_type=whisper_compute_type,
                silence_duration=2.0,  # Process after 2 seconds of silence
            )
            print(f"Using Whisper for speech-to-text (model: {whisper_model_size}, device: {whisper_device})")
            print("Live transcription enabled - you'll see text as it's being recognized")
        else:
            if LiveSpeechStreamer is None:
                raise RuntimeError(
                    "Neither Whisper nor Vosk is available. "
                    "Install faster-whisper (for Whisper) or vosk (for Vosk)."
                )
            self.streamer = LiveSpeechStreamer()
            print("Using Vosk for speech-to-text (fallback - Whisper not available)")
        
        self._current_speaker: Optional[str] = None
        self._pending_final_text: Optional[str] = None
        self._audio_buffer: list[bytes] = []
        self._processing_lock = threading.Lock()
        self._is_processing = False
        self._message_queue: queue.Queue[tuple[str, str]] = queue.Queue()  # (text, speaker_name)
        
        # Voice sample collection
        self._voice_audio_buffer: list[bytes] = []  # Buffer audio for voice sample
        self._voice_buffer_lock = threading.Lock()
        self._last_voice_save_time = time.time()
        self._voice_save_interval = 30.0  # Save voice database every 30 seconds
        self._next_speaker_id = 1  # Counter for auto-generating speaker names
        
        # Start message processor thread
        self._processor_thread = threading.Thread(
            target=self._process_message_queue,
            daemon=True,
            name="MessageQueueProcessor"
        )
        self._processor_thread.start()

        print("AI Voice Session initialized!")

    def _on_audio_chunk(self, chunk: bytes, sample_rate: int) -> None:
        """Handle audio chunks for voice identification and sample collection.
        
        This callback must be FAST to avoid audio buffer overflow.
        We minimize work here and do heavy processing asynchronously.
        """
        if not self.voice_identifier:
            return
        
        try:
            # Fast: Just buffer the audio chunk (minimal work)
            with self._voice_buffer_lock:
                self._voice_audio_buffer.append(chunk)
                # Limit buffer size efficiently (keep last ~5 seconds at 16kHz)
                max_buffer_size = sample_rate * 5 * 2  # 5 seconds * 2 bytes per sample
                # Quick size check - only trim if buffer is getting large
                if len(self._voice_audio_buffer) > 100:  # Only check when buffer is large
                    total_size = sum(len(c) for c in self._voice_audio_buffer)
                    while total_size > max_buffer_size and len(self._voice_audio_buffer) > 1:
                        removed = self._voice_audio_buffer.pop(0)
                        total_size -= len(removed)
            
            # Skip voice identification in callback to avoid blocking
            # Identification will happen when transcript is final (in _collect_voice_sample_async)
            # This prevents audio overflow warnings
        except Exception:
            pass  # Silently fail if voice ID not available

    def _on_transcript(self, event: TranscriptEvent) -> None:
        """Handle incoming transcript events."""
        try:
            if not event.text.strip():
                return

            text = event.text.strip()
            speaker_name = self._current_speaker or "Unknown"
            speaker_label = f"[{speaker_name}]"
            
            if event.is_final:
                # Final transcript - queue it for processing
                print(f"\n[USER] {speaker_label} {text}")
                
                # Collect voice sample asynchronously (if we have audio buffer)
                # Don't pass speaker_name - let identification determine it fresh from audio
                self._collect_voice_sample_async()
                
                # Add to queue (non-blocking, STT continues immediately)
                try:
                    self._message_queue.put_nowait((text, speaker_name))
                    print(f"[DEBUG] Message queued successfully (queue size: {self._message_queue.qsize()})")
                except queue.Full:
                    print("[WARNING] Message queue full, dropping message")
                # Return immediately so STT can continue listening
            else:
                # Partial/live transcript - just show it
                print(f"\r[LIVE] {speaker_label} {text}", end="", flush=True)
        except Exception as e:
            # Never let exceptions in callback stop the streamer
            print(f"\n[ERROR] Error in transcript handler: {e}")
            import traceback
            traceback.print_exc()

    def _process_message_queue(self) -> None:
        """Background thread that processes messages from the queue."""
        print("[DEBUG] Message queue processor thread started")
        while True:
            try:
                # Wait for messages (blocking, but in separate thread)
                text, speaker_name = self._message_queue.get(timeout=1.0)
                remaining = self._message_queue.qsize()
                print(f"[DEBUG] Processing queued message (remaining in queue: {remaining}): {text[:50]}...")
                try:
                    self._process_message_async(text, speaker_name)
                    print(f"[DEBUG] Finished processing message, ready for next (queue size: {self._message_queue.qsize()})")
                except Exception as e:
                    # Error processing one message, but continue processing others
                    print(f"[ERROR] Error processing message '{text[:50]}...': {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue to next message
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Fatal error in message queue processor: {e}")
                import traceback
                traceback.print_exc()
                # Continue anyway - don't let the thread die
                time.sleep(0.1)

    def _process_message_async(self, text: str, speaker_name: str) -> None:
        """Process message asynchronously to avoid blocking STT."""
        try:
            print(f"[STATUS] Processing message from {speaker_name}...")
            # Store user message in memory with format "[username]: message" (async)
            formatted_content = f"[{speaker_name}]: {text}"
            
            # Store in memory (this is fast, but do it first)
            process_turn(
                self.memory_manager,
                self.session_id,
                role="user",
                content=formatted_content,
                speaker=self._current_speaker,
            )
            print(f"[STATUS] Message stored in memory")
            
            # Process with AI (this can take time, but should return)
            # Pass the formatted content (same as stored in memory) to AI
            self._process_user_message(formatted_content)
            print(f"[STATUS] Finished processing message from {speaker_name}")
        except Exception as e:
            print(f"[ERROR] Error processing message: {e}")
            import traceback
            traceback.print_exc()
            # Exception is caught, queue processor will continue

    def _process_user_message(self, formatted_message: str) -> None:
        """Process a final user message with AI.
        
        Args:
            formatted_message: The message in the same format as stored in memory (e.g., "[username]: message")
        """
        # Use fast context building - just get recent turns from working memory
        # Skip expensive vector searches for speed
        print("[STATUS] Building context (fast mode - recent turns only)...")
        recent_turns = self.memory_manager.working_memory.get_recent_turns()
        
        # Build simple context pack (fast, no expensive searches)
        from memory_system.types import ContextPack
        context = ContextPack(
            system_instructions=self.memory_manager.system_instructions or [],
            persona_snippets=[],  # Skip persona for speed
            task_state=self.memory_manager.working_memory.get_task_state(),
            episodic_snippets=[],  # Skip episodic search for speed
            semantic_snippets=[],  # Skip semantic search for speed
            recent_turns=recent_turns[-10:],  # Last 10 turns for context
        )

        # Build prompt with context
        prompt_parts = []
        if context.system_instructions:
            prompt_parts.append("\n".join(context.system_instructions))
        if context.persona_snippets:
            prompt_parts.append("Persona context:\n" + "\n".join(context.persona_snippets))
        if context.episodic_snippets:
            prompt_parts.append("Previous conversations:\n" + "\n".join(context.episodic_snippets))
        if context.semantic_snippets:
            prompt_parts.append("Knowledge:\n" + "\n".join(context.semantic_snippets))
        if context.recent_turns:
            # Use the content as stored in memory (already includes speaker format like "[username]: message")
            recent_lines = []
            for t in context.recent_turns[-3:]:
                # Use the content directly - it's already in "[username]: message" format
                recent_lines.append(f"{t.role}: {t.content}")
            prompt_parts.append(f"Recent conversation:\n" + "\n".join(recent_lines))

        # Add current user message - use the formatted message (same as stored in memory)
        prompt_parts.append(f"Current message:\nuser: {formatted_message}")

        prompt = "\n\n".join(prompt_parts) if prompt_parts else formatted_message

        # Get AI response
        print("[AI] Thinking...")
        try:
            response = self.ai_session.chat(prompt)
            if not response or not response.strip():
                print("[WARNING] AI returned empty response")
                return
            print(f"[AI] {response}")

            # Store AI response in memory asynchronously (don't block)
            def store_response_async():
                try:
                    process_turn(
                        self.memory_manager,
                        self.session_id,
                        role="assistant",
                        content=response,
                    )
                    print("[STATUS] AI response stored in memory")
                except Exception as e:
                    print(f"[WARNING] Failed to store AI response: {e}")
                    import traceback
                    traceback.print_exc()

            # Store in background thread
            store_thread = threading.Thread(target=store_response_async, daemon=True)
            store_thread.start()

            # Speak response in a separate thread (non-blocking)
            if self.tts:
                def speak_async():
                    try:
                        self.tts.speak(response)
                        print("[STATUS] TTS finished speaking")
                    except Exception as e:
                        print(f"[WARNING] TTS error: {e}")
                        import traceback
                        traceback.print_exc()
                
                tts_thread = threading.Thread(target=speak_async, daemon=True)
                tts_thread.start()
                # Don't wait for TTS - let it run in background
        except Exception as e:
            error_msg = f"Error getting AI response: {e}"
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            if self.tts:
                def speak_error_async():
                    try:
                        self.tts.speak("Sorry, I encountered an error.")
                    except:
                        pass
                threading.Thread(target=speak_error_async, daemon=True).start()

    def start(self) -> None:
        """Start the AI voice session."""
        print("\n" + "="*60)
        print("AI Voice Session Starting")
        print("="*60)
        print(f"Model: {self.model}")
        print(f"Session ID: {self.session_id}")
        print("Listening for speech... (Press Ctrl+C to stop)")
        print("="*60 + "\n")

        try:
            self.streamer.start(
                self._on_transcript,
                emit_partials=True,  # Enable live/partial transcripts
                on_audio_chunk=self._on_audio_chunk,
            )
            # Keep running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nStopping...")
            self.stop()

    def stop(self) -> None:
        """Stop the AI voice session."""
        self.streamer.stop()
        if self.voice_identifier:
            # Save voice database
            try:
                self.voice_identifier.save_database(self.voice_db_path)
                known_speakers = list(self.voice_identifier.known_speakers)
                if known_speakers:
                    print(f"Saved voice database to {self.voice_db_path}")
                    print(f"Registered speakers: {', '.join(known_speakers)}")
                else:
                    print(f"Voice database saved to {self.voice_db_path} (no speakers registered)")
            except Exception as e:
                print(f"Warning: Failed to save voice database: {e}")
                import traceback
                traceback.print_exc()
        print("Session stopped.")

    def _collect_voice_sample_async(self) -> None:
        """Asynchronously collect and update voice sample from audio buffer.
        
        Automatically detects if speaker exists, creates new entry if not, or improves existing one.
        Does NOT rely on _current_speaker - identifies fresh from audio each time.
        """
        if not self.voice_identifier:
            return
        
        # Get audio buffer (copy it to avoid locking for too long)
        with self._voice_buffer_lock:
            if not self._voice_audio_buffer:
                return
            audio_chunks = self._voice_audio_buffer.copy()
            self._voice_audio_buffer.clear()  # Clear buffer after copying
        
        # Process in background thread
        def collect_sample():
            try:
                import numpy as np
                
                # Combine all audio chunks
                if not audio_chunks:
                    return
                
                # Convert chunks to numpy array
                audio_arrays = []
                for chunk in audio_chunks:
                    audio = np.frombuffer(chunk, dtype="<i2").astype(np.float32)
                    if audio.size > 0:
                        audio /= 32768.0
                        audio_arrays.append(audio)
                
                if not audio_arrays:
                    return
                
                # Concatenate all audio
                full_audio = np.concatenate(audio_arrays)
                
                # Need at least 0.5 seconds of audio
                if len(full_audio) < self.voice_identifier.sample_rate * 0.5:
                    return
                
                # Try to identify the speaker from the full audio sample
                # Use reasonable threshold (0.70) - if recognized, use and improve; if not, create new
                result = self.voice_identifier.identify(full_audio, sample_rate=self.voice_identifier.sample_rate, threshold=0.70)
                
                # Determine the speaker name to use
                final_speaker_name = None
                is_new_speaker = False
                
                # Simple logic: if recognized as existing speaker, use and improve; otherwise create new
                if result.is_known and result.confidence >= 0.70:
                    # We recognized an existing speaker - use that name and improve their embedding
                    final_speaker_name = result.speaker
                    # Don't use "Unknown" as a valid speaker name - treat it as unknown
                    if final_speaker_name == "Unknown":
                        final_speaker_name = None
                    else:
                        # We found a known speaker - use it and improve their embedding
                        old_speaker = self._current_speaker
                        self._current_speaker = final_speaker_name
                        is_new_speaker = False  # This is an existing speaker - we'll improve their embedding
                        if old_speaker != final_speaker_name:
                            print(f"[VOICE DB] Recognized existing speaker: {final_speaker_name} (confidence: {result.confidence:.2f}, was: {old_speaker})")
                        else:
                            print(f"[VOICE DB] Confirmed speaker: {final_speaker_name} (confidence: {result.confidence:.2f})")
                
                # If we didn't recognize a speaker, create a new one
                if final_speaker_name is None:
                    # Not recognized - create a new speaker entry
                    # Find next available speaker name
                    known_speakers = set(self.voice_identifier.known_speakers)
                    # Remove "Unknown" from known speakers (we don't want to use it)
                    known_speakers.discard("Unknown")
                    
                    while f"Speaker_{self._next_speaker_id}" in known_speakers:
                        self._next_speaker_id += 1
                    final_speaker_name = f"Speaker_{self._next_speaker_id}"
                    is_new_speaker = True
                    self._next_speaker_id += 1
                    self._current_speaker = final_speaker_name
                    # Show why we created a new speaker
                    if result.speaker and result.confidence > 0:
                        print(f"[VOICE DB] Created new speaker: {final_speaker_name} (best match: {result.speaker} with confidence {result.confidence:.2f}, below threshold 0.70)")
                    else:
                        print(f"[VOICE DB] Created new speaker: {final_speaker_name} (no match found)")
                
                # Add/update voice sample (add_sample handles both new and existing)
                self.voice_identifier.add_sample(
                    final_speaker_name,
                    full_audio,
                    sample_rate=self.voice_identifier.sample_rate
                )
                
                duration = len(full_audio) / self.voice_identifier.sample_rate
                if is_new_speaker:
                    print(f"[VOICE DB] Created voice sample for '{final_speaker_name}' ({duration:.1f}s)")
                else:
                    # Get sample count to show improvement
                    if final_speaker_name in self.voice_identifier._prints:
                        sample_count = self.voice_identifier._prints[final_speaker_name].sample_count
                        print(f"[VOICE DB] Improved voice sample for '{final_speaker_name}' ({duration:.1f}s, {sample_count} samples)")
                    else:
                        print(f"[VOICE DB] Updated voice sample for '{final_speaker_name}' ({duration:.1f}s)")
                
                # Save database periodically (async, don't block)
                current_time = time.time()
                if current_time - self._last_voice_save_time >= self._voice_save_interval:
                    try:
                        self.voice_identifier.save_database(self.voice_db_path)
                        self._last_voice_save_time = current_time
                        print(f"[VOICE DB] Auto-saved voice database")
                    except Exception as e:
                        print(f"[VOICE DB] Warning: Failed to auto-save: {e}")
            except Exception as e:
                print(f"[VOICE DB] Error collecting voice sample: {e}")
                import traceback
                traceback.print_exc()
        
        # Run in background thread
        thread = threading.Thread(target=collect_sample, daemon=True, name="VoiceSampleCollector")
        thread.start()

    def register_speaker(self, name: str, duration: float = 3.0) -> None:
        """
        Register a new speaker or update existing speaker's voice sample.

        Args:
            name: Speaker name/username
            duration: Recording duration in seconds
        """
        if not self.voice_identifier:
            print("Voice identification not available")
            return

        print(f"\n{'='*60}")
        print(f"Registering speaker: {name}")
        print(f"Please speak for {duration} seconds...")
        print("="*60)
        try:
            self.voice_identifier.capture_and_sample(name, duration=duration)
            # Save immediately
            self.voice_identifier.save_database(self.voice_db_path)
            print(f"\n✓ Voice sample recorded and saved for '{name}'")
            print(f"✓ Voice database saved to {self.voice_db_path}")
            known_speakers = list(self.voice_identifier.known_speakers)
            print(f"✓ Total registered speakers: {len(known_speakers)}")
        except Exception as e:
            print(f"✗ Error recording voice sample: {e}")

    def add_speaker_sample(self, name: str, duration: float = 3.0) -> None:
        """Alias for register_speaker for backward compatibility."""
        self.register_speaker(name, duration)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="AI Voice Assistant with Memory")
    parser.add_argument(
        "--model",
        default="P2Wdisabled/lyra:7b",
        help="Ollama model name (default: P2Wdisabled/lyra:7b)",
    )
    parser.add_argument(
        "--session-id",
        default=f"session-{int(time.time())}",
        help="Session identifier",
    )
    parser.add_argument(
        "--memory-dir",
        default="memory_data",
        help="Memory storage directory",
    )
    parser.add_argument(
        "--voice-db",
        help="Path to voice database file",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable text-to-speech",
    )
    parser.add_argument(
        "--use-gtts",
        action="store_true",
        help="Use gTTS instead of pyttsx3",
    )
    parser.add_argument(
        "--system-prompt",
        help="System prompt for AI",
    )
    parser.add_argument(
        "--whisper-model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"],
        help="Whisper model size (default: large-v3-turbo)",
    )
    parser.add_argument(
        "--whisper-device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Whisper device (default: cpu)",
    )
    parser.add_argument(
        "--whisper-compute-type",
        default="int8",
        choices=["int8", "int8_float16", "int16", "float16", "float32"],
        help="Whisper compute type (default: int8)",
    )
    parser.add_argument(
        "--register-speaker",
        help="Register a new speaker (provide name, e.g., --register-speaker Alice)",
    )
    parser.add_argument(
        "--register-duration",
        type=float,
        default=3.0,
        help="Duration for speaker registration in seconds (default: 3.0)",
    )

    args = parser.parse_args()

    # Handle speaker registration
    if args.register_speaker:
        # Create a minimal session just for registration
        from modules.voice_identity import VoiceIdentifier
        
        voice_db = Path(args.voice_db) if args.voice_db else Path("voice_database.json")
        try:
            if voice_db.exists() and voice_db.stat().st_size > 10:
                identifier = VoiceIdentifier.load_database(voice_db)
            else:
                identifier = VoiceIdentifier()
            
            print(f"\n{'='*60}")
            print(f"Registering speaker: {args.register_speaker}")
            print(f"Please speak for {args.register_duration} seconds...")
            print("="*60)
            identifier.capture_and_sample(args.register_speaker, duration=args.register_duration)
            identifier.save_database(voice_db)
            print(f"\n✓ Speaker '{args.register_speaker}' registered successfully!")
            print(f"✓ Voice database saved to {voice_db}")
            known = list(identifier.known_speakers)
            print(f"✓ Total registered speakers: {len(known)}")
            sys.exit(0)
        except Exception as e:
            print(f"✗ Error registering speaker: {e}")
            sys.exit(1)

    # Create and start session
    session = AIVoiceSession(
        model=args.model,
        session_id=args.session_id,
        memory_storage_dir=Path(args.memory_dir),
        voice_db_path=Path(args.voice_db) if args.voice_db else None,
        use_tts=not args.no_tts,
        tts_use_gtts=args.use_gtts,
        system_prompt=args.system_prompt,
        whisper_model_size=args.whisper_model,
        whisper_device=args.whisper_device,
        whisper_compute_type=args.whisper_compute_type,
    )

    session.start()


if __name__ == "__main__":
    main()

