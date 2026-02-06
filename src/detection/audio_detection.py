import pyaudio
import numpy as np
import threading
from collections import deque
import time

# Whisper is optional; only required when whisper_enabled is True in config
whisper = None

class AudioMonitor:
    def __init__(self, config):
        self.config = config['detection']['audio_monitoring']
        self.sample_rate = self.config['sample_rate']
        self.chunk_size = 512  # 32ms chunks for low latency
        self.energy_threshold = self.config['energy_threshold']
        self.zcr_threshold = self.config['zcr_threshold']
        self.running = False
        self.audio_buffer = deque(maxlen=15)  # 480ms buffer
        self.alert_system = None
        self.alert_logger = None
        self.whisper_model = None

        if self.config.get('whisper_enabled', False):
            try:
                import whisper as _whisper
                globals()['whisper'] = _whisper
                self.whisper_model = _whisper.load_model(self.config.get('whisper_model', 'tiny.en'))
            except ImportError:
                if self.alert_logger is None:
                    import sys
                    print("Warning: openai-whisper not installed. Set whisper_enabled: false in config or run: pip install openai-whisper", file=sys.stderr)
        
    def start(self):
        """Start audio monitoring thread"""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop audio monitoring"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1)
            
    def _run(self):
        """Main audio processing loop"""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        try:
            while self.running:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16)
                self.audio_buffer.append(audio)
                
                if self._is_voice(audio):
                    self._handle_voice_detection()
                    
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
    
    def _is_voice(self, audio):
        """Ultra-fast voice detection"""
        audio_norm = audio / 32768.0
        
        # 1. Energy detection
        energy = np.mean(audio_norm**2)
        if energy < self.energy_threshold:
            return False
            
        # 2. Zero-crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(audio_norm))))
        if zcr > self.zcr_threshold:
            return False
            
        return True
    
    def _handle_voice_detection(self):
        """Process detected voice"""
        if self.alert_system:
            self.alert_system.speak_alert("VOICE_DETECTED")
            
        if self.alert_logger:
            self.alert_logger.log_alert("VOICE_DETECTED", "Voice activity detected")
            
        if self.config['whisper_enabled']:
            self._process_with_whisper()
    
    def _process_with_whisper(self):
        """Optional Whisper processing"""
        if self.whisper_model is None:
            return
        try:
            audio = np.concatenate(self.audio_buffer)
            result = self.whisper_model.transcribe(
                audio.astype(np.float32) / 32768.0,
                fp16=False,
                language='en'
            )
            
            text = result['text'].strip().lower()
            if any(word in text for word in ['help', 'answer', 'whisper']):
                if self.alert_system:
                    self.alert_system.speak_alert("SPEECH_VIOLATION")
                    
        except Exception as e:
            if self.alert_logger:
                self.alert_logger.log_alert("WHISPER_ERROR", str(e))