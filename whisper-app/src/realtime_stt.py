import whisper
import pyaudio
import numpy as np
import webrtcvad
import threading
import collections
import queue
import time

class RealtimeSTT:
    def __init__(self):
        self.model = whisper.load_model("base")
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.vad = webrtcvad.Vad(3)  # 灵敏度设置为3
        self.buffer = collections.deque(maxlen=30)  # 30帧缓冲区
        self.running = False
        
    def start_recording(self, device_index=None):
        CHUNK_SIZE = 480  # 30ms at 16kHz
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            input_device_index=device_index
        )
        
        self.running = True
        
        def audio_callback():
            while self.running:
                try:
                    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    is_speech = self.vad.is_speech(data, RATE)
                    if is_speech:
                        self.buffer.append(data)
                    elif len(self.buffer) > 0:
                        audio_data = b''.join(self.buffer)
                        self.audio_queue.put(audio_data)
                        self.buffer.clear()
                except Exception as e:
                    print(f"录音错误: {e}")
                    
        def process_audio():
            while self.running:
                try:
                    if not self.audio_queue.empty():
                        audio_data = self.audio_queue.get()
                        np_audio = np.frombuffer(audio_data, dtype=np.int16)
                        float_audio = np_audio.astype(np.float32) / 32768.0
                        result = self.model.transcribe(float_audio, language='zh')
                        if result['text'].strip():
                            self.result_queue.put(result['text'])
                    else:
                        time.sleep(0.1)
                except Exception as e:
                    print(f"处理错误: {e}")
        
        recording_thread = threading.Thread(target=audio_callback)
        processing_thread = threading.Thread(target=process_audio)
        
        recording_thread.start()
        processing_thread.start()
        
        try:
            while True:
                if not self.result_queue.empty():
                    text = self.result_queue.get()
                    print(f"\r{text}", end='¥n', flush=True)
        except KeyboardInterrupt:
            self.running = False
            recording_thread.join()
            processing_thread.join()
            stream.stop_stream()
            stream.close()
            p.terminate()

if __name__ == "__main__":
    stt = RealtimeSTT()
    stt.start_recording()