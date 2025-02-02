import whisper
import pyaudio
import numpy as np
import torch

def list_audio_devices():
    p = pyaudio.PyAudio()
    device_count = p.get_device_count()
    for i in range(device_count):
        dev = p.get_device_info_by_index(i)
        if dev.get('maxInputChannels') > 0:  # 只显示输入设备
            print(f"设备 {i}: {dev.get('name')}")
    p.terminate()
    return device_count

def start_voice_to_text():
    # 设置音频参数
    FRAME_RATE = 16000
    CHANNELS = 1
    CHUNK_SIZE = 16000  # 调整为1秒的数据
    FORMAT = pyaudio.paInt16
    RECORD_SECONDS = 3  # 每次处理3秒的音频

    # 显示可用设备
    print("\n可用的音频输入设备：")
    device_count = list_audio_devices()

    # 让用户选择设备
    device_index = None
    while device_index is None:
        try:
            device_id = input("\n请选择输入设备编号 (输入数字): ")
            device_id = int(device_id)
            if 0 <= device_id < device_count:
                device_index = device_id
            else:
                print("无效的设备编号，请重试")
        except ValueError:
            print("请输入有效的数字")

    # 加载Whisper模型
    model = whisper.load_model("base")  # 可以选择不同的模型，如 "small", "medium", "large"

    # 开始录音
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=FRAME_RATE, input=True,
                    frames_per_buffer=CHUNK_SIZE, input_device_index=device_index)

    print("开始说话... (按 Ctrl+C 停止)")

    try:
        while True:
            frames = []
            for _ in range(0, int(FRAME_RATE / CHUNK_SIZE * RECORD_SECONDS)):
                try:
                    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    frames.append(data)
                except OSError as e:
                    if e.errno == -9981:
                        print("输入溢出，跳过当前块")
                    else:
                        raise e

            # 将帧数据转换为numpy数组
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
            
            # 使用whisper的transcribe方法
            result = model.transcribe(audio_data, language='zh', fp16=False)
            print(f"\r{result['text']}", end='\n', flush=True)

    except KeyboardInterrupt:
        print("\n停止录音")
    finally:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    start_voice_to_text()