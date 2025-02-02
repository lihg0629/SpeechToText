import whisper  
import pyaudio  
import wave  
import numpy as np  
import tempfile  
import os  
from datetime import datetime  

def list_audio_devices():  
    p = pyaudio.PyAudio()  
    device_count = p.get_device_count()  
    for i in range(device_count):  
        dev = p.get_device_info_by_index(i)  
        if dev.get('maxInputChannels') > 0:  
            print(f"设备 {i}: {dev.get('name')}")  
    p.terminate()  
    return device_count  

def record_audio(device_index, duration=10):  
    # 音频设置  
    CHUNK = 1024  
    FORMAT = pyaudio.paInt16  
    CHANNELS = 1  
    RATE = 16000  

    p = pyaudio.PyAudio()  
    stream = p.open(format=FORMAT,  
                   channels=CHANNELS,  
                   rate=RATE,  
                   input=True,  
                   input_device_index=device_index,  
                   frames_per_buffer=CHUNK)  

    print(f"\n开始录音 {duration} 秒...")  
    frames = []  

    # 录制音频  
    for i in range(0, int(RATE / CHUNK * duration)):  
        data = stream.read(CHUNK, exception_on_overflow=False)  
        frames.append(data)  
        # 显示进度条  
        progress = (i + 1) / (RATE / CHUNK * duration) * 100  
        print(f"\r录音进度: {progress:.1f}%", end="")  

    print("\n录音结束!")  

    stream.stop_stream()  
    stream.close()  
    p.terminate()  

    # 创建临时文件  
    temp_dir = tempfile.gettempdir()  
    temp_file = os.path.join(temp_dir, f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")  

    # 保存音频文件  
    wf = wave.open(temp_file, 'wb')  
    wf.setnchannels(CHANNELS)  
    wf.setsampwidth(p.get_sample_size(FORMAT))  
    wf.setframerate(RATE)  
    wf.writeframes(b''.join(frames))  
    wf.close()  

    return temp_file  

def transcribe_audio(model, audio_file):  
    try:  
        # 转写音频  
        result = model.transcribe(audio_file)  
        return result  
    except Exception as e:  
        print(f"转写出错: {str(e)}")  
        return None  

def start_voice_to_text():  
    # 显示可用设备  
    print("\n可用的音频输入设备：")  
    device_count = list_audio_devices()  

    # 选择设备  
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
    print("\n正在加载Whisper模型...")  
    model = whisper.load_model("base")  
    print("模型加载完成！")  

    while True:  
        try:  
            # 设置录音时长  
            duration = input("\n请输入要录音的秒数 (直接回车默认10秒): ")  
            duration = int(duration) if duration.strip() else 10  

            # 录制音频  
            audio_file = record_audio(device_index, duration)  

            # 转写音频  
            print("\n正在转写音频...")  
            result = transcribe_audio(model, audio_file)  

            if result:  
                print("\n识别结果:")  
                print("-" * 50)  
                print(f"检测到的语言: {result['language']}")  
                print(f"文字内容: {result['text']}")  
                print("-" * 50)  

            # 删除临时文件  
            os.remove(audio_file)  

            # 询问是否继续  
            choice = input("\n是否继续录音？(y/n): ")  
            if choice.lower() != 'y':  
                break  

        except KeyboardInterrupt:  
            print("\n程序已停止")  
            break  
        except Exception as e:  
            print(f"\n发生错误: {str(e)}")  
            break  

if __name__ == "__main__":  
    start_voice_to_text()