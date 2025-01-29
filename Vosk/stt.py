from vosk import Model, KaldiRecognizer  
import pyaudio  
import json  
import sys  

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
    CHUNK_SIZE = 8000  
    FORMAT = pyaudio.paInt16  

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

    # 加载模型  
    try:  
        model = Model("vosk-model-small-cn-0.22")  
        recognizer = KaldiRecognizer(model, FRAME_RATE)  
    except Exception as e:  
        print(f"错误：无法加载模型 - {str(e)}")  
        sys.exit(1)  

    # 初始化音频  
    audio = pyaudio.PyAudio()  
    stream = audio.open(format=FORMAT,  
                       channels=CHANNELS,  
                       rate=FRAME_RATE,  
                       input=True,  
                       input_device_index=device_index,  # 使用选择的设备  
                       frames_per_buffer=CHUNK_SIZE)  

    print("\n=" * 50)  
    print("语音转文字程序已启动")  
    print("开始说话吧！(按 Ctrl+C 停止)")  
    print("=" * 50)  

    try:  
        while True:  
            data = stream.read(4000, exception_on_overflow=False)  
            if recognizer.AcceptWaveform(data):  
                result = json.loads(recognizer.Result())  
                text = result.get('text', '')  
                if text:  
                    print(f"识别结果: {text}")  
            else:  
                partial = json.loads(recognizer.PartialResult())  
                partial_text = partial.get('partial', '')  
                if partial_text:  
                    print(f"正在识别: {partial_text}", end='\r')  

    except KeyboardInterrupt:  
        print("\n\n程序已停止")  
    finally:  
        stream.stop_stream()  
        stream.close()  
        audio.terminate()  

if __name__ == "__main__":  
    start_voice_to_text()