# Whisper App

这是一个使用Whisper库实现实时语音转文本的Python应用程序。

## 项目结构

```
whisper-app
├── src
│   ├── stt.py          # 应用程序的主要代码
├── requirements.txt    # 项目所需的Python库
└── README.md           # 项目的文档
```

## 安装

1. 克隆这个仓库或下载源代码。
2. 确保你已经安装了Python 3.7或更高版本。
3. 在项目根目录下，运行以下命令安装所需的依赖项：

```
pip install -r requirements.txt
```

## 使用

1. 运行 `src/stt.py` 文件：
```
python src/stt.py
```
2. 按照提示选择音频输入设备。
3. 开始说话，程序将实时输出文本。

## 依赖项

- Whisper
- PyAudio
- Vosk

请确保在使用前安装所有依赖项。


## 参考
