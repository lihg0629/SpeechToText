# Speech to text (vosk sample)
## 1. install 
```
brew install portaudio  
pip3 install vosk pyaudio  
```

## 2. [test](./testVosk.py)
```
# test.py  
from vosk import Model, KaldiRecognizer  
import pyaudio  

print("Vosk 导入成功!")  
print("PyAudio 导入成功!")
```

## 3. download
```
https://alphacephei.com/vosk/models
```

## 4. folder tree

```
.
├── readme.md
├── stt.py
├── testVosk.py
├── vosk-model-small-cn-0.22
```

## 5. demerit
> do not support language recognition and translation