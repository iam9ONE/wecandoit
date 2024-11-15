https://velog.io/@tjdwjdgus99/YOLOv8-%EA%B0%9C%EB%B0%9C%ED%99%98%EA%B2%BD-%EC%84%A4%EC%B9%98-cuda-11.8-cndnn-8.7.0 여기 블로그 참조

아나콘다 설치

yolov8 설치(둘 중 하나) 
pip install ultralytics
https://github.com/ultralytics/ultralytics 여기서 파일 받은 후 ultralytics 폴더에 넣어줌

설치 확인
python

from IPython import display
display.clear_output()
import ultralytics
ultralytics.checks()

파이토치 설치
# PyTorch 설치 (CUDA 11.8)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
(중간에 오류 뜨면 torchvision 이거 버전 바꿔주기--->gpt이용)

cudatoolkit 설치
conda install cudatoolkit

다시 확인
python

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

import torch
torch.cuda.is_available()

import tensorflow as tf

if tf.test.is_gpu_available():
    print("GPU 사용 가능")
    # GPU 장치 목록 확인
    print("사용 가능한 GPU 장치 목록:")
    for device in tf.config.experimental.list_physical_devices('GPU'):
        print(device)
else:
    print("GPU 사용 불가능")


data.yaml 파일 만들기
경로 지정후

아나콘다로 실행ㄱㄱ
(v8) C:\Users\admin>python
Python 3.8.20 (default, Oct  3 2024, 15:19:54) [MSC v.1929 64 bit (AMD64)] :: Anaconda, Inc. on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> from ultralytics import YOLO
>>>
>>> # YOLO 모델을 초기화합니다
>>> model = YOLO('yolov8n.pt')
>>>
>>> # 모델 학습을 시작합니다
>>> model.train(data='C:/Users/admin/anaconda3/envs/v8/ultralytics-main/ultralytics/cfg/datasets/data.yaml',
...             epochs=50,
...             imgsz=640)

deepsort
https://velog.io/@junwoo0525/YOLOv8%EC%9D%84-OpenCV%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%98%EC%97%AC-%EB%8F%99%EC%9E%91%EC%8B%9C%ED%82%A4%EA%B8%B0
이거 보고 참조하면 될듯


