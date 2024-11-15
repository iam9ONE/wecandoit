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

여기서 망한 점...
Results saved to C:\Users\admin\runs\detect\train11
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Users\admin\anaconda3\envs\v8\lib\site-packages\ultralytics\utils\__init__.py", line 216, in __repr__
    return self.__str__()
  File "C:\Users\admin\anaconda3\envs\v8\lib\site-packages\ultralytics\utils\__init__.py", line 204, in __str__
    v = getattr(self, a)
  File "C:\Users\admin\anaconda3\envs\v8\lib\site-packages\ultralytics\utils\__init__.py", line 221, in __getattr__
    raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")
AttributeError: 'DetMetrics' object has no attribute 'curves_results'. See valid attributes below.

    Utility class for computing detection metrics such as precision, recall, and mean average precision (mAP) of an
    object detection model.

    Args:
        save_dir (Path): A path to the directory where the output plots will be saved. Defaults to current directory.
        plot (bool): A flag that indicates whether to plot precision-recall curves for each class. Defaults to False.
        on_plot (func): An optional callback to pass plots path and data when they are rendered. Defaults to None.
        names (dict of str): A dict of strings that represents the names of the classes. Defaults to an empty tuple.

    Attributes:
        save_dir (Path): A path to the directory where the output plots will be saved.
        plot (bool): A flag that indicates whether to plot the precision-recall curves for each class.
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
        names (dict of str): A dict of strings that represents the names of the classes.
        box (Metric): An instance of the Metric class for storing the results of the detection metrics.
        speed (dict): A dictionary for storing the execution time of different parts of the detection process.

    Methods:
        process(tp, conf, pred_cls, target_cls): Updates the metric results with the latest batch of predictions.
        keys: Returns a list of keys for accessing the computed detection metrics.
        mean_results: Returns a list of mean values for the computed detection metrics.
        class_result(i): Returns a list of values for the computed detection metrics for a specific class.
        maps: Returns a dictionary of mean average precision (mAP) values for different IoU thresholds.
        fitness: Computes the fitness score based on the computed detection metrics.
        ap_class_index: Returns a list of class indices sorted by their average precision (AP) values.
        results_dict: Returns a dictionary that maps detection metric keys to their computed values.
        curves: TODO
        curves_results: TODO

이게 무슨 에러일까.... 잠이나 디비자보도록하자자

deepsort
https://velog.io/@junwoo0525/YOLOv8%EC%9D%84-OpenCV%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%98%EC%97%AC-%EB%8F%99%EC%9E%91%EC%8B%9C%ED%82%A4%EA%B8%B0
이거 보고 참조하면 될듯



