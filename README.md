# Description

Darknet을 통해 YOLO 모델을 위한 커스텀 데이터를 만들고, 학습하고, 테스트하는 방법을 설명합니다.   
딥러닝과 Object Detection에 대한 원리 혹은 개념 설명은 포함하지 않습니다.   

# Labelling

라벨링을 위한 툴은 아래 링크의 프로그램을 추천드립니다. 본 예제는 이 프로그램을 통해 진행하였습니다.   
우선, ```train``` 이라는 디렉토리를 생성한 뒤, 안에 학습하고자 하는 이미지를 저장합니다.   
이후, ```obj.names```라는 파일을 생성합니다. 이 파일에는 검출하고자 하는 객체의 이름이 쓰여져 있어야 합니다. 가령 사람과 고양이를 검출하는 모델을 학습하고자 한다면, 본 파일의 내용은 아래와 같습니다.   
```
person
cat
```
혹은 다음과 같이 될 수도 있습니다.
```
human
the-cutest-creature
```
이처럼 객체명은 모델에게 "이건 a야. 이건 b야." 라고 알려주는 용도입니다. 그러므로 객체명은 단지 내가 알아볼 수 있기만 하면 됩니다.   
   
이제 앞서 말한 프로그램을 설치하고 실행할 차례입니다. 방법은 다음과 같습니다.   

1. Open Files 버튼을 클릭한 뒤 ```train``` 디렉토리를 선택합니다.
2. 이후 ```data/obj.names``` 파일을 선택합니다.
3. 라벨링을 진행합니다.

<img width="1640" alt="example" src="https://user-images.githubusercontent.com/49421142/109983005-c66a4200-7d45-11eb-8e42-61f26f749098.png">

하나의 이미지에 대해 라벨링을 완료하고 다음 사진으로 넘어가면 라벨링 텍스트파일이 자동으로 저장됩니다. 위 이미지 (```train/000001.jpg```)에 대해 저장된 라벨링 텍스트파일 ```train/000001.txt```은 다음과 같은 내용으로 저장되었습니다.
```
0 0.741016 0.563019 0.511719 0.862881
1 0.186328 0.522853 0.225781 0.331025
1 0.406250 0.546399 0.239062 0.378116
1 0.257812 0.355263 0.228125 0.209141
1 0.426953 0.378809 0.100781 0.148199
```
여기에서 person의 클래스 인덱스가 0, cat의 클래스 인덱스가 1인 것을 알 수 있습니다. 이는 ```obj.names``` 에 작성된 순서입니다.

# Struct Custom Dataset

```dataset``` 라는 디렉토리를 하나 만듭니다. 본 디렉토리는 다음과 같이 구성되어야 합니다.   
```
   data
    ├── obj.data
    ├── obj.names
    ├── test
    │   ├── 000001.jpg
    │   ├── 000001.txt
    │   ├── 000002.jpg
    │   └── 000002.txt
    ├── test.txt
    ├── train
    │   ├── 000001.jpg
    │   ├── 000001.txt
    │   ├── 000002.jpg
    │   ├── 000002.txt
    │   ├── 000003.jpg
    │   ├── 000003.txt
    │   ├── 000004.jpg
    │   ├── 000004.txt
    │   ├── 000005.jpg
    │   └── 000005.txt
    └── train.txt
```
이제 각 파일이 어떻게 구성되어있는지 살펴보겠습니다.

### Directory: data/train
학습에 사용될 이미지와 그 이미지의 라벨링이 저장된 디렉토리입니다. 이미지와 그에 해당하는 라벨링 텍스트파일의 이름이 같음에 유의해야 합니다.   

### File: data/train.txt
```data/train``` 디렉토리 내에 있는 이미지 파일의 목록이 쓰여져있습니다. 위 디렉토리를 예로 들면 다음과 같습니다.
```
data/train/000001.jpg
data/train/000002.jpg
data/train/000003.jpg
data/train/000004.jpg
data/train/000005.jpg
```
경로가 ```data```부터 시작함에 유의해야 합니다. Darknet 학습 실행 시 본 텍스트파일을 통해 학습이미지를 불러옵니다. 그러므로 ```data/train``` 내 이미지는 번호 순일 필요는 없습니다.

### Directory: data/test
테스트에 사용될 이미지와 그 이미지의 라벨링이 저장된 디렉토리입니다. 이미지와 그에 해당하는 라벨링 텍스트파일의 이름이 같음에 유의해야 합니다.   
※ *통상적으로 ```train```과 ```test```의 비율을 9:1로 맞추면 됩니다. 물론, 테스트에 사용되는 이미지는 학습에 사용되는 이미지들과 중복이 없어야 하며, 학습에 사용된 이미지와 비슷한 구성으로 존재해야 합니다. 가령 학습에는 서있는 고양이와 앉아있는 고양이, 그리고 벌러덩 누워있는 고양이 등의 이미지가 골고루 존재하지만 테스트에 사용되는 이미지에는 누워있는 고양이밖에 존재하지 않을 경우 테스트 결과가 정확하다고 판단할 수 없습니다. 하지만 학습 자체를 경험해보기 위해 10장 정도의 매우 적은 이미지만을 학습할 경우, Underfitting 으로 인해 학습에 사용되지 않은 전혀 새로운 검출이 제대로 되지 않을 확률이 높으므로 그냥 학습에 사용된 이미지를 동시에 테스트에도 사용하여 검출여부만 따지면 될 것 같습니다.*

### File: data/test.txt
```data/test``` 디렉토리 내에 있는 이미지 파일의 목록이 쓰여져있습니다. 위 디렉토리를 예로 들면 다음과 같습니다.
```
data/train/000001.jpg
data/train/000002.jpg
```

### File: data/obj.names
만약 내 데이터셋에는 총 2개의 객체가 라벨링되어있는데 본 파일에는 1개 혹은 3개 이상의 클래스명이 작성되어있다면 학습이 진행되지 않습니다.

### File: data/obj.data
지금까지 구성한 모든 파일의 경로를 알려주는 파일입니다. 본 예제를 예로 들면 파일의 내용은 다음과 같습니다.
```
classes=2
train=data/train.txt
valid=data/test.txt
names=data/obj.names
backup=backup/
```

+ ```classes```는 ```data/obj.names```에 적힌 클래스명의 수를 의미합니다.   
+ ```train```는 ```data/train.txt``` 파일의 경로를 의미합니다. 경로가 ```data```부터 시작함에 유의해야 합니다.
+ ```valid```는 ```data/test.txt``` 파일의 경로를 의미합니다. 경로가 ```data```부터 시작함에 유의해야 합니다.
+ ```names```는 ```data/obj.names``` 파일의 경로를 의미합니다. 경로가 ```data```부터 시작함에 유의해야 합니다.
+ ```backup```는 학습된 모델이 저장될 경로를 의미합니다.

# (Optional) GPU Requirements in Ubuntu

+ 본 장은 nVidia의 GPU가 존재하는 우분투 환경에서 CUDA 를 사용하여 학습을 가속화하기 위해 그래픽드라이버 및 CUDA, cuDNN 을 설치하는 과정을 설명합니다. 우분투에 Nvidia 드라이버가 설치되어있지 않아야 합니다.   
+ **Ubuntu 16.04 및 2020년 9월 기준입니다.** Ubuntu 20.04 를 사용할 경우 본 장을 무시하고 다음 링크를 참고하여 설치하시기 바랍니다. CUDA는 10.0 이상, cuDNN은 7.0 이상의 버전이어야 합니다.
    + [CUDA 설치 가이드](https://linuxconfig.org/how-to-install-cuda-on-ubuntu-20-04-focal-fossa-linux)
    + [cuDNN 다운로드 nVidia 공식 페이지](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn_741/cudnn-install/index.html)

## CUDA 10.0
1. Download [CUDA Toolkit 10.0 in nvidia site](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal).

2. Open downloaded file.
```console
$ sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
```

3. (Optional) Add apt-key from cuda-repo.
```console
$ sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
```

4. Install cuda.
```console
$ sudo apt-get update
$ sudo apt-get install cuda-10-0
```

5. Configure .bashrc.
```console
$ vim ~/.bashrc
```
and add below two lines.
```console
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

6. Reboot system and confirm installation.
```console
$ sudo reboot now
$ nvcc --version
```

## cnDNN 7.5
1. Download [cuDNN 7.5 in nvidia site](https://developer.nvidia.com/cudnn).
- cuDNN Runtime Library for Ubuntu18.04 (Deb)
- cuDNN Developer Library for Ubuntu18.04 (Deb)

2. Open downloaded files.
~~~console 
$ sudo dpkg -i <filename>
~~~

# Build Darknet

본 예제에 존재하는 darknet의 경우 [AlexayAB의 Darknet](https://github.com/AlexeyAB/darknet)을 클론해왔습니다.   

### Download Pre-trained Model
https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137 를 다운로드하여 darknet 디렉토리 안에 저장합니다. 리눅스의 경우 다음을 통해 다운받을 수 있습니다.
```console
$ wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
```

### Compile and Build Darknet
빌드 전 ```darknet/Makefile```을 본인의 환경에 맞게 수정해야 합니다.   
   
```Makefile```의 상위 9줄은 다음과 같습니다.
```
GPU=0
CUDNN=0
CUDNN_HALF=0
OPENCV=0
AVX=0
OPENMP=0
LIBSO=0
ZED_CAMERA=0
ZED_CAMERA_v2_8=0
```
+ nVidia의 GPU를 사용할 경우 ```GPU=1```, ```CUDNN=1``` 로 변경합니다.
+ ```CUDNN_HALF```의 경우 Xavier 등의 특수한 GPU 사용을 위해 존재합니다. 일반적인 GTX 혹은 RTX 계열 GPU의 경우 0으로 두면 됩니다.
+ 학습 컴퓨터에 OpenCV가 설치되어있다면 ```OPENCV=1```, 없다면 ```OPENCV=0```으로 두면 됩니다. 학습 및 테스트의 용도로 OpenCV는 필요없으며, 검출결과 이미지를 직접 확인하고 싶다면 필요합니다.
+ ```AVX```와 ```OPENMP```는 CPU 가속을 하기 위함입니다. ```OPENMP=1```로 한 뒤 에러가 나면 다시 0으로 두고```AVX=1```로 하면 됩니다. GPU를 사용한다면 이 두 변수는 모두 0으로 해도 되나 확실하진 않습니다.
+ ```ZED_CAMERA```와 ```ZED_CAMERA_v2_8```은 제드카메라라는 특수한 장비를 위한 설정 같습니다. 저도 잘 모릅니다..   
   
다음으로 20번째 줄부터 다음과 같은 내용이 있습니다.
```Makefile
ARCH= -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
	    -gencode arch=compute_61,code=[sm_61,compute_61]

OS := $(shell uname)

# GeForce RTX 3070, 3080, 3090
# ARCH= -gencode arch=compute_86,code=[sm_86,compute_86]
...
```
   
GPU를 사용할 경우, 주석을 보며 본인의 GPU에 맞는 라인의 ```ARCH= ``` 주석을 해제하면 됩니다.
   
다음부터는 리눅스와 윈도우를 구별하여 설명하겠습니다.

## Linux
darknet 디렉토리 내에서 ```make``` 커맨드로 컴파일 및 빌드를 진행합니다.
```console
darknet$ make
```
   
성공하였다면, yolov4.cfg 파일을 수정해줘야 합니다. 본 레포지터리 내에 있는 darknet의 경우 ```darknet/yolov4.cfg```로 저장되어있지만, AlexayAB의 darknet은 ```darknet/cfg/yolov4.cfg``` 내에 있습니다.   
```yolov4.cfg```의 상위 17~22번째 줄(두 번째 문단)은 다음과 같습니다.
```shell
learning_rate=0.0013
burn_in=1000
max_batches = 500500
policy=steps
steps=400000,450000
scales=.1,.1
```   
여기에서 다음과 같은 내용을 변경합니다.
+ ```max_batches```는 총 학습 횟수를 의미합니다. 기본적으로 ```내 데이터에서 검출하고자 하는 객체의 수 * 2000``` 의 수를 지정합니다. 본 예제에서는 2개의 객체를 학습하고자 하므로 ```4000```으로 변경합니다. 
+ ```steps```는 총 학습 횟수의 80%, 90% 를 의미합니다. 본 예제에서 총 학습횟수는 ```max_batches=4000```이므로 ```steps=3200,3600```으로 수정합니다.   
   
이후 상위 8개 줄(첫 번째 문단)을 수정할 차례입니다.
```shell
[net]
batch=64
subdivisions=8
# Training
#width=512
#height=512
width=608
height=608
```
+ ```batch```는 한 번의 학습에 사용되는 데이터의 수를 의미합니다. 변경할 필요 없습니다.
+ ```subdivisions```는 한 번의 학습에서 ```batch```를 다시 몇 분할로 나누어서 학습할 것인지를 나타내는 값입니다. 64 이하인 2의 거듭제곱 형태여야 하며, 추후 학습시 메모리가 부족하다며 학습이 종료될 경우 이 값을 늘려야 합니다.
+ ```width=608```과 ```height=608```을 주석처리 한 후 ```width=512```과 ```height=512```의 주석을 해제합니다.
   
이제 ```classes=80```인 부분을 찾습니다. 총 세 줄이 존재하며, 각각 968, 1056, 1144 번 째 줄에 있습니다. 이를 내가 학습하고자 하는 객체의 수로 변경하여줍니다. 본 예제의 경우 2이므로 ```classes=2```가 됩니다.   
또한, 각 ```classes``` 바로 위에 있는 ```filters=255```를 ```(객체수 + 5) * 3```으로 변경해줍니다. 각각 961, 1049, 1137 번 째 줄에 있습니다. 본 예제의 경우 ```(2 + 5) * 3 = 21``` 이므로 ```filters=21```이 됩니다.   

위 모든 사항을 진행하였다면, 이제 학습을 진행할 차례입니다. 위에서 만들었던 ```data``` 디렉토리를 ```darknet``` 디렉토리 내로 이동한 뒤, ```darknet``` 디렉토리 내에서 다음과 같이 학습용 darknet을 실행합니다.

```console
darknet$ nohup ./darknet detector train data/obj.data yolov4.cfg yolov4.conv.137 -dont_show -mjpeg_port 8090 -map &
```   
각 파라미터는 다음을 의미합니다.
```console
darknet$ nohup ./darknet detector train ".data 경로" ".cfg 경로" "pre-trained 모델 경로" -dont_show -mjpeg_port 8090 -map &
```   
+ 학습은 백그라운드에서 진행되므로 터미널을 종료해도 됩니다. 
+ ```darknet/nohup.out``` 파일에 학습 로그가 실시간으로 저장됩니다.
+ 학습 그래프는 ```darknet/chart.png```에 실시간으로 저장됩니다. 이 학습 그래프는 웹브라우저에서 http://localhost:8090 에 접속하여 확인할 수 있습니다.
+ 실행 후 약 30초 정도 기다린 뒤 로그를 확인하여 Error 등이 없는지, 혹은 ```ps -ef``` 명령어를 통해 프로세스가 잘 작동되고 있는지 확인하여 학습이 잘 진행중임을 확인해야 합니다.
+ ```backup/``` 디렉토리에 1000번 단위로 학습된 모델이 저장됩니다.



### 테스트
~~~
./darknet detector map ".data 경로" ".cfg 경로" "학습된 모델 경로" -dont_show -ext_output < data/test.txt > result.txt
~~~

- backup/ 에서 best 가중치파일 사용
- result.txt에 결과값이 저장됨

* 결과이미지를 얻고 싶다면 다음 darknet을 사용   
https://github.com/vincentgong7/VG_AlexeyAB_darknet

./darknet detector batch <.data> <.cfg> <.weights> -dont_show batch <input images dir/> <output images dir/> -ext_output > result.txt

- 이를 사용하려면 테스트 이미지인 002566.jpg ~ 002850.jpg 를 data/dataset/images/ 가 아닌 다른 디렉토리에 따로 저장해야함
