# Description

Darknet을 통해 YOLO 모델을 위한 커스텀 데이터를 만들고, 학습하고, 테스트하는 방법을 설명합니다.   
딥러닝과 Object Detection에 대한 원리 혹은 개념 설명은 포함하지 않습니다.   
+ [Labelling](https://github.com/wooseokyourself/how-to-yolo#labelling)
+ [Struct Custom Dataset](https://github.com/wooseokyourself/how-to-yolo#struct-custom-dataset)
+ [Modify yolov4.cfg](https://github.com/wooseokyourself/how-to-yolo#modify-yolov4cfg)
+ [(Optional) GPU Requirements in Linux](https://github.com/wooseokyourself/how-to-yolo#optional-gpu-requirements-in-linux)
+ Build Darknet
    + [Download Pre-trained Model](https://github.com/wooseokyourself/how-to-yolo#download-pre-trained-model)
    + [Compile and Build Darknet : Linux (using ```make```) - Not Recommended](https://github.com/wooseokyourself/how-to-yolo#compile-and-build-darknet--linux-using-make---not-recommended)
    + [Compile and Build Darknet : Linux (using ```CMake```) - Recommended](https://github.com/wooseokyourself/how-to-yolo#compile-and-build-darknet--linux-using-cmake---recommended)
    + [Compile and Build Darknet : Windows (using ```CMake```) - Not Recommended](https://github.com/wooseokyourself/how-to-yolo#compile-and-build-darknet--windows-using-cmake---not-recommended)
    + [Compile and Build Darknet : Windows (using ```vcpkg```) - Recommended](https://github.com/wooseokyourself/how-to-yolo#compile-and-build-darknet--windows-using-cmake---recommended)
+ Train and Test
    + [Run Train](https://github.com/wooseokyourself/how-to-yolo#run-train)
    + [Run Test](https://github.com/wooseokyourself/how-to-yolo#run-test)

# Labelling

라벨링을 위한 툴은 아래 링크의 프로그램을 추천드립니다. 본 예제는 이 프로그램을 통해 진행하였습니다.   
https://github.com/developer0hye/Yolo_Label   
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

```data``` 라는 디렉토리를 하나 만듭니다. 본 디렉토리는 다음과 같이 구성되어야 합니다.   
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

# Modify yolov4.cfg
yolov4.cfg 파일을 본인의 데이터셋에 맞게 수정해줘야 합니다. 본 레포지터리 내에 있는 darknet의 경우 ```darknet/yolov4.cfg```에 저장되어있지만, AlexayAB의 darknet은 ```darknet/cfg/yolov4.cfg``` 내에 있습니다.   
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

# (Optional) GPU Requirements in Linux

+ 본 장은 nVidia의 GPU가 존재하는 우분투 환경에서 CUDA 를 사용하여 학습을 가속화하기 위해 그래픽드라이버 및 CUDA, cuDNN 을 설치하는 과정을 설명합니다. 우분투를 기준으로 하였으며, 우분투에 Nvidia 드라이버가 설치되어있지 않아야 합니다.   
+ 윈도우로는 한 번도 해보지 않아서 잘 모르겠습니다. 윈도우에서도 CUDA와 cuDNN을 설치하면 됩니다. 
+ **Ubuntu 16.04 및 2020년 9월 기준입니다.** Ubuntu 20.04 를 사용할 경우 본 장을 무시하고 다음 링크를 참고하여 설치하시기 바랍니다. CUDA는 10.0 이상, cuDNN은 7.0 이상의 버전이어야 합니다.
    + [CUDA 설치 가이드](https://linuxconfig.org/how-to-install-cuda-on-ubuntu-20-04-focal-fossa-linux)
    + [cuDNN 다운로드 nVidia 공식 페이지](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn_741/cudnn-install/index.html)

### CUDA 10.0
1. Download [CUDA Toolkit 10.0 in nvidia site](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal).

2. Open downloaded file.
```console
you@you:~ $ sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
```

3. (Optional) Add apt-key from cuda-repo.
```console
you@you:~ $ sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
```

4. Install cuda.
```console
you@you:~ $ sudo apt-get update
you@you:~ $ sudo apt-get install cuda-10-0
```

5. Configure .bashrc.
```console
you@you:~ $ vim ~/.bashrc
```
and add below two lines.
```console
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

6. Reboot system and confirm installation.
```console
you@you:~ $ sudo reboot now
you@you:~ $ nvcc --version
```

### cnDNN 7.5
1. Download [cuDNN 7.5 in nvidia site](https://developer.nvidia.com/cudnn).
- cuDNN Runtime Library for Ubuntu18.04 (Deb)
- cuDNN Developer Library for Ubuntu18.04 (Deb)

2. Open downloaded files.
~~~console 
you@you:~ $ sudo dpkg -i <filename>
~~~

# Build Darknet

##### **본 예제에 존재하는 darknet의 경우 [AlexayAB의 Darknet](https://github.com/AlexeyAB/darknet)을 클론해왔으며, 이 레포지토리의 ```README.md``` 를 번역한 수준에 지나지 않습니다. 자세한 내용은 본 레포지토리의 설명을 참고하시는 걸 추천드립니다.([```darknet/READMD.md```](https://github.com/wooseokyourself/how-to-yolo/tree/main/darknet))**       

### Download Pre-trained Model
https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137 를 다운로드하여 darknet 디렉토리 안에 저장합니다. 리눅스의 경우 다음을 통해 다운받을 수 있습니다.
```console
you@you:~ $ wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
```

## Compile and Build Darknet : Linux (using ```make```) - Not Recommended

```darknet/Makefile```을 본인의 환경에 맞게 수정해야 합니다.   
   
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
   
이후 darknet 디렉토리 내에서 ```make``` 커맨드로 컴파일 및 빌드를 진행합니다.
```console
you@you:~/darknet $ make
```

## Compile and Build Darknet : Linux (using ```CMake```) - Recommended

위의 ```Makefile```에서 내 환경에 맞는 변수를 직접 수정했던 것과 달리, ```CMake```를 사용할 경우 환경변수들을 자동으로 설정해준다는 편리함이 있습니다.   
단지 ```darknet``` 디렉토리 내에서 ```./build.sh```를 실행하기만 하면 됩니다.
```console
you@you:~/darknet $ ./build.sh
```
   

## Compile and Build Darknet : Windows (using ```CMake```) - Not Recommended
***아래는 원본 darknet의 README.md를 그대로 복사해 온 내용입니다.***   
   
Requires: 
* MSVS: https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community
* Cmake GUI: `Windows win64-x64 Installer`https://cmake.org/download/
* Download Darknet zip-archive with the latest commit and uncompress it: [master.zip](https://github.com/AlexeyAB/darknet/archive/master.zip)

In the Windows: 


* Start (button) -> All programms -> Cmake -> Cmake (gui) -> 

* [look at image](https://habrastorage.org/webt/pz/s1/uu/pzs1uu4heb7vflfcjqn-lxy-aqu.jpeg) In Cmake: Enter input path to the darknet Source, and output path to the Binaries -> Configure (button) -> Optional platform for generator: `x64`  -> Finish -> Generate -> Open Project -> 

* in MS Visual Studio: Select: x64 and Release -> Build -> Build solution

* find the executable file `darknet.exe` in the output path to the binaries you specified

![x64 and Release](https://habrastorage.org/webt/ay/ty/f-/aytyf-8bufe7q-16yoecommlwys.jpeg)

## Compile and Build Darknet : Windows (using ```vcpkg```) - Recommended

***아래는 원본 darknet의 README.md를 그대로 복사해 온 내용입니다.***   
   
1. Install Visual Studio 2017 or 2019. In case you need to download it, please go here: [Visual Studio Community](http://visualstudio.com)

2. Install CUDA (at least v10.0) enabling VS Integration during installation.

3. Open Powershell (Start -> All programs -> Windows Powershell) and type these commands:

```PowerShell
PS Code\>              git clone https://github.com/microsoft/vcpkg
PS Code\>              cd vcpkg
PS Code\vcpkg>         $env:VCPKG_ROOT=$PWD
PS Code\vcpkg>         .\bootstrap-vcpkg.bat
PS Code\vcpkg>         .\vcpkg install darknet[full]:x64-windows #replace with darknet[opencv-base,cuda,cudnn]:x64-windows for a quicker install of dependencies
PS Code\vcpkg>         cd ..
PS Code\>              cd darknet # darknet 디렉토리로 이동
PS Code\darknet>       powershell -ExecutionPolicy Bypass -File .\build.ps1
```

# Train and Test
### Run Train

위 모든 사항을 진행하였다면, 이제 학습을 진행할 차례입니다. 위에서 만들었던 ```data``` 디렉토리를 ```darknet``` 디렉토리 내로 이동한 뒤, ```darknet``` 디렉토리 내에서 다음과 같이 학습용 darknet을 실행합니다.

```console
you@you:~/darknet $ nohup ./darknet detector train data/obj.data yolov4.cfg yolov4.conv.137 -dont_show -mjpeg_port 8090 -map &
```   
각 파라미터는 다음을 의미합니다.
```console
you@you:~/darknet $ nohup ./darknet detector train ".data 경로" ".cfg 경로" "pre-trained 모델 경로" -dont_show -mjpeg_port 8090 -map &
```   
+ 학습은 백그라운드에서 진행되므로 터미널을 종료해도 됩니다. 
+ ```darknet/nohup.out``` 파일에 학습 로그가 실시간으로 저장됩니다.
+ 실행 후 약 30초 정도 기다린 뒤 로그를 확인하여 Error 등이 없는지, 혹은 ```ps -ef``` 명령어를 통해 프로세스가 잘 작동되고 있는지 확인하여 학습이 잘 진행중임을 확인해야 합니다.
+ ```backup/``` 디렉토리에 1000번 단위로 학습된 모델이 저장됩니다.
+ 학습 그래프는 ```darknet/chart.png```에 실시간으로 저장됩니다. 이 학습 그래프는 웹브라우저에서 http://localhost:8090 에 접속하여 확인할 수 있습니다.
+ 학습 그래프에서 파랑색 선은 Loss Function(실제 값과 예측 값의 차이), 빨강색 선은 mAP(객체인식 정확도)를 나타냅니다. 통상적으로 학습이 진행될수록 Loss 값은 감소, mAP값은 증가되어야 합니다.
+ 학습 시의 mAP는 학습데이터를 기준으로 계산된 값입니다. 가령, 한 번의 학습에 64개의 이미지를 사용한다면 학습이 누적된 모델에 대해 ```darknet/data/train```에 존재하지만 이전 학습에 사용되지 않은 20개의 이미지를 이용하여 mAP를 계산합니다. (정확한 숫자는 아닙니다.)
+ YOLO 개발자는 경험상 하나의 객체당 2000번의 학습을 진행하는 것이 Overfitting을 방지하면서 최대한의 학습횟수를 보장한다고 합니다. 학습이 완료될 경우 ```darknet/backup```에 ```yolov4-1000.weights```, ```yolov4-2000.weights```, ```yolov4-best.weights```, ```yolov4-final.weights``` 과 같은 형식으로 모델이 생성되는데, 제 경험상 클래스 3개 이하로 적은 환경에서는 ```yolov4-best.weights```가 매우 적은 학습횟수에 잡히는 경우가 종종 있었습니다. 그러므로 ```darknet/backup``` 디렉토리에서 ```ls -la``` 혹은 다른 수단을 통해 각 모델이 생성된 시점을 확인한 뒤, ```yolov4-best.weights```와 ```yolov4-final.weights``` 이 생성된 시점을 비교하여 둘의 차이가 극명하다면 ```yolov4-final.weights```를 실제 모델로 채택하고, 그렇지 않다면 ```yolov4-best.weights```를 채택하는 것을 추천합니다.   
   
학습 그래프 이미지는 다음과 같습니다. (본 예제를 실제로 학습한 그래프가 아닙니다.)   
   
![chart_my-yolov4](https://user-images.githubusercontent.com/49421142/110059747-2c87b100-7da8-11eb-82dc-c7364f3e4d39.png)


### Run Test
```console
you@you:~/darknet $ ./darknet detector map data/obj.data yolov4.cfg backup/yolov4-final.weights -dont_show -ext_output < data/test.txt > result.txt
```
각 파라미터는 다음을 의미합니다.
```console
you@you:~/darknet $ ./darknet detector map ".data 경로" ".cfg 경로" "학습된 모델 경로" -dont_show -ext_output < "test이미지 경로가 저장된 파일" > "테스트 결과를 출력할 파일"
```

테스트가 완료되면 ```result.txt``` 파일의 모습은 다음과 같습니다. (본 예제를 실제로 학습하고 테스트한 결과가 아닙니다.)
```
 CUDNN_HALF=1 
net.optimized_memory = 0 
mini_batch = 1, batch = 64, time_steps = 1, train = 0 
nms_kind: greedynms (1), beta = 0.600000 
nms_kind: greedynms (1), beta = 0.600000 
nms_kind: greedynms (1), beta = 0.600000 

 seen 64, trained: 384 K-images (6 Kilo-batches_64) 

 calculation mAP (mean average precision)...
 Detection layer: 139 - type = 28 
 Detection layer: 150 - type = 28 
 Detection layer: 161 - type = 28 

 detections_count = 1029, unique_truth_count = 578  
 rank = 0 of ranks = 1029 
 rank = 100 of ranks = 1029 
 rank = 200 of ranks = 1029 
 rank = 300 of ranks = 1029 
 rank = 400 of ranks = 1029 
 rank = 500 of ranks = 1029 
 rank = 600 of ranks = 1029 
 rank = 700 of ranks = 1029 
 rank = 800 of ranks = 1029 
 rank = 900 of ranks = 1029 
 rank = 1000 of ranks = 1029 
class_id = 0, name = excavator, ap = 98.15%   	 (TP = 224, FP = 14) 
class_id = 1, name = dump_truck, ap = 92.66%   	 (TP = 238, FP = 35) 
class_id = 2, name = concrete_mixer_truck, ap = 97.13%   	 (TP = 78, FP = 5) 

 for conf_thresh = 0.25, precision = 0.91, recall = 0.93, F1-score = 0.92 
 for conf_thresh = 0.25, TP = 540, FP = 54, FN = 38, average IoU = 78.74 % 

 IoU threshold = 50 %, used Area-Under-Curve for each unique Recall 
 mean average precision (mAP@0.50) = 0.959766, or 95.98 % 

Set -points flag:
 `-points 101` for MS COCO 
 `-points 11` for PascalVOC 2007 (uncomment `difficult` in voc.data) 
 `-points 0` (AUC) for ImageNet, PascalVOC 2010-2012, your custom dataset
```

여기에서 ```mean average precision (mAP@0.50) = 0.959766, or 95.98 % ``` 이 모델을 ```darknet/data/test``` 내 데이터들로 테스트하여 도출된 mAP를 의미합니다. 각 객체별 mAP값은 그 위의 ```class_id = 0```, ```class_id = 1``` 등에 나타납니다. 

+ 결과이미지를 얻고 싶다면 다음 darknet을 사용하는것을 추천합니다. ```Makefile```, ```yolov4.cfg```, ```data```, 모델파일을 새로 옮기고 재컴파일 하여 사용하면 됩니다.
https://github.com/vincentgong7/VG_AlexeyAB_darknet
```console
you@you:~/darknet $ ./darknet detector batch <.data> <.cfg> <.weights> -dont_show batch <input images dir/> <output images dir/> -ext_output > result.txt
```
