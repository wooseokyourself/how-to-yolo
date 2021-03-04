# Description

Darknet을 통해 YOLO 모델을 위한 커스텀 데이터를 만들고, 학습하고, 테스트하는 방법을 설명합니다.   
딥러닝과 Object Detection에 대한 원리 혹은 개념 설명은 포함하지 않습니다.   

# Prepare Custom Dataset

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

### File: data/test.txt
```data/test``` 디렉토리 내에 있는 이미지 파일의 목록이 쓰여져있습니다. 위 디렉토리를 예로 들면 다음과 같습니다.
```
data/train/000001.jpg
data/train/000002.jpg
```

### File: data/obj.names
학습을 통해 검출하고자 하는 객체의 클래스명이 작성된 파일입니다. 가령, 내 데이터셋이 사람과 고양이 두 객체를 각각 첫 번째(id=0)와 두 번째(id=1) 클래스로 라벨링하였다면 본 파일의 내용은 다음과 같습니다.
```
person
cat
```
혹은 다음과 같이 될 수도 있습니다.
```
human
the-cutest-creature
```
이처럼 클래스명은 모델에게 "이 객체는 a야. 이 객체는 b야." 라고 알려주는 용도입니다. 그러므로 클래스명은 단지 내가 알아볼 수 있기만 하면 됩니다. 하지만 지정된 클래스의 id 순서대로 작성되어야 합니다. 본인이 사람을 고양이로, 고양이를 사람으로 부르겠다면 말리지는 않겠습니다. 만약 내 데이터셋에는 총 2개의 객체가 라벨링되어있는데 본 파일에는 1개 혹은 3개 이상의 클래스명이 작성되어있다면 학습이 진행되지 않습니다.

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