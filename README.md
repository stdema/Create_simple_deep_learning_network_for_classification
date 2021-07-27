# Create_simple_deep_learning_network_for_classification
  - 딥러닝 분류용으로 간단한 컨벌루션 신경망을 만들고 훈련시켰다.
  
  컨벌루션 신경망은 딥러닝 분야의 필수 툴로서, 특히 영상 인식에 적합하다. 
  영상 데이터를 불러와서 살펴본다, 신경망 아키텍처를 정의한다, 훈련 옵션을 지정한다, 신경망을 훈련시킨다, 새로운 데이터의 레이블을 예측하고 분류 정확도를 계산한다.
  
### 영상 데이터를 불러와서 살펴보기
샘플 숫자 데이터를 영상 데이터저장소로서 불러온다.

```c
digitDatasetPath = fullfile(matlabroot, 'toolbox', 'nnet', 'nndemos', ...
    'nndatasets', 'DigitDataset'); %매트랩이 설치되어 있는 파일로부터 digitdataset 폴더의 경로를 지정
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
```
데이터 저장소의 영상 몇 개를 표시한다.
```c
figure;
perm = randperm(10000, 20);
for i = 1:20
    subplot(4, 5, i)
    imshow(imds.Files{perm(i)}); %만 개의 데이터 중 무작위로 20개 뽑아서 보임. 셀 형 데이터 불러올 때는 중괄호
end
```
![untitled](https://user-images.githubusercontent.com/86040099/127124367-e7fddb6c-784e-4196-9cd9-97fd398cca1e.jpg)

각 범주에 속한 이미지의 개수를 계산한다. 데이터 ㅈ장소에는 0부터 9까지의 각 숫자에 대해 1000개씩 총 10,000개의 이미지가 포함된다.
```c
labelCount = countEachLabel(imds)
```
![image](https://user-images.githubusercontent.com/86040099/127124730-328da53c-a551-4f1d-80c0-5c2111903b38.png)

digitData에 있는 첫 번째 이미지의 크기를 확인한다. (28*28*1 픽셀)
```c
img = readimage(imds, 1);
size(img)
```
![image](https://user-images.githubusercontent.com/86040099/127124999-beba5b3c-01e7-432f-9a0f-7eb9331f8732.png)

### 훈련 세트와 검증 세트 지정하기
한 라벨 당 훈련 750장, 검증 250장으로 분할한다.
```c
numTrainFiles = 750;
[imdsTrain, imdsValidation] = splitEachLabel(imds, numTrainFiles, 'randomize');
```

### 신경망 아키텍처 정의하기
```c
layers = [imageInputLayer([28 28 1])
    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
```
*영상 입력 계층* : (imageInputLayer) 영상 크기 지정

*컨벌루션 계층* : (convolution2dLayer) 특정 특징을 활성화하는 필터를 사용하여 이미지를 통과시킴 

*배치 정규화 계층* : (batchNormalizationLayr) 신경망 전체에 전파되는 활성화 값과 기울기 값을 정규화하여 신경망 훈련을 보다 쉬운 최적화 문제로 만듦

*ReLU 계층* : (reluLayer) 비선형 활성화 함수

*최댓값 풀링 계층* : (maxPooling2dLayer) 특징 맵의 공간 크기를 줄이고 중복된 공간 제거. 풀링 입력값이 나타내는 영역의 최댓값을 반환

*완전 연결 계층* : (fullyConnectedLayer) 뉴런들을 직전 계층의 모든 뉴런에 연결

*소프트맥스 계층* : (softmaxLayer) 완전 연결 계층의 출력값을 정규화함

*분류 계층* : (classificationLayer) 소프트맥스 활성화 함수가 각 입력값에 대해 반환한 확률을 사용하여 상호 배타적인 클래스 중 하나에 입력값을 할당아고 손실을 계산함

### 훈련 옵션 지정하기
SGDM : 모멘텀을 사용한 확률적 경사하강법
```c
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
```

### 훈련 데이터를 사용하여 신경망 훈련시키기
```c
net = trainNetwork(imdsTrain,layers,options);
```
![image](https://user-images.githubusercontent.com/86040099/127133542-fbc027a5-aca9-4daf-ae3f-9f6896bbd290.png)


### 검증 영상을 분류하고 정확도 계산하기
```c
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
```
![image](https://user-images.githubusercontent.com/86040099/127133759-bfd9275c-f05b-4dcd-88e8-5434d731062a.png)

정확도 99%
