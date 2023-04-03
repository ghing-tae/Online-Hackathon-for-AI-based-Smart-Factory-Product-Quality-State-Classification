## Online-Hackathon-for-AI-based-Smart-Factory-Product-Quality-State-Classification
> 
* 팀명 : 가슴이 두근두근
* 팀원 : 민초맛대흉근, 월요일은대흉근, itsming, 청국장, 킁킁이
* 코드 : EDA_catboost_stratify_kfold.ipynb
* 데이터 : train.csv, test.csv, sample_submission.csv
---
## EDA
![EDA_012](https://user-images.githubusercontent.com/52441719/229600711-7da3569b-45d3-4172-89f3-56eb332a748d.png)
* 'Y_Class'와 'Y_Quality' 비교 -> 정상범위 최소값 : 0.525066667, 정상범위 최대값 : 0.534842857
* 정상범위 안에 있어야 불량 -> 정상으로 판단했을 때 발생되는 기업손실을 막을 수 있다.
  > 가정 : 기업손실을 줄일수록 더 좋은 점수를 받을 것
  
1. 'Y_Class' 데이터 불균형 해결 : Borderline, Stratified k-fold
2. 'LINE' 별로 heatmap을 그렸을 때 비슷한 'LINE' 존재 : 비슷한 'LINE'끼리 묶어 결측치 보간
---
## Data 전처리
1. TRAIN 전체가 NAN 삭제 : 74개
2. TRAIN의 고유값이 1개 삭제 : 462개
3. 상관관계 그래프 비슷한 LINE별로 평균값 보간
  * T100306, T100304
  * T010306, T010305
  * T050304, T050307
4. 남은 결측치 0으로 보간
5. 'Y_Quality'와 비교했을 때 25% 보다 낮은 상관관계 지우기 : 698개
---
## Scailing
* RobustScaler : 이상치 제거 기능

## Regressor
* CatBoostRegressor voting 3-Fold

## Classification
* BorderlineSMOTE : 데이터 불균형 해소
* StratifiedKFold : 선택된 최종 모델은 가장 좋은 검증 점수를 가진 모델로 설정
