#### 目標：用指甲照片預測病人是否有貧血

#### 檔案介紹：

1.train.ipynb: 利用efficientnet b6區分經過yolo v1切割後的指甲照片是否有貧血的特徵

2.valid.ipynb: 用眼睛去看那些分錯的照片有沒有什麼共同的特徵，ex:yolo 切不好，膚色差異等等）

3.train_center.ipynb: 只拿照片的center做RGB顏色histagram的分析，利用BernoulliNB, xgboost, RandomForestClassifier, KNeighborsClassifier, svm, 以及NN做訓練。

#### 結果：

ACC:

    BernoulliNB:0.5074

    xgboost:0.64

    RandomForestClassifier:0.55

    KNeighborsClassifier:0.5074

    svm:0.5074

    NN:0.67

發現膚色可能會影響模型的判斷，所以試著將膚色也考慮進去了，多一個特徵，圖片center的平均。

最後NN的ACC可以達到0.71

### Future work

將所有model的結果做結合，希望可以讓結果更好。

### Progress

1/6

NN+HSV_mean: 0.70(5000epoch)

<img width="449" alt="image" src="https://user-images.githubusercontent.com/101687024/210930850-f91bce29-05d9-404e-b54f-96b2c913dd11.png">

NN+HSV_mean+RGB_mean: 0.73(5000 epoch)

<img width="449" alt="image" src="https://user-images.githubusercontent.com/101687024/210931949-ae80002b-7a9e-4f5f-9d50-4f3f05f93184.png">

NN+HSV_mean+RGB_mean: 0.746(5000-10000 epoch)

<img width="449" alt="image" src="https://user-images.githubusercontent.com/101687024/210933233-4a50e9cd-1c92-48c9-9abc-b3ac33a5ef4e.png">

