目標：將指甲的部分從一張圖片取出，以便用CNN做訓練，只取一根手指的原因是因為貧血應該每根手指都有症狀，一根就夠了。

### 檔案介紹
1. model.py: yolo v1架構
2. loss.py: IOU + mean-squared error
3. evaluate.py: 計算MSE
4. helper.py: 計算training, valid score(MSE), 將yolo v1的output decode
5. train.ipynb: 建構training pipeline
6. valid.ipynb: 監控validation set performance肉眼上的變化。
7. dataset.py: augmentation(因為資料有點少，所以加了蠻重的augmentation: random_noise, random_flip_lr, random_flip_ud, random_scale, random_blur, random_brightness, random_hue, random_saturation, random_shift, random_crop)，但發現有些會造成更難訓練。


### Data introduction
training data: 361 images, validation data: 91 images

input data:

<img width="370" alt="image" src="https://user-images.githubusercontent.com/101687024/210759097-c00e930d-80d9-4f27-8fa4-ed1fc86623e6.png">

output data:

<img width="370" alt="image" src="https://user-images.githubusercontent.com/101687024/210758468-9ae77290-4c67-4e5c-847c-9c1b1b3614fa.png">

### Performance
train MSE :  0.16528548005519034

valid MSE :  0.30960401192401715
