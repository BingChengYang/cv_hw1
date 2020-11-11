# cv_hw1

* 1.Training
* 2.Testing

## Training model
使用以下指令訓練模型:
```
python train.py --lr=0.00001 --epoches=10 --mini_batch_size=32 --load_model=False --model="model.pkl --img_size=300"
```
lr代表learning rate的大小，default = 0.00001</br>
epoches代表總共訓練幾個epoch，default = 10</br>
mini_batch_size代表會使用mini_batch的大小，default = 32</br>
load_model代表是否要使用訓練到一半的model，False代表重新訓練一個resnet50的模型，
True則可以選擇要接下去訓練的model，default = False</br>
model代表使用load_model=True時，要接下去的model名稱，default = "model.pkl"</br>
img_size代表要將訓練圖片重新resize的大小，default = 300</br>
## Testing model
使用下列指令來產生預測結果:
```
python test.py --test_model="model90.2pkl"
```
test_model代表所要選擇測試的model，default = "model90.2pkl"</br>
結束後會產生answer.csv，為給kaggle的檔案。
