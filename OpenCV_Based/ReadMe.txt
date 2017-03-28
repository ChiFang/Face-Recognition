這專案操作環境如下:
OS: win10
語言: python 2.7.10
主要第三方函式庫: OpenCV 2.4.9

Note:
1. 理論上 Python 能跑的環境都能跑 (Windows Mac Linux)
2. python3 應該也可以(未測試)
3. openCV3 應該也可以(未測試)


動機:
主要是看到Kinect v2 SDK影提供 Face-Recognition API
其演算法主要是 "Eigenfaces algorithm"，剛好opencv有提供相關演算法


OpenCV提供3種Face-Recognition:

1. Eigenfaces:
有初步測試，但還未調適好，因為模型訓練跟使用不易 (有resize跟對位問題)

2. Fisherfaces
尚未測試

3. Local Binary Patterns Histograms:
這次完成的是這種，流程較簡單

流程如下:
1. 使用 CreatSample.py 抓取要辨識的人建檔，一次一個，每人抓10張

2. 將產生的圖片所屬資料夾(目前內定是sample)移到你要的資料夾下並改成你要的名子

ex:
FaceDataFolder>>>person 0 (10 smple image)
                 person 1 (10 smple image)
                 .
                 .
                 .
                 person n (10 smple image)
                 
3. 使用 Train_Recognizer_LBP.py 指定 FaceDataFolder(ex: FaceRecData_LBPH_Test) 和輸出檔案名稱 (ex: model_LBPH_test.yml)下去訓練


4. 使用 facedRec_LBP.py 來做人臉偵測並且將人臉做辨識，分數越低表示越像
這裡必須指定 label檔和模型檔 (ex: Label_test.txt and model_LBPH_test.yml)