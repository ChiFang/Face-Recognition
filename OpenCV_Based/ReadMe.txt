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

USAGE: CreatSample.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [--outdir <OutDir>] [--sample-num <SampleNum>]

cascade: 人臉偵測模型 不輸入會用預設路徑中的檔案
nested-cascade: 人眼偵測模型 不輸入會用預設路徑中的檔案
outdir: 人臉圖片擷取後存放路徑 不輸入會在當前目錄名為sample的資料夾底下存放
sample-num: 人臉圖片擷取數量 不輸入會預設擷取10張

2. 將產生的圖片所屬資料夾(目前內定是sample)移到你要的資料夾下並改成你要的名子

ex:
FaceDataFolder>>>person 0 (10 smple image)
                 person 1 (10 smple image)
                 .
                 .
                 .
                 person n (10 smple image)
                 
3. 使用 Train_Recognizer_LBP.py 下去訓練

USAGE: Train_Recognizer_LBP.py [--path <Path>]  [--model-name <ModelName>] [--label-name <LabelName>]

path: FaceDataFolder 路徑 不輸入則預設為 FaceRecData_LBPH_Test
model-name: 輸出模型檔名 不輸入則預設為 Model_default.yml
label-name: label 清單名稱 不輸入則預設為 Label_default.txt

4. 使用 facedRec_LBP.py 來做人臉偵測並且將人臉做辨識，分數越低表示越像

USAGE: facedRec_LBP.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>]  [--model-name <ModelName>] [--label-name <LabelName>]

cascade: 人臉偵測模型 不輸入會用預設路徑中的檔案
nested-cascade: 人眼偵測模型 不輸入會用預設路徑中的檔案
model-name: 模型檔名 不輸入則預設為 Model_default.yml
label-name: label 清單檔 不輸入則預設為 Label_default.txt