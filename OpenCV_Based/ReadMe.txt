�o�M�׾ާ@���Ҧp�U:
OS: win10
�y��: python 2.7.10
�D�n�ĤT��禡�w: OpenCV 2.4.9

Note:
1. �z�פW Python ��]�����ҳ���] (Windows Mac Linux)
2. python3 ���Ӥ]�i�H(������)
3. openCV3 ���Ӥ]�i�H(������)


�ʾ�:
�D�n�O�ݨ�Kinect v2 SDK�v���� Face-Recognition API
��t��k�D�n�O "Eigenfaces algorithm"�A��nopencv�����Ѭ����t��k


OpenCV����3��Face-Recognition:

1. Eigenfaces:
����B���աA���٥��վA�n�A�]���ҫ��V�m��ϥΤ��� (��resize������D)

2. Fisherfaces
�|������

3. Local Binary Patterns Histograms:
�o���������O�o�ءA�y�{��²��

�y�{�p�U:
1. �ϥ� CreatSample.py ����n���Ѫ��H���ɡA�@���@�ӡA�C�H��10�i

USAGE: CreatSample.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [--outdir <OutDir>] [--sample-num <SampleNum>]

cascade: �H�y�����ҫ� ����J�|�ιw�]���|�����ɮ�
nested-cascade: �H�������ҫ� ����J�|�ιw�]���|�����ɮ�
outdir: �H�y�Ϥ��^����s����| ����J�|�b��e�ؿ��W��sample����Ƨ����U�s��
sample-num: �H�y�Ϥ��^���ƶq ����J�|�w�]�^��10�i

2. �N���ͪ��Ϥ����ݸ�Ƨ�(�ثe���w�Osample)����A�n����Ƨ��U�ç令�A�n���W�l

ex:
FaceDataFolder>>>person 0 (10 smple image)
                 person 1 (10 smple image)
                 .
                 .
                 .
                 person n (10 smple image)
                 
3. �ϥ� Train_Recognizer_LBP.py �U�h�V�m

USAGE: Train_Recognizer_LBP.py [--path <Path>]  [--model-name <ModelName>] [--label-name <LabelName>]

path: FaceDataFolder ���| ����J�h�w�]�� FaceRecData_LBPH_Test
model-name: ��X�ҫ��ɦW ����J�h�w�]�� Model_default.yml
label-name: label �M��W�� ����J�h�w�]�� Label_default.txt

4. �ϥ� facedRec_LBP.py �Ӱ��H�y�����åB�N�H�y�����ѡA���ƶV�C��ܶV��

USAGE: facedRec_LBP.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>]  [--model-name <ModelName>] [--label-name <LabelName>]

cascade: �H�y�����ҫ� ����J�|�ιw�]���|�����ɮ�
nested-cascade: �H�������ҫ� ����J�|�ιw�]���|�����ɮ�
model-name: �ҫ��ɦW ����J�h�w�]�� Model_default.yml
label-name: label �M���� ����J�h�w�]�� Label_default.txt