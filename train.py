from PIL import Image
import os
from feature import NPDFeature
from ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import numpy as np

def getAllImage(path):
    '''
    函数作用：读取所有图片的路径
    :param path:图片所在文件夹路径
    :return:列表，元素是图片的路径
    '''
    ImageList = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
    return ImageList

def convertImage(imageList, big):
    '''
    函数作用：将图片转化灰度图
    :param imageList:列表，元素是图片路径
    :param big: int，图片大小
    :return:列表，元素是图片的（big*big）的灰度格式
    '''
    grayList = [Image.open(im).convert("L") for im in imageList]
    grayList = [im.resize((big, big)) for im in grayList]
    grayList = [np.array(im) for im in grayList]
    return grayList

def getNPDFeature(imageList):
    '''
    函数作用：提取灰度格式图片的NPD特征
    :param imageList:列表，元素是灰度格式图片
    :return:列表，元素是每个图片的NPD特征
    '''
    #temp = NPDFeature(imageList)
    featureList = [NPDFeature(im).extract() for im in imageList]
    return featureList

def organizeData(inputFileName, outputFileName):
    '''
    函数作用：读取图片，转化成灰度格式图片，抽取NPD特征，保存到本地
    :param outputFileName:输出文件名
    :param inputFileName:输入文件名
    :return:
    '''
    ImageList = getAllImage(inputFileName)
    grayList = convertImage(ImageList, 24)
    featureList = getNPDFeature(grayList)
    AdaBoostClassifier.save(featureList, outputFileName)

def dataProcess():
    '''
	函数作用：组织数据 
    :return: None
    '''
    faceImageFileName = './datasets/original/face'
    nonfaceImageFileName = './datasets/original/nonface'
    faceOutputFileName = './dataFeature/face'
    nonfaceOutputFileName = './dataFeature/nonface'
    organizeData(faceImageFileName, faceOutputFileName)
    organizeData(nonfaceImageFileName, nonfaceOutputFileName)

def calculateAcc(predictY, y):
    """
	函数作用：计算准确率
    :param predictY:列表，元素是预测值
    :param y:列表，元素是标签值
    :return:准确率
    """
    errorNum = 0
    for i in range(len(y)):
        if y[i] != predictY[i]:
            errorNum += 1
    rate = 1 - errorNum / len(y)
    return rate

def divideData(faceFileName,nonfaceFileName):
	'''
	函数作用：切分数据
	:param faceFileName:人脸特征文件名
	:param nonfaceFileName:非人脸特征文件名
	:return: 训练数据，验证数据，训练标签，验证标签
	'''
    faceImage =AdaBoostClassifier.load(faceFileName)
    nonfaceIamge = AdaBoostClassifier.load(nonfaceFileName)
    faceLen = len(faceImage)
    nonfaceLen = len(nonfaceIamge)
    faceLabel = [1] * faceLen
    nonfaceLabel = [-1] * nonfaceLen
    trainImage = []
    trainImage.extend(nonfaceIamge)
    trainImage.extend(faceImage)
    trainLabel = []
    trainLabel.extend(nonfaceLabel)
    trainLabel.extend(faceLabel)
    X_train, X_test, Y_train, Y_test = train_test_split(trainImage, trainLabel, test_size = 0.33)
    return np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)


if __name__ == "__main__":
    # write your code here
    #dataProcess()
    trainData, testData, trainLabel, testLabel = divideData('./dataFeature/face', './dataFeature/nonface')
    clf = DecisionTreeClassifier(max_depth=2)
    row, col = trainData.shape
    weightArray = [(1 / row)] * row
    clf.fit(trainData, trainLabel, weightArray)
    predictY = clf.predict(testData)
    rate = calculateAcc(predictY, testLabel)
    print(rate)
    target_name = ['class-1', 'class1']
    with open("./report.txt", 'w') as fp:
        fp.write(classification_report(testLabel, predictY, target_names=target_name))
    print(classification_report(testLabel, predictY, target_names=target_name))
    clf2 = AdaBoostClassifier(DecisionTreeClassifier, 20)
    clf2.fit(trainData, trainLabel)
    predictY2 = clf2.predict(testData)
    rate = calculateAcc(predictY2, testLabel)
    print(rate)
    target_name = ['class-1', 'class1']
    with open("./report1.txt", 'w') as fp:
        fp.write(classification_report(testLabel, predictY2, target_names=target_name))
    print(classification_report(testLabel, predictY2, target_names=target_name))
