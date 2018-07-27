# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 09:41:28 2018

@author: Administrator
"""

'''
（1）将每个图片（即txt文本，以下提到图片都指txt文本）转化为一个向量，即32*32的数组转化为1*1024的数组，这个1*1024的数组用机器学习的术语来说就是特征向量。

（2）训练样本中有10*10个图片，可以合并成一个100*1024的矩阵，每一行对应一个图片。（这是为了方便计算，很多机器学习算法在计算的时候采用矩阵运算，可以简化代码，有时还可以减少计算复杂度）。

（3）测试样本中有10*5个图片，我们要让程序自动判断每个图片所表示的数字。
同样的，对于测试图片，将其转化为1*1024的向量，然后计算它与训练样本中各个图片的“距离”（这里两个向量的距离采用欧式距离），然后对距离排序，选出较小的前k个，因为这k个样本来自训练集，是已知其代表的数字的，所以被测试图片所代表的数字就可以确定为这k个中出现次数最多的那个数字。
'''
import numpy as np
import os
import operator
'''
将图像转化为1*1024向量
MNIST图像为32*32----->1*1024
'''
def img2Vec(filename):
    #创建一个需要返回的1*1024向量
    returnVect=np.zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        #读取文件每一行
        lineStr=fr.readline()
        for j in range(32):
            #将32*32图像转为1*1024向量
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

'''
将训练集图片合并成100*1024的大矩阵，同时逐一对测试集中的样本分类
'''
def handWritingClassTest():
    #加载训练集到大矩阵trainingMat
    hwLabels=[]
    #os模块中的listdir('str')可以读取目录str下的所有文件名，返回一个字符串列表
    trainingFilelist=os.listdir('digits/trainingDigits')
    m=len(trainingFilelist)
    trainingMat=np.zeros((m,1024))
    #修改训练样本名
    for i in range(m):
        #获取训练数据文件名称
        fileNameStr=trainingFilelist[i]
        #取出0_0.txt前半部分0_0，依次类推
        fileStr=fileNameStr.split('.')[0]
        #取出各图片类别0_0--->0
        classNumStr=int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        #读取所有训练文件并将他们转化为1934*1024大矩阵
        trainingMat[i,:]=img2Vec('digits/trainingDigits/%s'%fileNameStr)
        
    #逐一读取测试图片，并将其进行分类
    testFileList=os.listdir('digits/testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        #获取测试数据文件名称
        fileNameStr=testFileList[i]
        #取出0_0.txt前半部分0_0，依次类推
        fileStr=fileNameStr.split('.')[0]
        #取出各图片类别0_0--->0
        classNumStr=int(fileStr.split('_')[0])
        #将测试图片转为1*1024向量
        vectorUnderTest=img2Vec('digits/testDigits/%s'%fileNameStr)
        #分类结果
        #K=3 vectorUnderTest测试向量集  trainingMat训练向量集 hwLabels训练+测试标签
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        #如果测试结果和真实结果不相同，则错误数量加一
        if (classifierResult!=classNumStr):
            errorCount+=1
    #测试总错误量
    print ("\nthe total number of errors is: %d" % errorCount)
    #错误率
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))

'''
重头戏：
KNN算法的实现
这里面的函数classify()为分类主体函数，计算欧式距离，并最终返回测试图片类别
inX:测试向量
dataSet:训练样本集1934*1024
labels:数字对应的标签
k:选取K值
'''
def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]#获取数据第二维度长度（多少行）
    #numpy.tile()函数在列方向上重复(inX)1次,行上重复dataSetSize次
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    #欧式距离的计算
    sqDifMat=diffMat**2
    sqDistances=sqDifMat.sum(axis=1)
    distances=sqDistances**0.5
    #选择距离最小的k个点，argsort()函数：由小到大排序，然后提取对应元素的索引，输出到y
    sortedDistIndicies=distances.argsort()
    #字典定义为{'A':2,'B':1}等此类型
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    #使用itemgetter方法，按照第二个元素的次序对元组进行排序，此处为逆序排序，按照从小到大的顺序，最后返回频率最高的元素标签
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
    
    

            


handWritingClassTest()
















