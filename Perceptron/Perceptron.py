# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 14:42:08 2018

@author: Administrator
"""

'''
感知机
'''
class Perceptron:
    '''
    初始化函数
    参数为学习率learning_rate
    权重w0,w1
    偏置b
    '''
    def __init__(self,learning_rate,w0,w1,b):
        self.learning_rate=learning_rate
        self.w0=w0
        self.w1=w1
        self.b=b
    
    '''
    模型
    此处的x[2]为label y
    '''
    def model(self,x):
        result=x[2]*(self.w0*x[0]+self.w1*x[1]+self.b)
        return result

    '''
    策略
    '''
    def iserror(self,x):
        result=self.model(x)
        if result<0:
            return True
        else :
            return False

    '''
    调整策略:Wi=Wi+n*yi*xi
    '''
    def gradientdescent(self,x):
        self.w0=self.w0+self.learning_rate*self.x[2]*self.x[0]
        self.w1=self.w1+self.learning_rate*self.x[2]+self.x[1]
        self.b=self.b+self.learning_rate*self.x[2]

    '''
    训练模型
    '''
    def trainModel(self,data):
        #训练次数
        times=0
        #是否完成标志
        done=False
        #循环直到所有分类都正确
        while not done:
            for i in range(0,6):
                #如果分类错误
                if self.iserror(data[i]):
                    #进行梯度调整
                    self.gradientdescent(data[i])
                    times+=1
                    done=False
                    break
                else:
                    #分类正确，直接返回完成为True
                    done =True
        print('total training times %g'%(times))
        print('After training,the parameters is w0:%d,w1:%d,b:%d'%(self.w0,self.w1,self.b))
        
    '''
    测试模型
    '''
    def testModel(self,x):
        result=self.w0*x[0]+self.w1*x[1]+self.b
        if result>=0:
            return 1
        else:
            return -1
            

def main():
    #w1 w0 b均赋值为0，学习率设为0.5
    p=Perceptron(0.5,0,0,0)
    trainData=[[3,3,1],[4,3,1],[1,1,-1],[2,2,-1],[5,4,1],[1,3,-1]]
    testData=[[4,4,-1],[1,2,-1],[1,4,-1],[3,2,-1],[5,5,1],[5,1,1],[5,2,1]]
    #模型的训练
    p.trainModel(trainData)
    for i in testData:
        print('%d %d %d'%(i[0],i[1],p.testModel(i)))
        
    return 0


if __name__=='__main__':
    main()























