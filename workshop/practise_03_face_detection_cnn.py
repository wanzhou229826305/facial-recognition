import torch,torchvision
import torch.utils.data
import torch.nn as nn
import numpy as np
import tqdm,logging,os
import sys

sys.path.insert(0, './ai_workshop')
import face_label,face_detection_cnn_utils

#请取消注释
# def createModel():
#     return FaceDetectionModel()

#请取消注释
# class FaceDetectionModel(nn.Module):
#     def __init__(self) -> None:
#         super(FaceDetectionModel, self).__init__()

#添加encode
##练习 1，请输入
        # self.encode1 = 
        
        # self.encode2 = 

        # self.encode3 = 
        
        # self.encode4 = 
        
        # self.encode5 = 
        
#添加decode
##练习 2，请输入
        # self.decode1 = 
        
        # self.decode2 = 
        
        # self.decode3 = 
        
        # self.decode4 = 
        
        # self.decode5 = 
        
        # self.classifier = 

#添加forward
##练习 3，请输入
    # def forward(self, x):

    #     return x

def train(model, dataPath, num_epoch):

    # 定义数据集
    # 取消注释
    # faceImageSet = face_detection_cnn_utils.FaceLabelDatasetFromFolder(dataPath)
    
    #定义 dataloader
    ##练习 4，请输入
    # loader = 

    #定义优化器
    ##练习 5，请输入
    # optimizer =      

    # 定义损失函数
    ##练习 6，请输入
    # loss_function = 

    # 取消注释
    # model.train()

    # 训练循环
    ##练习 7，请输入

    pass


#请补充下面的函数
##练习 8，请输入
# def segment(model,targetImageFile):

#     return result

#请取消注释
if __name__ == '__main__':

    # cnn = createModel()

    # train(cnn.model,'data/train-segment',num_epoch=10)

    # segment(cnn.model,'data/train-segment/yonwei.jpg')
    pass