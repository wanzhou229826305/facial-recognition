import torch,torchvision
import torch.nn as nn

import numpy as np
import tqdm
import logging

from workshop import face_recognition_cnn_utils
from workshop import face_label

currentModelName = 'face-recognition-v10'


#请取消注释
#学员操作 1
# def createModel(modelName=currentModelName):
#     class ModelFunctions():
#         def __init__(self,train) -> None:
#             self.train = train

#     if modelName == currentModelName:
#         model = FaceRecognitionModel10()
#         funcs = ModelFunctions(train)
#         funcs.embedding = embedding
#         funcs.test = test
#         funcs.similarity = similarity
#         return model,funcs
#     return None

#请复制模型定义 (包含 __init__）
#学员操作 2
# class FaceRecognitionModel10(nn.Module):
#     def __init__(self) -> None:
        
#请复制forward
#学员操作 3
    # def forward(self, x):


#请取消注释
#学员操作 4
# global_transform = torchvision.transforms.Compose([
#                     torchvision.transforms.RandomEqualize(),
#                     torchvision.transforms.ToTensor()])

# g_image_size = (160,160)
# batch_size = 16

#请取消注释
#学员操作 5
# def train(model, dataPath, num_epoch):

#     imageset = face_recognition_cnn_utils.FacePairDatasetFromFolder(dataPath)
    
#请复制 collation function
#学员操作 6
    # def colalate_function(facePairs):

#请复制 optimizer loss_function
#学员操作 7


#请取消注释
#学员操作 8
    # for epoch in range(num_epoch):
    #     progress = tqdm.tqdm(loader, desc='{}/{},loss={}'.format(epoch,
    #                          num_epoch, '%0.15f' % (currentLoss)))

    #     trainNumber = 0
    #     trainLoss = 0

#请复制 train loop, 包含anchors_embedding,positives_embedding
#学员操作 9

#请取消注释
#学员操作 10
# def test(model,srcPath, embeddings):

#     names = [  name for name,_ in embeddings]
#     names = np.array(names)
#     embeddings = [  emdbedding for _,emdbedding in embeddings]

#     testFaces = face_recognition_cnn_utils.FaceDatasetFromFolder(srcPath)

#请复制 collate_fn 以及 DataLoader
#学员操作 11
    # model.eval()
    # results = []

#请复制 collate_fn 以及 DataLoader
#学员操作 12
    # with torch.no_grad():
    #     embeddings = torch.tensor(embeddings)
        
#请复制 计算 embedding 部分
#学员操作 13

#请取消注释
#学员操作 14
# def embedding(model, srcPath):

#     enrolledFaces = face_recognition_cnn_utils.FaceDatasetFromFolder_Flat(srcPath)

#请复制 collate_fn 以及 DataLoader
#学员操作 15

#请复制 计算 embedding 部分
#学员操作 16


#请取消注释
#学员操作 17
# def similarity(model, embeddings,target,top=10,threshold = 0.9):

#     names = [  name for name,_ in embeddings]
#     names = np.array(names)

#     embeddings = [  emdbedding for _,emdbedding in embeddings]

#     result = []

#     model.eval()

#请复制 计算 embedding 以及相似度计算 部分
#学员操作 18

#请复制 _selectName 部分
#学员操作 19

if __name__ == '__main__':

    # model,funcs = createModel()
    # train(model,'data/train-segment',num_epoch=10)

    # torch.save(model.cpu().state_dict(), 'ai_face_detection-cpu.pt')

    # segment(model,'data/train-segment/yonwei.jpg')
    pass