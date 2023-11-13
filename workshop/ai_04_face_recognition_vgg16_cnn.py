import torch,torchvision
import torch.nn as nn
import torchvision.models as models

import numpy as np
import tqdm
import logging

from workshop import face_recognition_cnn_utils
from workshop import face_label

currentModelName = 'face-recognition-vgg16'

def createModel(modelName=currentModelName):
    class ModelFunctions():
        def __init__(self,train) -> None:
            self.train = train

    if modelName == currentModelName:
        model = FaceRecognitionModelVGG16()
        funcs = ModelFunctions(train)
        funcs.embedding = embedding
        funcs.test = test
        funcs.similarity = similarity
        return model.vgg16,funcs
    return None

class FaceRecognitionModelVGG16():
    def __init__(self) -> None:

        # resnet18 = models.resnet18()
        # alexnet = models.alexnet()
        # squeezenet = models.squeezenet1_0()
        # densenet = models.densenet_161()

        # Load the pre-trained VGG16 model
        self.vgg16 = models.vgg16(pretrained=True)
            # Freeze the pre-trained layers
        for param in self.vgg16.parameters():
            param.requires_grad = False

        self.vgg16.classifier = nn.Sequential(nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(4096, 2622))
            

global_transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    
g_image_size = (224,224)
batch_size = 16

def train(model, dataPath, num_epoch):

    vgg16 = model

    imageset = face_recognition_cnn_utils.FacePairDatasetFromFolder(dataPath)
    
    def colalate_function(facePairs):
        def faceDataToTensor(face):
            if face.imageData == None:
                face.imageData = face.getImageData(g_image_size)
                face.imageData = global_transform(face.imageData)
            return face.imageData
        
        anchors =[faceDataToTensor(anchor) for anchor,positive,similarity in facePairs]
        positives =[faceDataToTensor(positive) for anchor,positive,similarity in facePairs]
        similarities =[torch.tensor(similarity) for anchor,positive,similarity in facePairs]

        return torch.stack(anchors),torch.stack(positives),torch.stack(similarities)

    loader = torch.utils.data.DataLoader(imageset,batch_size=batch_size,shuffle=1,collate_fn=colalate_function)

    # optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    optimizer = torch.optim.SGD(vgg16.classifier.parameters(),lr=0.001,momentum=0.9)
    loss_function = nn.CosineEmbeddingLoss()

    # vgg16.train()

    currentLoss = 0.0

    for epoch in range(num_epoch):
        progress = tqdm.tqdm(loader, desc='{}/{},loss={}'.format(epoch,
                             num_epoch, '%0.15f' % (currentLoss)))

        trainNumber = 0
        trainLoss = 0
        for anchors,positives,similarities in progress:
            anchors_embedding = vgg16(anchors)
            positives_embedding = vgg16(positives)
 
            loss = loss_function(anchors_embedding,positives_embedding,similarities)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainLoss += loss.item() * len(anchors)
            trainNumber += len(anchors)

            progress.set_description('{}/{},loss={},batchLoss={}'.format(epoch,
                             num_epoch, '%0.15f' % (currentLoss),'%0.15f' % (loss.item())))


        currentLoss = trainLoss / trainNumber
        logging.debug('train==> %d/%d loss=%0.15f'%(epoch,num_epoch,currentLoss))

def test(model,srcPath, embeddings):

    names = [  name for name,_ in embeddings]
    names = np.array(names)
    embeddings = [  emdbedding for _,emdbedding in embeddings]

    testFaces = face_recognition_cnn_utils.FaceDatasetFromFolder(srcPath)

    def colalate_function(faces):
        def faceDataToTensor(face):
            if face.imageData == None:
                face.imageData = face.getImageData(g_image_size)
                face.imageData = global_transform(face.imageData)
            return face.imageData
        
        faces =[faceDataToTensor(face) for face in faces]
        return torch.stack(faces)
    loader = torch.utils.data.DataLoader(testFaces,batch_size=1,shuffle=False,collate_fn=colalate_function)

    model.eval()
    results = []

    with torch.no_grad():
        embeddings = torch.tensor(embeddings)
        
        for idx, testFace in enumerate(loader):
            face = testFaces.faceData[idx]
            testFace_embedding = model(testFace)

            result = _selectName(names,embeddings, testFace_embedding.squeeze(),1)

            results.append((face.imagePath,face.folderName,result[0]))

    return results

def embedding(model, srcPath):

    enrolledFaces = face_recognition_cnn_utils.FaceDatasetFromFolder_Flat(srcPath)

    def colalate_function(faces):
        def faceDataToTensor(face):
            if face.imageData == None:
                face.imageData = face.getImageData(g_image_size)
                face.imageData = global_transform(face.imageData)
            return face.imageData
        
        faces =[faceDataToTensor(face) for face in faces]
        return torch.stack(faces)

    loader = torch.utils.data.DataLoader(enrolledFaces,batch_size=1,shuffle=False,collate_fn=colalate_function)

    allResults = []
    model.eval()

    with torch.no_grad():
        progress = tqdm.tqdm(zip(enrolledFaces.faceData,loader),total=len(enrolledFaces))

        for face,imageData in progress:
            output = model(imageData)
            allResults.append((face.name,output.squeeze()))

    return allResults

def similarity(model, embeddings,target,top=10,threshold = 0.9):

    names = [  name for name,_ in embeddings]
    names = np.array(names)

    embeddings = [  emdbedding for _,emdbedding in embeddings]

    result = []

    model.eval()
    with torch.no_grad():
        transform = global_transform

        embeddings = torch.tensor(embeddings)

        try:
            face = face_label.FaceLabel(target)
            image = face.getImageData(g_image_size)
            image = transform(image)
            image = image.unsqueeze(0)
            embedding = model(image)

            embedding = embedding.squeeze()
            result = _selectName(names,embeddings,embedding,top,threshold)

        except Exception as e:
            logging.error(str(e))

    return result

def _selectName(names,embeddings,embedding,topN,threshold=0.0):
    
    cos = torch.matmul(embeddings, embedding) / (torch.sum(embeddings * embeddings, dim=1) * torch.sum(embedding * embedding) + 1e-9).sqrt()
    simiarity, topk = torch.topk(cos, k=topN)
    simiarity = simiarity.cpu().numpy()
    topk = topk.cpu().numpy()

    results = [(n,p) for n,p in zip(names[topk].tolist(),simiarity.tolist())]
    results = [x for x in filter(lambda result: result[1]>=threshold, results)]

    return results


if __name__ == '__main__':


    model,funcs = createModel()
    train(model,'data/train-recognition-extracted',num_epoch=10)
    torch.save(model.cpu().state_dict(), 'ai_face_recognition-cpu.pt')
