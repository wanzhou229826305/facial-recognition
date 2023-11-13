import torch,torchvision
import torch.utils.data
import torch.nn as nn
import numpy as np
import tqdm,logging

import face_label,face_detection_cnn_utils

currentModelName = 'face-detection'

class ModelFunctions():
    def __init__(self,train) -> None:
        self.train = train

def createModel(modelName=currentModelName):
    if modelName == currentModelName:
        model = FaceDetectionModel()
        funcs = ModelFunctions(train)
        funcs.segment = segment
        return model,funcs
    return None

transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    
class FaceDetectionModel(nn.Module):
    def __init__(self) -> None:
        super(FaceDetectionModel, self).__init__()
        self.encode1 = nn.Sequential(nn.Conv2d(3, 64, 3,padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(True),
                            nn.MaxPool2d(2,2))
        
        self.encode2 = nn.Sequential(nn.Conv2d(64, 128, 3,padding=1),
                            nn.BatchNorm2d(128),
                            nn.ReLU(True),
                            nn.MaxPool2d(2,2))

        self.encode3 = nn.Sequential(nn.Conv2d(128, 256, 3,padding=1),
                            nn.BatchNorm2d(256),
                            nn.ReLU(True),
                            nn.MaxPool2d(2,2))
        
        self.encode4 = nn.Sequential(nn.Conv2d(256, 512, 3,padding=1),
                            nn.BatchNorm2d(512),
                            nn.ReLU(True),
                            nn.MaxPool2d(2,2))
        
        self.encode5 = nn.Sequential(nn.Conv2d(512, 512, 3,padding=1),
                            nn.BatchNorm2d(512),
                            nn.ReLU(True),
                            nn.MaxPool2d(2,2))
        
        self.decode1 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=3,padding=1,stride=2,output_padding=1),
                            nn.BatchNorm2d(256),
                            nn.ReLU(True))
        
        self.decode2 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=3,padding=1,stride=2,output_padding=1),
                            nn.BatchNorm2d(128),
                            nn.ReLU(True))
        
        self.decode3 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3,padding=1,stride=2,output_padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(True))
        
        self.decode4 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=3,padding=1,stride=2,output_padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(True))
        
        self.decode5 = nn.Sequential(nn.ConvTranspose2d(32, 16, kernel_size=3,padding=1,stride=2,output_padding=1),
                            nn.BatchNorm2d(16),
                            nn.ReLU(True))
        
        self.classifier = nn.Sequential(nn.Conv2d(16, 2, kernel_size=1),
                                        nn.Sigmoid())

    def forward(self, x):
        x = self.encode1(x)
        x = self.encode2(x)
        x = self.encode3(x)
        x = self.encode4(x)
        x = self.encode5(x)

        x = self.decode1(x)
        x = self.decode2(x)
        x = self.decode3(x)
        x = self.decode4(x)
        x = self.decode5(x)

        x = self.classifier(x)

        return x

def train(model, dataPath, num_epoch):

    def collate_FaceObject(facesWithLabel):
        def faceDataToTensor(faceWithLabel):
            if faceWithLabel.imageData == None:
                faceWithLabel.imageData = faceWithLabel.getImageData((224, 224))
                faceWithLabel.imageData = transform(faceWithLabel.imageData)
            return faceWithLabel.imageData
        def faceLabelToTensor(faceWithLabel):
            label = np.array(faceWithLabel.getMask((224,224)))
            label = np.eye(2)[label.astype('uint8')].astype('float32')
            label = torch.from_numpy(label.transpose((2,0,1)))  
            return label   

        faces = [faceDataToTensor(faceWithLabel) for faceWithLabel in facesWithLabel]
        labels = [faceLabelToTensor(faceWithLabel) for faceWithLabel in facesWithLabel]
        return torch.stack(faces),torch.stack(labels)


    faceImageSet = face_detection_cnn_utils.FaceLabelDatasetFromFolder(dataPath)
    loader = torch.utils.data.DataLoader(faceImageSet.faceData,batch_size=16,shuffle=1,collate_fn=collate_FaceObject)

    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)        
    loss_function = torch.nn.BCELoss()
    model.train()

    currentLoss = 0.0
    for epoch in range(num_epoch):
        progress = tqdm.tqdm(loader, desc='{}/{},loss={}'.format(epoch,
                             num_epoch, '%0.15f' % (currentLoss)))

        trainNumber = 0
        trainLoss = 0
        for imageData, label in progress:
            output = model(imageData)
            loss = loss_function(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainLoss += loss.item() * len(imageData)
            trainNumber += len(imageData)

        currentLoss = trainLoss / trainNumber
        logging.debug('train==> %d/%d loss=%0.15f'%(epoch,num_epoch,currentLoss))

def segment(model,targetImageFile):

    result = None

    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])])

    model.eval()
    with torch.no_grad():
        try:
            face = face_label.FaceLabel(targetImageFile)

            image = face.getImageData((224,224))
            image = transform(image)
            image = image.unsqueeze(0)
            output = model(image)

            output_np = output.cpu().data.numpy().copy()
            output_np = np.argmin(output_np, axis=1)
            output_np = output_np*255
            result = face.attachMask(np.uint8(output_np.squeeze()))
            result.show()
        except Exception as e:
            logging.error(str(e))

    return result


if __name__ == '__main__':

    model,funcs = createModel()
    train(model,'data/train-segment',num_epoch=10)

    torch.save(model.cpu().state_dict(), 'ai_face_detection-cpu.pt')

    segment(model,'data/train-segment/yonwei.jpg')