import torch
import torchvision
import torch.utils.data
import numpy as np
import tqdm
import torch
from torchvision import datasets, transforms
from logging import debug, info, warn, error
import os
from PIL import Image,ImageEnhance
import copy
import shutil
from workshop import face_label

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device('cpu')

class FaceLabelDatasetFromFolder(torch.utils.data.Dataset):
    def __init__(self, labelFolder):

        self.faceData = []
        self.loadFaceLabels(labelFolder)

    def __getitem__(self, index):
        return self.faceData[index]

    def __len__(self):
        return len(self.faceData)
    
    def loadFaceLabels(self,labelFolder):
        info('loading faces from %s'%labelFolder)

        entries = os.scandir(labelFolder)
        for entry in entries:
            if entry.is_file() and os.path.splitext(entry.name.lower())[1] == '.json':
                try:
                    face = face_label.FaceLabel(entry.path)
                    if os.path.exists(face.imagePath):
                        self.faceData.append(face)
                    else:
                        error("Image not found [%s],%s"%(face.imagePath,entry.path))

                except Exception as e:
                    error("Error handling [%s], %s"%(entry.path,str(e)))
        
        info('%d faces loaded from %s'%(len(self.faceData),labelFolder))

    def augment(self,destPath):
        if not os.path.exists(destPath):
            os.makedirs(destPath)
    
        lastName=''
        progress = tqdm.tqdm(self.faceData, desc='Data Augment,%s'.format(lastName))
        for face in progress:
            self.augmentFace(face,destPath)

    def augmentFace(self,face,destPath):
        folder = os.path.dirname(face.imagePath)
        filename,ext = os.path.splitext(os.path.basename(face.imagePath))

        shutil.copy(face.imagePath,os.path.join(destPath,os.path.basename(face.name)))
        face.saveAsJson(os.path.join(destPath,os.path.basename(face.name))+'.json')

        def scale(face,factor):
            # w,h = face.
            img = img.resize((int(w*factor),int(h*factor)))

            name = 'scale-%0.2f'%(factor)
            return img,name

        # def rotate(img,factor):
        #     img = img.rotate(factor)
        #     name = 'rotate-%d'%(factor)
        #     return img,name    

        def addNoise(face,factor=0.05):

            def add_noise(image_array, noise_factor):
                noisy_image_array = image_array.copy()
                noise = np.random.randn(*noisy_image_array.shape) * noise_factor
                noisy_image_array = noisy_image_array + noise
                noisy_image_array = np.clip(noisy_image_array, 0, 255).astype(np.uint8)
                return noisy_image_array

            img_arr = np.array(face.imageData)
            img_arr = add_noise(img_arr,factor)

            augmentedImage = Image.fromarray(img_arr)
            name = 'noise-%0.2f'%(factor)
            
            return augmentedImage,name
        
        def brightness(face,factor):
            enhancer = ImageEnhance.Brightness(face.imageData)
            augmentedImage = enhancer.enhance(factor)
            name = 'brightness-%0.2f'%(factor)
            return augmentedImage,name

        def flip(face,factor):
            augmentedImage = face.imageData.transpose(Image.FLIP_LEFT_RIGHT)
            name = 'flip'
            return augmentedImage,name

        result = [face]
        face.imageData = face.getImageData()

        def applyAlgorithmAndFactor(algorithm,factors):
            for factor in factors:
                augmentedImage,augmentDesc = algorithm(face,factor)
                
                copyOfFace = copy.copy(face)
                copyOfFace.name = '%s-%s%s'%(filename,augmentDesc,ext)

                augmentedImagePath = os.path.join(destPath,copyOfFace.name)
                augmentedImage.save(augmentedImagePath)
                augmentedLabelPath = os.path.join(destPath,'%s%s'%(copyOfFace.name,'.json'))
                copyOfFace.saveAsJson(augmentedLabelPath)

        
        # factors = range(80,120,10)
        # factors = [option/100 for option in factors]
        # if 1 in factors:
        #     factors.remove(1)
        # applyAlgorithmAndFactor(scale,factors)

        factors = [1.2,1.5,1.8]
        applyAlgorithmAndFactor(addNoise,factors)
        
        # factors = range(-45,45,15)
        # factors = [factor for factor in factors]
        # if 0 in factors:
        #     factors.remove(0)
        # applyAlgorithmAndFactor(rotate,factors)

        factors = [0.8,1.2,1.4]
        applyAlgorithmAndFactor(brightness,factors)

        # applyAlgorithmAndFactor(flip,[0])
        return result

imageTypes =['.jpg','.jpeg','.png']
class FaceDatasetFromFolder(torch.utils.data.Dataset):
    def __init__(self, labelFolder):
        self.transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
        
        self.faceData = []
        self.loadFaces(labelFolder)
        
    def __getitem__(self, index):
        image = self.faceData[index].getImageData((224,224))
        image = self.transform(image).to(device)
        return image

    def __len__(self):
        return len(self.faceData)
    
    def loadFaces(self,labelFolder):
        info('loading faces from %s'%labelFolder)

        entries = os.scandir(labelFolder)
        for entry in entries:
            _,ext = os.path.splitext(entry.name.lower())
            if entry.is_file() and ext in imageTypes:
                try:
                    self.faceData.append(face_label.FaceLabel(entry.path))
                except Exception as e:
                    error("Error handling [%s], %s"%(entry.path,str(e)))
        
        info('%d faces loaded from %s'%(len(self.faceData),labelFolder))
        

