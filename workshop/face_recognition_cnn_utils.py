import torch
import numpy as np
import tqdm
import os,re,logging,random,shutil
from PIL import Image,ImageEnhance
import face_label

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device('cpu')

imageTypes =['.jpg','.jpeg','.png']
number_of_negative_faces = 5


class FacePairDatasetFromFolder(torch.utils.data.Dataset):
    def __init__(self, labelFolder):
        super(FacePairDatasetFromFolder,self).__init__()
        self.faceData = []
        self.faceGroup = {}
        self.loadFaceLabels(labelFolder)
        self.buildFaceGroups()
        self.buildFacePairs()

    def __getitem__(self, index):
        return self.facePairs[index]

    def __len__(self):
        return len(self.facePairs)
    
    def loadFaceLabels(self,labelFolder):

        logging.info('Loading faces from root folder [%s]'%(labelFolder))

        def loadOnePerson(folder):
            name = os.path.basename(folder)
            faces = []
            faceFiles = os.scandir(folder)
            for faceFile in faceFiles:
                if faceFile.is_file() and os.path.splitext(faceFile.name.lower())[1] in imageTypes:
                    face = face_label.FaceLabel(faceFile.path)
                    face.name = "%s-%s"%(name,faceFile.name)
                    faces.append(face)
            return faces

        rootEntries = os.scandir(labelFolder)
        for rootEntry in rootEntries:
            if rootEntry.is_dir():
                self.faceData += loadOnePerson(rootEntry.path)

        self.faceData = sorted(self.faceData,key=lambda p: p.name)

        logging.info('%d faces from [%s]'%(len(self.faceData),labelFolder))

    def buildFaceGroups(self):
        faceGroup = {}
        def extractName(name):
            x = re.search(r'([\d|\w]+).*',name)
            if x != None:
                groups = x.groups()
                if len(groups) > 0:
                    return groups[0]
            return name,name
        
        for anotherFace in self.faceData:
            name = extractName(anotherFace.name)
            if faceGroup.get(name) == None:
                faceGroup[name] = []
            
            faceGroup[name].append(anotherFace)

        faceGroup = {name: sorted(faces,key=lambda x:(len(x.name),x.name)) for name,faces in faceGroup.items()}
        self.faceGroup = faceGroup

    def buildFacePairs(self):

        logging.info('Building pairs, [%d] faces'%(len(self.faceData)))

        logging.info('Building pairs,[%d] persons, [%d] faces'%(len(self.faceGroup),len(self.faceData)))

        idx = [idx for idx,_ in enumerate(self.faceGroup)]
        idx_to_face = {idx:group for idx,group in enumerate(self.faceGroup)}
        face_to_idx = {group:idx for idx,group in enumerate(self.faceGroup)}

        facePairs=[]
        for (name,faces) in self.faceGroup.items():
            anchor = faces[0]
            for anotherFace in faces:
                if anotherFace is not anchor:
                    logging.debug('pair,1,%s, %s'%(anchor.name,anotherFace.name))
                    facePairs.append((anchor,anotherFace,1))

            others = idx.copy()
            others.pop(face_to_idx[name])

            for i in range(number_of_negative_faces):
                random.shuffle(others)

                negativeFaces = self.faceGroup[idx_to_face[others[0]]]
                negativeFaceIdx = random.choice(range(len(negativeFaces)))
                negativeFace = negativeFaces[negativeFaceIdx]
                logging.debug('pair,-1,%s, %s'%(anchor.name,negativeFace.name))
                facePairs.append((anchor,negativeFace,-1))

        random.shuffle(facePairs)

        self.facePairs = facePairs

        logging.info('Building pairs,[%d] persons, [%d] faces, [%d] pairs'%(len(self.faceGroup),len(self.faceData),len(facePairs)))

class FaceDatasetFromFolder(torch.utils.data.Dataset):
    def __init__(self, labelFolder):

        self.faceData = []
        self.loadFaces(labelFolder)
        
    def __getitem__(self, index):
        return self.faceData[index]

    def __len__(self):
        return len(self.faceData)
    
    def loadFaces(self,labelFolder):

        logging.info('Loading faces from root folder [%s]'%(labelFolder))

        def loadOnePerson(folder):
            name = os.path.basename(folder)
            faces = []
            faceFiles = os.scandir(folder)
            for faceFile in faceFiles:
                if faceFile.is_file() and os.path.splitext(faceFile.name.lower())[1] in imageTypes:
                    face = face_label.FaceLabel(faceFile.path)
                    face.name = "%s-%s"%(name,faceFile.name)
                    faces.append(face)
            return faces

        loadOnePerson(labelFolder)

        rootEntries = os.scandir(labelFolder)
        for rootEntry in rootEntries:
            if rootEntry.is_dir():
                self.faceData += loadOnePerson(rootEntry.path)

        self.faceData = sorted(self.faceData,key=lambda p: p.name)

        logging.info('%d faces from [%s]'%(len(self.faceData),labelFolder))

class FaceDatasetFromFolder_DataAugment(FaceDatasetFromFolder):
    def __init__(self, folder):
        super(FaceDatasetFromFolder_DataAugment, self).__init__(folder)
        
    def augment(self,destPath):

        lastName=''
        progress = tqdm.tqdm(self.faceData, desc='Data Augment,%s'.format(lastName))
        for face in progress:
            lastName = face.name
            pathOfFace = os.path.dirname(face.imagePath)
            pathOfFace = os.path.join(destPath,os.path.basename(pathOfFace))
            if not os.path.exists(pathOfFace):
                os.makedirs(pathOfFace)

            self.augmentFace(face.imagePath,pathOfFace)

    def augmentFace(self,imagePath,destPath):
        folder = os.path.dirname(imagePath)
        filename,ext = os.path.splitext(os.path.basename(imagePath))

        shutil.copy(imagePath,os.path.join(destPath,os.path.basename(imagePath)))

        def scale(img,factor):
            w,h = img.size
            img = img.resize((int(w*factor),int(h*factor)))

            name = 'scale-%0.2f'%(factor)
            return img,name

        def rotate(img,factor):
            img = img.rotate(factor)
            name = 'rotate-%d'%(factor)
            return img,name    

        def addNoise(img,factor=0.05):
            def add_noise(image_array, noise_factor):
                noisy_image_array = image_array.copy()
                noise = np.random.randn(*noisy_image_array.shape) * noise_factor
                noisy_image_array = noisy_image_array + noise
                noisy_image_array = np.clip(noisy_image_array, 0, 255).astype(np.uint8)
                return noisy_image_array

            img_arr = np.array(img)
            img_arr = add_noise(img_arr,factor)
            img = Image.fromarray(img_arr)

            name = 'noise-%0.2f'%(factor)
            return img,name
        
        result = [imagePath]
        originImg = Image.open(imagePath)

        def applyAlgorithmAndFactor(algorithm,factors):
            result =[]
            for factor in factors:
                augmentedImg,augmentDesc = algorithm(originImg,factor)
                filePath = os.path.join(destPath,'%s-%s%s'%(filename,augmentDesc,ext))
                augmentedImg.save(filePath)
                # augmentedImg.show()
                result.append(filePath)
            return result
        
        def brightness(img,factor):
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
            name = 'brightness-%0.2f'%(factor)
            return img,name
        
        def flip(img,factor):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            name = 'flip'
            return img,name

        factors = [0.8,1.2,1.6]
        applyAlgorithmAndFactor(scale,factors)

        # noiseFactors = range(1,10,2)
        # noiseFactors = [noise/100 for noise in noiseFactors]
        # applyAlgorithmAndFactor(addNoise,noiseFactors)
        
        # factors = range(-45,45,15)
        # factors = [factor for factor in factors]
        # if 0 in factors:
        #     factors.remove(0)
        # applyAlgorithmAndFactor(rotate,factors)

        factors = [0.8,1.4]
        applyAlgorithmAndFactor(brightness,factors)

        applyAlgorithmAndFactor(flip,[0])
        return result



class FaceDatasetFromFolder_Flat(torch.utils.data.Dataset):
    def __init__(self, labelFolder):
        self.faceData = []
        self.loadFaces(labelFolder)
        
    def __getitem__(self, index):
        return self.faceData[index]

    def __len__(self):
        return len(self.faceData)
    
    def loadFaces(self,labelFolder):
        logging.info('loading faces from %s'%labelFolder)

        entries = os.scandir(labelFolder)
        for entry in entries:
            _,ext = os.path.splitext(entry.name.lower())
            if entry.is_file() and ext in imageTypes:
                try:
                    self.faceData.append(face_label.FaceLabel(entry.path))
                except Exception as e:
                    logging.error("Error handling [%s], %s"%(entry.path,str(e)))
        
        logging.info('%d faces loaded from %s'%(len(self.faceData),labelFolder))
   