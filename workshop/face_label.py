import json
import os
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
import PIL.ImageDraw
from PIL import Image, ImageEnhance,ImageFilter

class FaceLabel:
    def __init__(self, labelFile) -> None:

        self.imageData = None
        self.videoFrame = None
        self.rectangles = None
        if type(labelFile) is str:
            filename,ext = os.path.splitext(os.path.basename(labelFile.lower()))
            if ext == '.json':
                self.loadFromFile(labelFile)
            else:
                img = PIL.Image.open(labelFile)
                self.points = None
                self.imageWidth,self.imageHeight = img.size
                self.name = filename
                self.folderName = os.path.basename(os.path.dirname(labelFile))
                self.imagePath = labelFile
        else:
            self.videoFrame = labelFile
            self.points = None
            
            image = PIL.Image.fromarray(self.videoFrame)
            
            self.imageWidth,self.imageHeight = image.size
            self.name = "N/A"
            self.imagePath = "N/A"

            
    def loadFromFile(self, labelFile):
        try:
            with open(labelFile, 'rb') as f:
                data = json.load(f)

            asset = data['asset']
            region= data['regions']

            name = asset['name']
            
            imagePath = os.path.join(os.path.dirname(labelFile), name)

            imageWidth = asset['size']['width']
            imageHeight = asset['size']['height']

            points = [ (p['x'],p['y']) for p in region[0]['points']]

        except Exception as e:
            raise e

        self.points = points
        self.imagePath = imagePath
        self.name = name
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight

    def saveAsJson(self,filePath):
        asset ={}

        asset['name'] = self.name
        w,h = self.getImageData().size
        size = {'width':w,'height':h}
        asset['size'] = size

        points = [ {'x':p[0],'y':p[1]} for p in self.points ]
        regions=[{'points':points}]

        d = {'asset':asset,
             'regions':regions}
        
        with open(filePath, 'w') as f:
            f.write(json.dumps(d,indent=4))

        return filePath



    def attachRectangles(self,rectangles):
        self.rectangles = rectangles

    def attachMask(self,mask):

        rectList = self.getRectFromMask(mask)


        image = self.getImageData(mask.shape)
        # mask =PIL.Image.fromarray(mask,mode='L').convert('RGBA')
        # mask_img = PIL.Image.blend(image.convert('RGBA'),
        #                mask.convert('RGBA'), 0.6)

        canvas = PIL.ImageDraw.Draw(image)


        for rect in rectList:
            canvas.rectangle(rect,outline ="red")


        # image.show()

        return image


    def getRectFromMask(self,mask):
        results = []

        w,h = mask.shape

        def search(x, y):
            l,t,r,b = x,y,x,y

            pending = []
            pending.append((x,y))

            while len(pending) > 0:
                xx,yy = pending.pop()
                mask[yy][xx] = 2
                if xx+1 < w:
                    if not mask[yy][xx+1]:
                        pending.append((xx+1,yy))
                        r = max(r,xx+1)

                if xx - 1 >= 0:
                    if not mask[yy][xx - 1]:
                        pending.append((xx-1,yy))
                        l = min(l,xx-1)

                if yy+1 < h:
                    if not mask[yy+1][xx]:
                        pending.append((xx,yy+1))
                        b = max(b,yy+1)

                if yy-1 >= 0:
                    if not mask[yy-1][xx]:
                        pending.append((xx,yy-1))
                        t = min(t,yy - 1)

            return l,t,r,b

        def reduce(l,t,r,b):

            kernelSize = 15
            if (r-l) <= 4*kernelSize or (b-t) <=4* kernelSize: 
                return l,t,r,b
            
            centerX, centerY = int(l + (r-l)/2), int(t + (b-t)/2)

            x1, x2 = max(centerX - int(kernelSize / 2),
                         0), min(centerX+int(kernelSize/2), w)
            y1, y2 = max(centerY - int(kernelSize/2),
                         0), min(centerY+int(kernelSize/2), h)

            innerL, innerR, innerT, innerB = 0, w, 0, h
            for i in range(centerX, 0, -1):
                for j in range(y1, y2+1):
                    if mask[j][i] != 2:
                        innerL = i
                        break
                if innerL != 0:
                    break

            for i in range(centerX, w):
                for j in range(y1, y2+1):
                    if mask[j][i] != 2:
                        innerR = i
                        break
                if innerR != w:
                    break

            for j in range(centerY, 0, -1):
                for i in range(x1, x2+1):
                    if mask[j][i] != 2:
                        innerT = j
                        break
                if innerT != 0:
                    break
            for j in range(centerY, h):
                for i in range(x1, x2+1):
                    if mask[j][i] != 2:
                        innerB = j
                        break
                if innerB != h:
                    break

            return innerL, innerT, innerR, innerB

        for y in range(0,w):
            for x in range(h):
                if mask[y][x]:
                    pass
                else:
                   l,t,r,b = search(x,y)
                   l,t,r,b = reduce(l,t,r,b)
                   if (r-l) * (b-t) >= 1600:
                        results.append((l,t,r,b))
        
        results = sorted(results, reverse=True,key=lambda rect: (rect[2]-rect[0]) * (rect[3]-rect[1]))

        return results

    def extract(self,mask):

        rectList = self.getRectFromMask(mask)

        if len(rectList) <= 0:
            return None,None

        image = self.getImageData(mask.shape)
        image = image.crop(rectList[0])

        return image,rectList[0]
    
    def extractToFile(self,mask,destPath):
        image,rect = self.extract(mask)
        if image:
            filename,ext = os.path.splitext(os.path.basename(self.imagePath))
            filename = os.path.join(destPath,filename+ext)
            image.save(filename)
            return filename 
        return None

    def saveImageWithLabel(self,destPath,size=None):

        image = self.getImageWithLabel()

        filename,ext = os.path.splitext(os.path.basename(self.imagePath))
        filename = os.path.join(destPath,filename+ext)

        image = self._resizeImage(image,size)
        image.save(filename)
        return filename
    
    def getImageWithLabel(self):
        image = self.getImageData()
        canvas = PIL.ImageDraw.Draw(image)
        if self.points != None:
            canvas.polygon(self.points, outline ="red",width=5)
        if self.rectangles != None:
            for rect in self.rectangles:
                canvas.rectangle(rect,outline ="red",width=5)
        return image

    def _resizeImage(self,img,size=None):
        if size != None:
            # img.thumbnail(size)
            # w,h = size
            # l = (img.width-w)/2
            # t = (img.height-h)/2
            # img = img.crop((l,t,l+w,t+h))

            img = img.resize(size)

        return img


    def getMask(self,size =  None):
        # Create a blank mask image
        mask = PIL.Image.new('L', (self.imageWidth,self.imageHeight), 0)

        # Draw the polygon on the mask
        draw = PIL.ImageDraw.Draw(mask)
        draw.polygon(self.points, fill=255)

        # Convert the mask to binary
        mask = mask.convert('1')

        mask = self._resizeImage(mask,size)

        return mask
    
    def getGrayImageData(self,size=None):
        if self.videoFrame is not None:
            image = PIL.Image.fromarray(self.videoFrame)
            image = image.convert('L')
            image = self._resizeImage(image,size)
        else:
            image = PIL.Image.open(self.imagePath)
            image = image.convert('L')
            image = self._resizeImage(image,size)
        return image
    
    def getProcessedImageData(self,size=None):
        image = self.getImageData(size)
        return self._applyImageProcessors(image)
        
    def _applyImageProcessors(self,image):
        image = image.convert('L')
        image = image.filter(ImageFilter.MedianFilter(size=3))
        sharp = ImageEnhance.Sharpness(image)
        image = sharp.enhance(1.5)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(4.0)
        return image
    
    def getImageData(self,size=None):
        if self.videoFrame is not None:
            image = PIL.Image.fromarray(self.videoFrame)
            image = image.convert('RGB')
            image = self._resizeImage(image,size)
        else:
            image = PIL.Image.open(self.imagePath)
            image = image.convert('RGB')
            image = self._resizeImage(image,size)
        return image

    
    def saveMask(self,destPath,size=None):
        mask = self.getMask()
        image = self.getImageData()

        # Apply the mask to the image
        result = PIL.Image.composite(image, PIL.Image.new('RGB', image.size, (0, 0, 0)), mask)
        result = self._resizeImage(result,size)

        filename,ext = os.path.splitext(os.path.basename(self.imagePath))
        filename = os.path.join(destPath,filename+'-mask'+ext)

        result.save(filename)

        return filename
    
 