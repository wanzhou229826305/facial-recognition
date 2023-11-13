import unittest


import os
import numpy as np
import PIL

import face_label

class FaceLabel_test(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def testLoadFrom(self):
        testee = face_label.FaceLabel('src/test-data/face-label/0e02f839a66b66a9ea5639fb5d45f134-asset.json')
        self.assertTrue(testee.name == 'aulionh.jpg')
        self.assertTrue(testee.imageWidth == 315)
        self.assertTrue(testee.imageHeight == 420)
        self.assertTrue(len(testee.points) > 0)

    def test_saveAsJson(self):
        testee = face_label.FaceLabel('src/test-data/face-label/0e02f839a66b66a9ea5639fb5d45f134-asset.json')

        savedLabelFile = testee.saveAsJson(testee.name+'.json')
        testee = face_label.FaceLabel(savedLabelFile)


    def test_saveImageWithLabel(self):
        testee = face_label.FaceLabel('src/test-data/face-label/0e02f839a66b66a9ea5639fb5d45f134-asset.json')
        filename = testee.saveImageWithLabel('src/test-data')
        self.assertTrue(os.path.exists(filename))

    def test_saveImageWithLabel_224x224(self):
        testee = face_label.FaceLabel('src/test-data/face-label/0e02f839a66b66a9ea5639fb5d45f134-asset.json')
        filename = testee.saveImageWithLabel('src/test-data',(224,224))
        self.assertTrue(os.path.exists(filename))

    def test_saveMask_224x224(self):
        testee = face_label.FaceLabel('src/test-data/face-label/0e02f839a66b66a9ea5639fb5d45f134-asset.json')
        filename = testee.saveMask('src/test-data',(224,224))
        self.assertTrue(os.path.exists(filename))

    def test_getGrayImageData(self):
        testee = face_label.FaceLabel('src/test-data/face-label/0e02f839a66b66a9ea5639fb5d45f134-asset.json')
        img = testee.getGrayImageData((224,224))
        self.assertTrue(img.mode == 'L')

    def test_getProcessedImageData(self):
        testee = face_label.FaceLabel('src/test-data/face-label/0e02f839a66b66a9ea5639fb5d45f134-asset.json')
        img = testee.getProcessedImageData((224,224))
        self.assertTrue(img != None)
        # img.show() 

        # img.show()

    # def test_fromArray(self):
    #     arr1 = np.zeros((112,224))
    #     arr2 = np.ones((112,224))

    #     arr = np.append(arr1,arr2,axis=0)*255
    #     img=PIL.Image.fromarray(np.uint8(arr),mode='L')
    #     img.show()

    #             #创建一个3*3的数组
    #     arr=np.array([[1,2,3],[4,5,6],[255,255,255]])

    #     #将数组转换为灰度图像
    #     img=PIL.Image.fromarray(np.uint8(arr),mode='L')
    #     img.show()

    def test_getRectFromMask(self):
        arr1 = np.zeros((112,224))
        arr2 = np.ones((112,224))

        arr = np.append(arr1,arr2,axis=0)*255
        arr = np.uint8(arr)

        testee = face_label.FaceLabel('src/test-data/face-label/0e02f839a66b66a9ea5639fb5d45f134-asset.json')
        rectList = testee.getRectFromMask(arr)
        self.assertTrue(len(rectList)>0)
        
    def test_extract_to_imageObj(self):
        arr1 = np.zeros((112,224))
        arr2 = np.ones((112,224))

        arr = np.append(arr1,arr2,axis=0)*255
        arr = np.uint8(arr)

        testee = face_label.FaceLabel('src/test-data/face-label/0e02f839a66b66a9ea5639fb5d45f134-asset.json')
        imageObj,rect = testee.extract(arr)
        self.assertTrue(imageObj != None)
        self.assertTrue(type(imageObj) == PIL.Image.Image)
        
        self.assertTrue(rect == (0,0,224,112))

   
    def test_extractToFile(self):
        arr1 = np.zeros((112,224))
        arr2 = np.ones((112,224))

        arr = np.append(arr1,arr2,axis=0)*255
        arr = np.uint8(arr)

        testee = face_label.FaceLabel('src/test-data/face-label/0e02f839a66b66a9ea5639fb5d45f134-asset.json')
        result = testee.extractToFile(arr,'src/test-data')
        self.assertTrue(os.path.exists(result))

if __name__ == "__main__":
    unittest.main()