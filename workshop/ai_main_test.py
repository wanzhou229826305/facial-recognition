import unittest
import os
import PIL
import numpy

from workshop import ai_main

class FaceRecognitionMain_test(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        

    def test_run_embedding_similarity_and_test_v10(self):
        modelName='face-recognition-v10'
        model,modelPath = ai_main.run_Train('','src/test-data/train-recognition-extracted',epoch=2,modelName='face-recognition-v10')

        self.assertTrue(model != None)

        embeddingFile = ai_main.run_embedding(modelPath,'src/test-data/faces',modelName)
        self.assertTrue(os.path.exists(embeddingFile))

        result = ai_main.run_similar(modelPath,'src/test-data/faces/annieh.jpg',modelName)
        self.assertTrue(result != None)

        imageObj = numpy.array(PIL.Image.open('src/test-data/faces/annieh.jpg'))
        result = ai_main.run_similar(modelPath,imageObj,modelName)
        self.assertTrue(result != None)

        ai_main.run_test(modelPath,'src/test-data/train-recognition-extracted',modelName)

    def test_run_embedding_similarity_and_test_vgg16(self):
        modelName='face-recognition-vgg16'
        model,modelPath = ai_main.run_Train('','src/test-data/train-recognition-extracted',epoch=2,modelName=modelName)

        self.assertTrue(model != None)

        embeddingFile = ai_main.run_embedding(modelPath,'src/test-data/faces',modelName)
        self.assertTrue(os.path.exists(embeddingFile))

        result = ai_main.run_similar(modelPath,'src/test-data/faces/annieh.jpg',modelName)
        self.assertTrue(result != None)

        imageObj = numpy.array(PIL.Image.open('src/test-data/faces/annieh.jpg'))
        result = ai_main.run_similar(modelPath,imageObj,modelName)
        self.assertTrue(result != None)

        ai_main.run_test(modelPath,'src/test-data/train-recognition-extracted',modelName)

    def test_train_segment(self):
        expectedVersion = 6
        model,modelPath = ai_main.run_Train('src/test-data/current_version.pt','src/test-data/face-label',5,modelName='face-detection')

        self.assertTrue(model != None)

        ai_main.run_segment(modelPath,'src/test-data/faces/ansheng.jpg',modelName='face-detection')




if __name__ == "__main__":
    unittest.main()