import unittest
from workshop import ai_04_face_recognition_cnn

class FaceRecognitionCNN_test(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_train(self):
        model,funcs = ai_04_face_recognition_cnn.createModel()
        self.assertTrue(model != None)

        funcs.train(model,'src/test-data/train-recognition-extracted',num_epoch=2)

    def test_train_embedding(self):
        model,funcs = ai_04_face_recognition_cnn.createModel()

        self.assertTrue(model != None)

        funcs.train(model,'src/test-data/train-recognition-extracted',num_epoch=2)

        funcs.embedding(model,'src/test-data/faces')


if __name__ == "__main__":
    unittest.main()