import unittest
import sys
import torch.utils
import PIL

from workshop import ai_03_face_detection_cnn

class FaceDetectionCNN_test(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_createModel_default(self):
        model,funcs = ai_03_face_detection_cnn.createModel()
        self.assertTrue(model != None)

    def test_train_and_extractFace_current(self):
        model,funcs = ai_03_face_detection_cnn.createModel()
        self.assertTrue(model != None)

        ai_03_face_detection_cnn.train(model,'src/test-data/face-label',5)

        face = ai_03_face_detection_cnn.segment(model,'src/test-data/face-label/aulionh.jpg')

if __name__ == "__main__":
    unittest.main()