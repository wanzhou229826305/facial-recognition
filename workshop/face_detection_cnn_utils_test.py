import sys

import unittest
# import utils
import torch.utils
import PIL

sys.path.insert(0, './ai_workshop')
import face_detection_cnn_utils
import face_label

class face_detection_cnn_utils_utils_test(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def testFaceLabelDataset(self):
        testee = face_detection_cnn_utils.FaceLabelDatasetFromFolder('src/test-data/face-label')
        self.assertTrue(len(testee) == 3)

        for face in testee:
            self.assertTrue(type(face) is face_label.FaceLabel)


    def test_augment(self):
        testee = face_detection_cnn_utils.FaceLabelDatasetFromFolder('src/test-data/face-label')
        self.assertTrue(len(testee) == 3)

        testee.augment('src/test-data/face-label-augmented')

    def testFaceDataset(self):
        testee = face_detection_cnn_utils.FaceDatasetFromFolder('src/test-data/faces')
        self.assertTrue(len(testee) >= 3)

        loader = torch.utils.data.DataLoader(testee,batch_size=16)

        for imageData in loader:
            self.assertTrue(len(imageData) >= 3)

if __name__ == "__main__":
    unittest.main()