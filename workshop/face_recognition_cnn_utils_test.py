import unittest
import face_recognition_cnn_utils

class FaceRecognitionCNN_test(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_FaceLabelDataset(self):
        testee = face_recognition_cnn_utils.FacePairDatasetFromFolder('src/test-data/train-recognition-extracted')
        self.assertTrue(len(testee) > 0)

        for facePair in testee:
            self.assertTrue(facePair != None)

    def test_FaceDatasetFromFolder_Flat(self):
        testee = face_recognition_cnn_utils.FaceDatasetFromFolder_Flat('src/test-data/faces')
        self.assertTrue(len(testee) > 0)

        for facePair in testee:
            self.assertTrue(facePair != None)

    def test_FaceDatasetFromFolder(self):
        testee = face_recognition_cnn_utils.FaceDatasetFromFolder('src/test-data/train-recognition-extracted')
        self.assertTrue(len(testee) > 0)

        for facePair in testee:
            self.assertTrue(facePair != None)

    def test_dataAugment(self):
        datasetAugment = face_recognition_cnn_utils.FaceDatasetFromFolder_DataAugment('src/test-data/train-recognition-extracted')
        datasetAugment.augment('unittest/train-recognition-augmented')

if __name__ == "__main__":
    unittest.main()