
import unittest
from unittest.mock import patch, MagicMock
import torch

import dir_class_predit

class TestDirClassPredit(unittest.TestCase):
    def setUp(self):
        self.metadata = {
            'class_names': ['cat', 'dog'],
            'model_type': 'mobilenet_v3_small',
            'img_size': 32,
            'normalization_mean': [0.5, 0.5, 0.5],
            'normalization_std': [0.2, 0.2, 0.2]
        }
        self.device = 'cpu'
        self.num_classes = 2

    @patch('dir_class_predit.torch.load')
    def test_load_checkpoint(self, mock_load):
        mock_load.return_value = {'metadata': self.metadata}
        checkpoint, metadata = dir_class_predit.load_checkpoint('path', self.device)
        self.assertEqual(metadata, self.metadata)

    @patch('dir_class_predit.models')
    def test_build_model(self, mock_models):
        mock_model = MagicMock()
        mock_model.classifier = [MagicMock(), MagicMock()]
        mock_model.classifier[-1].in_features = 10
        mock_models.mobilenet_v3_small.return_value = mock_model
        model = dir_class_predit.build_model(self.metadata, self.num_classes)
        self.assertTrue(hasattr(model, 'classifier'))

    def test_get_transform(self):
        transform = dir_class_predit.get_transform(self.metadata)
        self.assertTrue(callable(transform))

    def test_get_actual_class(self):
        class_names = ['cat', 'dog']
        folder = 'dataset128/dog'
        actual = dir_class_predit.get_actual_class(class_names, folder)
        self.assertEqual(actual, 1)

    @patch('dir_class_predit.Image.open')
    @patch('dir_class_predit.os.listdir')
    def test_evaluate_folder(self, mock_listdir, mock_open):
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.__call__ = MagicMock(return_value=torch.tensor([[0.1, 0.9]]))
        mock_model.return_value = torch.tensor([[0.1, 0.9]])
        mock_listdir.return_value = ['img1.jpg', 'img2.jpg']
        mock_image = MagicMock()
        mock_image.convert.return_value = mock_image
        mock_open.return_value = mock_image
        transform = MagicMock(return_value=torch.zeros(3, 32, 32))
        with patch('dir_class_predit.predict_image', return_value=1):
            with patch('builtins.print'):
                acc = dir_class_predit.evaluate_folder(
                    mock_model, 'dataset128/dog', ['cat', 'dog'], transform, self.device
                )
        self.assertEqual(acc, 100.0)

if __name__ == '__main__':
    unittest.main()