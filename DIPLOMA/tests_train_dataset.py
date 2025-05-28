import unittest
from unittest.mock import patch, MagicMock, mock_open
import torch
import sys

# Patch imports for modules not under test
sys.modules['custom_dataset_loader'] = MagicMock()
sys.modules['model_get'] = MagicMock()
sys.modules['normalizer'] = MagicMock()

import train_own_dataset


class TestTrainOwnDataset(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset with 10 samples
        self.mock_dataset = MagicMock()
        self.mock_dataset.__len__.return_value = 10
        self.mock_dataset.class_to_idx = {'a': 0, 'b': 1}
        self.mock_dataset.__getitem__.side_effect = lambda idx: (
            torch.zeros(3, 128, 128), torch.tensor(idx % 2)
        )

    @patch('train_own_dataset.DataLoader')
    @patch('train_own_dataset.compute_mean_std', return_value=([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]))
    def test_make_normilize_transform(self, mock_compute, mock_loader):
        transform, mean, std = train_own_dataset.make_normilize_transform(self.mock_dataset)
        self.assertIsNotNone(transform)
        self.assertEqual(mean, [0.5, 0.5, 0.5])
        self.assertEqual(std, [0.2, 0.2, 0.2])

    def test_make_normilize_transform_empty(self):
        empty_dataset = MagicMock()
        empty_dataset.__len__.return_value = 0
        with self.assertRaises(ValueError):
            train_own_dataset.make_normilize_transform(empty_dataset)

    @patch('train_own_dataset.DataLoader')
    @patch('train_own_dataset.get_model')
    @patch('train_own_dataset.train_test_split', return_value=([0, 1, 2, 3, 4, 5, 6, 7], [8, 9]))
    @patch('train_own_dataset.SubsetRandomSampler')
    @patch('train_own_dataset.nn.CrossEntropyLoss')
    @patch('train_own_dataset.optim.Adam')
    @patch('train_own_dataset.time.time', return_value=1000)
    @patch('train_own_dataset.open', new_callable=mock_open)
    @patch('train_own_dataset.torch.save')
    @patch('train_own_dataset.plt')
    def test_train(self, mock_plt, mock_save, mock_openfile, mock_time, mock_optim,
                   mock_loss_class, mock_sampler, mock_split, mock_get_model, mock_loader):
        # Mock model
        mock_model = MagicMock(spec=torch.nn.Module)
        mock_model.to.return_value = mock_model
        mock_model.state_dict.return_value = {}

        def model_forward(x):
            return torch.tensor([[0.6, 0.4]] * x.shape[0], device=x.device)

        mock_model.side_effect = model_forward

        mock_model.eval = MagicMock()
        mock_model.train = MagicMock()
        mock_model.parameters.return_value = []
        mock_get_model.return_value = mock_model

        # Mock optimizer
        mock_optimizer = MagicMock()
        mock_optim.return_value = mock_optimizer

        # Mock loss function
        def fake_loss_call(outputs, labels):
            t = torch.tensor(1.0, requires_grad=True)
            t.backward = MagicMock()
            return t

        mock_loss_instance = MagicMock()
        mock_loss_instance.side_effect = fake_loss_call
        mock_loss_class.return_value = mock_loss_instance

        # Fake dataloader output
        mock_loader.return_value = [(torch.zeros(2, 3, 128, 128), torch.tensor([0, 1]))] * 2

        # Call train
        train_own_dataset.train(self.mock_dataset, [0.5], [0.5], show_plot=False)

        self.assertTrue(mock_get_model.called)
        self.assertTrue(mock_loss_class.called)
        self.assertTrue(mock_optim.called)
        self.assertTrue(mock_save.called)


if __name__ == '__main__':
    unittest.main()
