import unittest
import numpy as np
import cv2
import os
from unittest.mock import patch, MagicMock
import dataset_generator
import sys

from dataset_generator import apply_augmentations, save_augmented_images


class TestVideoProcessing(unittest.TestCase):

    def setUp(self):
        """Ініціалізація необхідних даних перед кожним тестом"""
        # Створимо тестове зображення 128x128 з випадковими кольорами
        self.test_img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        self.settings = {
            "rotate": True,
            "flip": True,
            "blur": True,
            "brightness": True,
            "contrast": True,
            "noise": True,
            "noise_level": 10,
            "brightness_level": 1.2,
            "width": 128,
            "height": 128,
            "stop_at": 0,
            "rewrite": True,
        }

    def test_apply_augmentations(self):
        """Тестування функції `apply_augmentations`"""
        augmented_images = apply_augmentations(self.test_img, self.settings)

        # Перевіряємо, що результуючий список містить більше одного зображення
        self.assertGreater(len(augmented_images), 1)

        # Перевіряємо, що результуючі зображення мають правильний розмір
        for img in augmented_images:
            self.assertEqual(img.shape, (128, 128, 3))

    @patch("cv2.imwrite")
    def test_save_augmented_images(self, mock_imwrite):
        """Тестування функції `save_augmented_images`"""
        save_path = "test_dataset"
        os.makedirs(save_path, exist_ok=True)

        # Ініціалізуємо дані для збереження
        frame_count = 0
        x, w, y, h = 10, 50, 20, 60
        frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
        class_name = "test_class"
        annotations = []

        # Викликаємо функцію
        save_augmented_images(
            obj_img=self.test_img,
            settings=self.settings,
            save_path=save_path,
            frame_count=frame_count,
            x=x,
            w=w,
            y=y,
            h=h,
            frame=frame,
            class_name=class_name,
            annotations=annotations
        )

        # Перевіряємо, чи були виклики cv2.imwrite
        self.assertGreater(mock_imwrite.call_count, 0)

        # Перевіряємо, що анотації додано
        self.assertGreater(len(annotations), 0)
        self.assertIn("bbox", annotations[0])
        self.assertIn("image", annotations[0])

    def tearDown(self):
        """Очищення після тестів"""
        if os.path.exists("test_dataset"):
            import shutil
            shutil.rmtree("test_dataset")



class TestDatasetGeneratorExtra(unittest.TestCase):
    def setUp(self):
        self.test_img = np.ones((128, 128, 3), dtype=np.uint8) * 127
        self.settings = {
            "rotate": True,
            "flip": True,
            "blur": True,
            "brightness": True,
            "contrast": True,
            "noise": True,
            "noise_level": 10,
            "brightness_level": 1.2,
            "width": 128,
            "height": 128,
            "stop_at": 0,
            "rewrite": True,
        }

    def test_bbox_normalization(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        annotations = []
        with patch("cv2.imwrite"):
            save_augmented_images(
                obj_img=self.test_img,
                settings=self.settings,
                save_path="test_dataset",
                frame_count=0,
                x=50, w=50, y=20, h=40,
                frame=frame,
                class_name="classA",
                annotations=annotations
            )
        bbox = annotations[0]["bbox"]
        # x_center = (50+25)/200 = 0.375, y_center = (20+20)/100 = 0.4, w_norm = 50/200 = 0.25, h_norm = 40/100 = 0.4
        self.assertAlmostEqual(bbox[0], 0.375, places=3)
        self.assertAlmostEqual(bbox[1], 0.4, places=3)
        self.assertAlmostEqual(bbox[2], 0.25, places=3)
        self.assertAlmostEqual(bbox[3], 0.4, places=3)

    def test_apply_augmentations_original_first(self):
        imgs = apply_augmentations(self.test_img, self.settings)
        self.assertTrue(np.array_equal(imgs[0], self.test_img))

    def test_all_augmentations(self):
        imgs = dataset_generator.apply_augmentations(self.test_img, self.settings)
        self.assertGreaterEqual(len(imgs), 7)

    @patch("dataset_generator.select_video", return_value="dummy.mp4")
    @patch("dataset_generator.select_settings", return_value={})
    @patch("dataset_generator.process_video")
    def test_main(self, mock_process, mock_settings, mock_video):
        with patch.object(sys, 'argv', ["dataset_generator.py"]):
            dataset_generator.main()
            mock_process.assert_called_once()

    def test_invalid_settings(self):
        bad_settings = {}
        imgs = dataset_generator.apply_augmentations(self.test_img, bad_settings)
        self.assertEqual(len(imgs), 1)

    @patch("cv2.imwrite")
    def test_save_augmented_images_stop_at(self, mock_imwrite):
        from dataset_generator import images_processed, extra_stop
        import dataset_generator

        settings = self.settings.copy()
        settings["stop_at"] = 1

        # Reset globals before test
        dataset_generator.images_processed = 0

        annotations = []
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        save_augmented_images(
            obj_img=self.test_img,
            settings=settings,
            save_path="test_dataset",
            frame_count=0,
            x=0, w=10, y=0, h=10,
            frame=frame,
            class_name="classB",
            annotations=annotations
        )

        self.assertLessEqual(len(annotations), 1)

    def tearDown(self):
        if os.path.exists("test_dataset"):
            import shutil
            shutil.rmtree("test_dataset")

if __name__ == "__main__":
    unittest.main()
