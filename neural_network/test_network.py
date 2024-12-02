import unittest
import torch
import os
from pathlib import Path
from predict import predict_image, process_folder
from train_network import NeuralNetwork
from PIL import Image
import numpy as np
import shutil

class TestNeuralNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = Path("test_data")
        cls.test_data_dir.mkdir(exist_ok=True)
        
        cls.test_image = cls.test_data_dir / "test_image.png"
        img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8), 'L')
        img.save(cls.test_image)
        
        if not Path('fashion_model.pth').exists():
            model = NeuralNetwork()
            torch.save(model.state_dict(), 'fashion_model.pth')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_data_dir)
        # Clean up results directory if exists
        results_dir = Path("results")
        if results_dir.exists():
            shutil.rmtree(results_dir)

    def setUp(self):
        # Clean up results before each test
        results_dir = Path("results")
        if results_dir.exists():
            shutil.rmtree(results_dir)

    def test_neural_network_structure(self):
        model = NeuralNetwork()
        self.assertEqual(len(model.linear_relu_stack), 5)
        self.assertEqual(model.linear_relu_stack[0].in_features, 784)
        self.assertEqual(model.linear_relu_stack[0].out_features, 512)
        self.assertEqual(model.linear_relu_stack[2].out_features, 256)
        self.assertEqual(model.linear_relu_stack[4].out_features, 10)

    def test_predict_image(self):
        predicted_class, probabilities = predict_image(self.test_image)
        
        self.assertIsInstance(predicted_class, str)
        self.assertIsInstance(probabilities, torch.Tensor)
        self.assertEqual(probabilities.shape[0], 10)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)

    def test_process_folder(self):
        process_folder(self.test_data_dir, sample_size=1.0)
        results_file = Path("results") / "results.txt"
        
        self.assertTrue(results_file.exists())
        with open(results_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn(self.test_image.name, content)
            self.assertIn("Predicted class", content)

    def test_model_output_shape(self):
        model = NeuralNetwork()
        test_input = torch.randn(1, 1, 28, 28)
        output = model(test_input)
        
        self.assertEqual(output.shape, (1, 10))

    def test_invalid_image_path(self):
        with self.assertRaises(Exception):
            predict_image("nonexistent_image.jpg")

    def test_empty_folder(self):
        empty_dir = self.test_data_dir / "empty"
        empty_dir.mkdir(exist_ok=True)
        
        # Clean up results directory before test
        results_dir = Path("results")
        if results_dir.exists():
            shutil.rmtree(results_dir)
            
        process_folder(empty_dir, sample_size=1.0)
        
        # Check if results directory wasn't created for empty folder
        results_file = Path("results") / "results.txt"
        self.assertFalse(results_file.exists())

    def test_model_save_load(self):
        original_model = NeuralNetwork()
        test_model_path = self.test_data_dir / "test_model.pth"
        
        torch.save(original_model.state_dict(), test_model_path)
        
        loaded_model = NeuralNetwork()
        loaded_model.load_state_dict(torch.load(test_model_path))
        
        for p1, p2 in zip(original_model.parameters(), loaded_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

if __name__ == '__main__':
    unittest.main(verbosity=2)