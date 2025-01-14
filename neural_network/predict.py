import torch
import torchvision.transforms as transforms
from PIL import Image
from model import NeuralNetwork
import os
from pathlib import Path
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        raise
    
    image = transform(image)
    image = image.unsqueeze(0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNetwork().to(device)
    
    model_path = Path('fashion_model.pth')
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path.absolute()}")
    
    try:
        model.load_state_dict(torch.load('fashion_model.pth'))
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
        
    model.eval()
    
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        return classes[predicted[0]], probabilities[0]

def display_predictions(image_path, predicted_class, top_predictions):
    img = Image.open(image_path)
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    
    pred_text = f"Top predictions:\n"
    for class_name, prob in top_predictions[:3]:
        pred_text += f"{class_name}: {prob:.2%}\n"
    
    plt.title(pred_text, pad=20)
    plt.show()

def process_folder(folder_path, sample_size=0.1):
    print("\n=== Starting Processing ===")
    
    image_files = [
        f for f in os.listdir(folder_path) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ]
    
    if not image_files:
        print(f"Warning: No images found in folder {folder_path}!")
        return
    
    # Create results directory first
    results_folder = Path("results")
    results_folder.mkdir(exist_ok=True)
    
    sample_size = int(len(image_files) * sample_size)
    selected_files = random.sample(image_files, sample_size)
    
    print(f"Found {len(image_files)} images")
    print(f"Randomly selected {sample_size} images to process")
    print("\nStarting image analysis...")
    
    results = []
    
    progress_bar = tqdm(
        selected_files,
        desc="Progress",
        unit="image",
        ncols=70,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
    )
    
    # Create and open results file
    with open(results_folder / "results.txt", "w", encoding='utf-8') as f:
        for filename in progress_bar:
            try:
                image_path = os.path.join(folder_path, filename)
                predicted_class, probabilities = predict_image(image_path)
                
                probs_sorted = sorted(
                    [(classes[i], prob.item()) for i, prob in enumerate(probabilities)],
                    key=lambda x: x[1],
                    reverse=True
                )
                
                results.append({
                    'path': image_path,
                    'filename': filename,
                    'predicted_class': predicted_class,
                    'probabilities': probs_sorted
                })
                
                # Write results to file immediately
                f.write(f"\nImage: {filename}\n")
                f.write(f"Predicted class: {predicted_class}\n")
                f.write("Probabilities:\n")
                for class_name, prob in probs_sorted:
                    f.write(f"{class_name}: {prob:.2%}\n")
                f.write("-" * 50 + "\n")
                f.flush()  # Ensure writing to file immediately
                
            except Exception as e:
                progress_bar.write(f"Skipped {filename}: {str(e)}")
                continue
    
    print("\n=== Analysis Complete ===")
    print(f"Successfully analyzed {len(results)} images")
    print("\nDisplaying results...")
    print("(Press Enter to see next image)")
    print("-" * 50)
    
    for i, result in enumerate(results, 1):
        print(f"\nImage {i} of {len(results)}: {result['filename']}")
        display_predictions(
            result['path'],
            result['predicted_class'],
            result['probabilities']
        )
        if i < len(results):
            input("Press Enter to see next image...")
        plt.close()

if __name__ == "__main__":
    try:
        folder_path = Path("C:/Users/micha/.cache/kagglehub/datasets/paramaggarwal/fashion-product-images-dataset/versions/1/fashion-dataset/images")
        if not folder_path.exists():
            print(f"Error: Folder {folder_path.absolute()} does not exist!")
        else:
            process_folder(folder_path, sample_size=0.01)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        traceback.print_exc()
    
    print("\n=== Program Finished ===")
    input("Press Enter to close...")