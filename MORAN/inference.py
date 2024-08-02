import torch
from torchvision import transforms
from PIL import Image
from models.moran import MORAN

# Define transforms for preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def load_model(checkpoint_path, device):
    # Initialize the model
    model = MORAN(nc=1, nclass=2, nh=256, targetH=32, targetW=100)
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)
    model.eval()
    return model

def predict(image_path, model, device):
    # Load and preprocess the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image, None, None, None, test=True)  # Test mode
        _, predicted = torch.max(outputs, 1)
    
    # Map class indices to labels
    labels = {0: 'Handwritten', 1: 'Printed'}
    predicted_label = labels[predicted.item()]
    print(f"The image contains: {predicted_label}")

if __name__ == "__main__":
    # Path to the checkpoint and image
    checkpoint_path = './checkpoints/best_model.pth' 
    image_path = './image.jpg' 

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and perform inference
    model = load_model(checkpoint_path, device)
    predict(image_path, model, device)
