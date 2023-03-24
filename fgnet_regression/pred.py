import torch
from torchvision import transforms
from PIL import Image

# load the trained model
model = torch.load('../pre-trained/FGNET_model.pth')

# define transformations for input data
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize input images to 224x224
    transforms.ToTensor(),         # convert image to tensor
    transforms.Normalize(          # normalize input data
        mean=[0.432, 0.359, 0.320],
        std=[0.30,  0.264,  0.252]
    )
])

# load and preprocess input image
img = Image.open('download.jpeg').convert('RGB')
img_tensor = transform(img).unsqueeze(0)

# run prediction
model.eval()
output = model.forward(img_tensor)
# post-process the output
predicted_age = round(float(output[0]), 2)  # round output to 2 decimal places
print(f"The predicted age is: {predicted_age*100}")
