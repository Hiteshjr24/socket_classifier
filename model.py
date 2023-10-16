import torch
import torchvision.transforms as transforms
from PIL import Image

class ImageClassifier:
    def __init__(self):
        self.model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def classify_image(self, image_matrix):
        input_image = self.transform(image_matrix)
        input_image = input_image.unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_image)
        return output

if __name__ == '__main__':
    classifier = ImageClassifier()
    # You can add code here to test the model standalone
