
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from torchvision import transforms
import torch.nn.functional as F
from io import BytesIO
import uvicorn
from pathlib import Path

classes = ['Early_Blight(Ранній фітофтороз)', 'Healthy(Здоровий)', 'Late_Blight(Фітофтороз)']
# PATH = r"C:\Users\despe\app\app\model.pt"

def load_model():

    BASE_DIR = Path(__file__).resolve(strict=True).parent
    
    class LeafDiseaseClassificationModel(nn.Module):
        def __init__(self):
            super(LeafDiseaseClassificationModel, self).__init__()

            # Define the convolutional layers
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

            # Define the fully connected layers
            self.fc1 = nn.Linear(in_features=32 * 56 * 56, out_features=256)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(in_features=256, out_features=3)  # Assuming 10 output classes

        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = x.view(x.size(0), -1)  # Flatten the tensor
            x = self.relu3(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = LeafDiseaseClassificationModel()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(f"{BASE_DIR}/model.pt", "rb") as f:
        model.load_state_dict(torch.load(f, map_location=torch.device(device)))

   
    model.eval()

    return model

def pre_process(image):
    image = Image.open(BytesIO(image))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = transform(image)
    img = img.unsqueeze(0)
    
    return img


def process_output(output):
    probabilities = F.softmax(output, dim=1)
    _, predicted_class = torch.max(probabilities, 1)
    return predicted_class.item()

app = FastAPI()


@app.post("/classify")
async def classify_leaf(file: UploadFile = File(...)):
    
    contents = await file.read()


    model = load_model()

    img = pre_process(contents)

    with torch.no_grad():
        output = model(img)

    result = process_output(output)

    # Повертаємо результат класифікації у форматі JSON
    return JSONResponse({"classification": classes[result]})


@app.get("/")

async def main():

    content =  """
 <!DOCTYPE html>
<html>
<head>
    <title>Upload and Classify Image</title>
    <style>

        body {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        #result-heading {
        font-family: "Helvetica", Arial, sans-serif;
        font-size: 20px;
        font-weight: bold;
        color: #333;
        /* Additional styles for the result heading */
        }
        .container {
            max-width: 500px;
            margin: 0 auto;
            padding: 20px;
            text-align: center;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        .heading {
            font-size: 24px;
            font-weight: bold;
            font-family: Arial, sans-serif;
            color: #333;
            margin-bottom: 20px;
        }
        .preview-image {
            max-width: 300px;
            margin-bottom: 20px;
            margin-top: 40px;
        }
        .load-file-button {
            background-color: green;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 10px;
        }
        .space-between-buttons {
            margin-top: 10px;
        }

        .classification-label-cat {
    margin-top: 10px;
    font-size: 24px;
    font-weight: bold;
    font-family: Arial, sans-serif;
    color: blue;
}

    .classification-label-1 {
        margin-top: 10px;
        font-size: 24px;
        font-weight: bold;
        font-family: Arial, sans-serif;
        color: orange;
    }

    .classification-label-2 {
        margin-top: 10px;
        font-size: 24px;
        font-weight: bold;
        font-family: Arial, sans-serif;
        color: green;
    }

    .classification-label-3 {
        margin-top: 10px;
        font-size: 24px;
        font-weight: bold;
        font-family: Arial, sans-serif;
        color: red;
    }
    
        
        .select-file-button {
            background-color: green;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="heading">Leaf Disease Classification</h1>
        <form action="/classify" method="post" enctype="multipart/form-data">
            <label for="file" class="select-file-button">Вибрати файл</label>
            <input type="file" name="file" id="file" accept="image/jpeg, image/png" required style="display: none;">
            <br>
            <img class="preview-image" id="preview-img" src="#" alt="Preview">
            <div class="space-between-buttons"></div>
            <input type="submit" class="load-file-button" value="Classify">
        </form>

        <div id="result" style="display: none;">
            <h2 id="result-heading">Classification Result:</h2>
            <p id="class-label"></p>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('file');
        const previewImg = document.getElementById('preview-img');
        const resultDiv = document.getElementById('result');
        const classLabel = document.getElementById('class-label');

        fileInput.addEventListener('change', () => {
            const file = fileInput.files[0];
            if (file) {
                previewImg.src = URL.createObjectURL(file);
            } else {
                previewImg.src = '#';
            }
        });

        const form = document.querySelector('form');
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            const formData = new FormData(form);
            
            fetch('/classify', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                const classificationResult = data.classification;
                classLabel.textContent = `Class: ${classificationResult}`;
                // Clear existing class styles
            classLabel.className = '';

            // Apply different styles based on classification result
            if (classificationResult === 'Early_Blight(Ранній фітофтороз)') {
                classLabel.classList.add('classification-label-1');
            } else if (classificationResult === 'Healthy(Здоровий)') {
                classLabel.classList.add('classification-label-2');
            } else if (classificationResult === 'Late_Blight(Фітофтороз)') {
                classLabel.classList.add('classification-label-3');
            } else {
                classLabel.classList.add('classification-label-default');
            }

            resultDiv.style.display = 'block';
        })
        .catch(error => {
            console.error('Error:', error);
        });
});
    </script>
</body>
</html>


    """

    return HTMLResponse(content=content)

def run_server():
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

# if __name__ == "__main__":
#     run_server()
