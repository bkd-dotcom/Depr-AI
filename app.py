import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Load pre-trained Wav2Vec2 model and processor
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
model.eval()

# Simple classifier
class SimpleClassifier(torch.nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(32, 256)
        self.fc2 = torch.nn.Linear(256, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize classifier
classifier = SimpleClassifier()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file found"}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if audio_file:
        filename = os.path.join('uploads', audio_file.filename)
        audio_file.save(filename)
        
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(filename)
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values

        # Get predictions from the model
        with torch.no_grad():
            logits = model(input_values).logits
        
        # Use the classifier
        features = logits.mean(dim=1)
        output = classifier(features)
        predicted_class = torch.argmax(output).item()

        # Interpret prediction
        emotion = "Depressed" if predicted_class == 1 else "Not Depressed"
        
        # Clean up the uploaded file
        os.remove(filename)
        
        return jsonify({"result": emotion})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)