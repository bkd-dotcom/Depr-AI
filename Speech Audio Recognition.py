import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load pre-trained Wav2Vec2 model and processor
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
model.eval()

# Function to load and preprocess audio
def load_audio(file_path):
    audio_input, _ = torchaudio.load(file_path)
    return audio_input.squeeze()

# Example audio file path
audio_file_path = "/content/Speech-Emotion-Analyzer_output10.wav"  # Replace with your audio file path

# Load and preprocess audio
audio = load_audio(audio_file_path)
input_values = processor(audio.numpy(), return_tensors="pt", sampling_rate=16000).input_values

# Get predictions from the model
with torch.no_grad():
    logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(32, 256) 
        self.fc2 = nn.Linear(256, 2)     

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize classifier
classifier = SimpleClassifier()

# Example for forward pass with logits
features = logits.mean(dim=1)  
output = classifier(features)
predicted_class = torch.argmax(output).item() # Remove dim=0

# Interpret prediction and print result
if predicted_class == 0:
    emotion = "Not Depressed"
else:
    emotion = "Depressed"
print(f"Predicted Emotion: {emotion}")