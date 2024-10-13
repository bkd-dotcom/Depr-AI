# Depr-AI
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Depr AI - Depression Detection Model</title>
</head>
<body>

<h1>Depr AI - Depression Detection Model</h1>

<h2>Overview</h2>
<p>Depr AI is a machine learning model designed to detect depression through speech analysis. This project aims to assist in identifying potential depression symptoms by analyzing audio recordings. The model leverages both TensorFlow and Keras for model building, with datasets curated from Kaggle and Hugging Face.</p>

<h2>Features</h2>
<ul>
    <li><strong>Speech-based Depression Detection</strong>: Depr AI analyzes audio recordings of speech to detect symptoms of depression.</li>
    <li><strong>User Interaction</strong>: Upon registration, users are asked 10 questions related to their mood and activities over the past week to help build a context for the detection model.</li>
    <li><strong>Datasets</strong>: The model is trained on the Kaggle Depression Audio Dataset and data from Hugging Face.</li>
</ul>

<h2>Installation</h2>
<p>To get started with Depr AI, clone the repository and install the required dependencies.</p>

<pre><code>git clone https://github.com/yourusername/DeprAI.git
cd DeprAI
pip install -r requirements.txt
</code></pre>

<h2>Requirements</h2>
<ul>
    <li>Python 3.x</li>
    <li>TensorFlow</li>
    <li>Keras</li>
    <li>Pandas</li>
    <li>Numpy</li>
</ul>

<p>To install the dependencies, run:</p>

<pre><code>pip install tensorflow keras pandas numpy
</code></pre>

<h2>Dataset</h2>
<p>We have used datasets from:</p>
<ol>
    <li><a href="https://www.kaggle.com/xyz">Kaggle Depression Audio Dataset</a></li>
    <li>Hugging Face for pretrained models like <code>rafalposwiata/deproberta-large-depression</code>.</li>
</ol>

<p>Make sure you download and preprocess the datasets before running the model.</p>

<h2>Usage</h2>
<ol>
    <li>Clone the repository.</li>
    <li>Preprocess the dataset.</li>
    <li>Train the model using the following command:</li>
</ol>

<pre><code>python train.py
</code></pre>

<ol start="4">
    <li>For inference, provide the audio input file and run:</li>
</ol>

<pre><code>python predict.py --audio_file path_to_audio.wav
</code></pre>

<h2>Model Details</h2>
<p>The model uses advanced NLP and audio feature extraction techniques combined with deep learning. For text-based detection, it uses Hugging Face's <code>rafalposwiata/deproberta-large-depression</code> model. For audio, it leverages features like MFCC (Mel-frequency cepstral coefficients) for input into the neural network.</p>

<h2>User Registration and Interaction</h2>
<p>Upon registration, users are asked to answer 10 questions that assess their mood and mental state over the past week. These responses help provide context for more accurate depression detection based on their speech input.</p>

<h2>Contribution</h2>
<p>We welcome contributions! If you would like to contribute to Depr AI, please follow the steps below:</p>
<ol>
    <li>Fork the repository.</li>
    <li>Create a new branch.</li>
    <li>Make your changes and commit.</li>
    <li>Open a pull request.</li>
</ol>

<h2>License</h2>
<p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>

<h2>Acknowledgements</h2>
<ul>
    <li><a href="https://www.kaggle.com/xyz">Kaggle Depression Audio Dataset</a></li>
    <li>Hugging Face for pretrained models like <code>rafalposwiata/deproberta-large-depression</code>.</li>
</ul>

</body>
</html>
