let recorder;
let audioStream;

document.getElementById('recordButton').onclick = async () => {
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new Recorder(new MediaStreamAudioSourceNode(new AudioContext(), { mediaStream: audioStream }));
    recorder.record();

    document.getElementById('recordButton').disabled = true;
    document.getElementById('stopButton').disabled = false;
};

document.getElementById('stopButton').onclick = () => {
    recorder.stop();
    audioStream.getTracks().forEach(track => track.stop());

    recorder.getBuffer(buffer => {
        const audioBlob = new Blob(buffer, { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        document.getElementById('audioPlayback').src = audioUrl;

        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(filename => {
            console.log('Uploaded:', filename);
        });

        document.getElementById('recordButton').disabled = false;
        document.getElementById('stopButton').disabled = true;
    });
};


