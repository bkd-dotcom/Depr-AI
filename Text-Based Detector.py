from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
model_name = "rafalposwiata/deproberta-large-depression"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
def detect_depression(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=3000)


    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)


    label = torch.argmax(predictions).item()
    score = predictions[0][label].item()


    return label, score


# Example usage
text = "She sat alone, staring at the empty chair where he once sat. The silence in the room was deafening, echoing the absence that filled her heart. Memories lingered, but his presence was gone, leaving only loneliness and unspoken words behind."
label, score = detect_depression(text)


if label > 0.5:
    print(f"Depressive content detected with confidence: {score:.2f}")
else:
    print(f"No signs of depression detected. Confidence: {score:.2f}")