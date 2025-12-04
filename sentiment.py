# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base")
# model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
# model.eval()

# # Long dummy text to allow dynamic shapes
# dummy_text = "This is a long dummy input for multilingual ONNX export."
# dummy = tokenizer(dummy_text, return_tensors="pt")

# torch.onnx.export(
#     model,
#     (dummy["input_ids"], dummy["attention_mask"]),
#     "xlm_multilingual.onnx",
#     input_names=["input_ids", "attention_mask"],
#     output_names=["logits"],
#     dynamic_axes={
#         "input_ids": {0: "batch", 1: "sequence"},
#         "attention_mask": {0: "batch", 1: "sequence"},
#     },
#     opset_version=18
# )


import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base")

session = ort.InferenceSession("models/xlm_multilingual.onnx", providers=["CPUExecutionProvider"])

labels=["negative","neutral","positive"]
count=np.array([0,0,0],dtype=float)
total_comments=0

def decode(logits):
    global count,total_comments

    for logit in logits:
       count[np.argmax(logit)]+=1
       total_comments+=1

    percentage=(count/total_comments)*100  
    return percentage

def predict_batch(comments):
    # Tokenize all comments together (batch)
    inputs = tokenizer(comments, return_tensors="np", padding=True, truncation=True)

    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }

    # Run ONNX inference
    logits = session.run(None, ort_inputs)[0]
    senti=decode(logits)
    return senti



