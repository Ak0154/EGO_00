import uvicorn
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Hugging Face model repo URL (replace with your model URL)
model_repo = "dreameater28/00_EGO_00"

# Load model & tokenizer from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(model_repo)
model = AutoModelForCausalLM.from_pretrained(model_repo, torch_dtype=torch.float16, device_map="auto")

@app.post("/chat")
async def chat(request: dict):
    message = request.get("message")
    if not message:
        return {"response": "No message provided."}

    # Tokenize the message and generate a response
    inputs = tokenizer(message, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True).replace(message, "").strip()

    return {"response": response}

# Needed only for local testing (not for Railway)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
