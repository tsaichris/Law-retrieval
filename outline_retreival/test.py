import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Login to Hugging Face (if needed)
from huggingface_hub import login
login(token="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")

# Load model and tokenizer
model_id = "lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)
# Set model to evaluation mode
model.eval()

def generate_response(prompt, max_length=2048):
    # Encode the input text
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.2,  # Add penalty for repetition
            no_repeat_ngram_size=3,  # Prevent repeating 3-grams
            pad_token_id=tokenizer.pad_token_id,
            top_p=1.0
        )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
prompt = """帶手指虎打傷人，手指虎會被「沒收」還是「沒入」？ [SEP] 知道沒收是刑法，沒入是行政法。 單純持有違禁品（手指虎）會遭到沒收， 但用違禁品傷人，是會被「沒收」還是「沒入」呢? 請告訴我有哪些法條是相關的:
"""
response = generate_response(prompt)
print(response)