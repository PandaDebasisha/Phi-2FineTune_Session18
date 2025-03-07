import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Add to requirements.txt:
# gradio>=4.0.0

# Create offload directory
offload_dir = "./offload"
os.makedirs(offload_dir, exist_ok=True)

def load_model():
    model_id = "debasisha/phi2-finetuned"  # replace with your model repository
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        offload_folder=offload_dir  # Add offload directory
    )
    return model, tokenizer

def generate_response(prompt, temperature=0.7, max_length=512):
    # Format the prompt
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:"
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and return
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    response_parts = response.split("### Response:")
    if len(response_parts) > 1:
        return response_parts[1].strip()
    return response.strip()

# Load model and tokenizer
print("Loading model...")
model, tokenizer = load_model()
print("Model loaded!")

# Create Gradio interface
demo = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(label="Enter your prompt", lines=4),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, label="Temperature"),
        gr.Slider(minimum=64, maximum=2048, value=512, step=64, label="Max Length"),
    ],
    outputs=gr.Textbox(label="Generated Response", lines=8),
    title="Phi-2 Fine-tuned Assistant",
    description="Enter your prompt and the model will generate a response.",
    examples=[
        ["Explain what is machine learning in simple terms."],
        ["Write a short poem about a sunset."],
        ["What are the main differences between Python and JavaScript?"],
    ]
)

# Launch the app
if __name__ == "__main__":
    demo.launch() 