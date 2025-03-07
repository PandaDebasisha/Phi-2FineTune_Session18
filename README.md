# Phi-2 Fine-tuned - With Oasst1 dataset with nf4 quantization

This repository contains code for fine-tuning the Microsoft Phi-2 model using QLoRA (Quantized Low-Rank Adaptation) on the OpenAssistant dataset.

## Model Details

- Base Model: [Microsoft/Phi-2](https://huggingface.co/microsoft/phi-2)
- Dataset: [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)
- Fine-tuning Method: QLoRA (4-bit quantization with LoRA)
- Hugging Face Model: [debasisha/phi2-finetuned](https://huggingface.co/debasisha/phi2-finetuned)

## Training Configuration
Fine tuning Phi2 Hugging face 
### Model Quantization
python
bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.float16,
bnb_4bit_use_double_quant=True,
)

### LoRA Configuration
python
lora_config = LoraConfig(
r=16, # LoRA rank
lora_alpha=32, # LoRA alpha
target_modules=["q_proj", "k_proj", "v_proj", "dense"],
lora_dropout=0.05,
bias="none",
task_type="CAUSAL_LM"
)

### Training Parameters
- Number of epochs: 3
- Batch size: 4
- Gradient accumulation steps: 4
- Learning rate: 2e-4
- FP16 training: Enabled
- Warmup ratio: 0.03
- Evaluation strategy: Steps
- Evaluation steps: 100
- Save steps: 100
- Logging steps: 10

## Dataset Processing

The training data is processed in the following way:
1. Filters for prompter messages from the OASST1 dataset
2. Matches prompts with corresponding assistant responses
3. Formats conversations into instruction-response pairs
4. Tokenizes with a maximum length of 512 tokens
5. Splits into train and validation sets (90/10 split)

## Model Architecture

The model uses:
- Base architecture: Phi-2
- Quantization: 4-bit precision
- LoRA adapters on attention and dense layers
- Automatic device mapping for memory optimization

## Usage

### Installation
pip install -r requirements.txt

### Training
bash
python train.py


## Model Checkpointing

The training process saves:
- Checkpoints every 100 steps
- Best model based on evaluation loss
- Final model after training completion
- Keeps last 3 checkpoints for space efficiency

## Logging

Detailed training logs include:
- Training metrics at each logging step
- Evaluation results
- Loss tracking
- Sample generations every 100 steps
- Best model checkpoints
- Training duration and final statistics

## License

This project is licensed under the MIT License.

## Acknowledgments

- Microsoft for the Phi-2 base model
- OpenAssistant for the training dataset
- Hugging Face for the transformers library and model hosting
