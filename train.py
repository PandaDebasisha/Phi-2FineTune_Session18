import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import os
import logging
from datetime import datetime
import json
from pathlib import Path
import sys

# Set up logging with more detailed configuration
def setup_logging(log_dir="./logs"):
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/training_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Create a custom handler to redirect stdout and stderr to the log file
    class PrintLogHandler:
        def __init__(self, logger):
            self.logger = logger

        def write(self, message):
            if message.strip() and not message.isspace():
                self.logger.info(message.strip())

        def flush(self):
            pass

    logger = logging.getLogger(__name__)
    
    # Redirect stdout and stderr to the log file
    sys.stdout = PrintLogHandler(logger)
    sys.stderr = PrintLogHandler(logger)
    
    return logger

logger = setup_logging()

class DetailedTrainingCallback(TrainerCallback):
    """Callback for detailed training logging and sample generation"""
    def __init__(self, tokenizer, sample_prompts):
        self.tokenizer = tokenizer
        self.sample_prompts = sample_prompts
        self.best_loss = float('inf')
        self.training_start_time = None
        self.step_loss_accumulator = []
        self.current_epoch = 0
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.training_start_time = datetime.now()
        logger.info("=== Training Started ===")
        logger.info(f"Model Parameters: {model.num_parameters():,}")
        logger.info(f"Trainable Parameters: {model.num_parameters(only_trainable=True):,}")
        logger.info("Training Configuration:")
        logger.info(f"Number of epochs: {args.num_train_epochs}")
        logger.info(f"Batch size: {args.per_device_train_batch_size}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Weight decay: {args.weight_decay}")
        logger.info(f"Warmup steps: {args.warmup_steps}")
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            if hasattr(state, 'loss_step') and state.loss_step is not None:
                self.step_loss_accumulator.append(state.loss_step)
                logger.info(f"Step {state.global_step}: Current loss = {state.loss_step:.4f}")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch += 1
        logger.info(f"\n=== Starting Epoch {self.current_epoch}/{args.num_train_epochs} ===")
        self.step_loss_accumulator = []
        
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.step_loss_accumulator:
            avg_loss = sum(self.step_loss_accumulator) / len(self.step_loss_accumulator)
            logger.info(f"=== Epoch {self.current_epoch} Summary ===")
            logger.info(f"Average Loss: {avg_loss:.4f}")
            logger.info(f"Learning Rate: {state.learning_rate}")
            logger.info(f"Steps Completed: {state.global_step}")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        
        # Log training metrics
        step_metrics = {
            "step": state.global_step,
            "epoch": state.epoch,
            "learning_rate": logs.get("learning_rate", 0),
            "train_loss": logs.get("loss", 0),
            "eval_loss": logs.get("eval_loss", None)
        }
        
        logger.info(f"Training Metrics at step {state.global_step}:")
        logger.info(json.dumps(step_metrics, indent=2))
        
        # Track best loss
        if "eval_loss" in logs:
            eval_loss = logs["eval_loss"]
            if eval_loss < self.best_loss:
                self.best_loss = eval_loss
                logger.info(f"New best eval loss: {eval_loss:.4f}")
                logger.info(f"Saving best model at step {state.global_step}")
        
        # Generate samples every 100 steps
        if state.global_step % 100 == 0:
            logger.info(f"\n=== Generation samples at step {state.global_step} ===")
            model = kwargs['model']
            model.eval()
            
            for prompt in self.sample_prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                )
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"\nPrompt: {prompt}\nGenerated: {generated_text}\n")
            
            model.train()
    
    def on_train_end(self, args, state, control, **kwargs):
        training_duration = datetime.now() - self.training_start_time
        logger.info("\n=== Training Completed ===")
        logger.info(f"Total training time: {training_duration}")
        logger.info(f"Best eval loss: {self.best_loss:.4f}")
        logger.info(f"Total steps: {state.global_step}")
        logger.info(f"Final learning rate: {state.learning_rate}")
        logger.info("=== Final Training Summary ===")
        logger.info(f"Total Epochs Completed: {self.current_epoch}")
        if hasattr(state, 'best_model_checkpoint'):
            logger.info(f"Best Model Checkpoint: {state.best_model_checkpoint}")

# Configure BitsAndBytes for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Initialize model and tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=16,  # rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "dense"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA adapter
model = get_peft_model(model, lora_config)
logger.info("Trainable parameters info:")
model.print_trainable_parameters()  # This will now be captured in the log file

# Load and preprocess dataset
dataset = load_dataset("OpenAssistant/oasst1", split="train")  # Explicitly load the train split

def format_conversation(example, dataset_dict):
    """Format the conversation into instruction-response format"""
    if example['role'] == 'assistant':
        return None  # Skip assistant messages as initial prompts
    
    # Find the corresponding response
    response = None
    parent_id = example['parent_id']
    
    # More efficient way to find the response
    if parent_id:
        matching_responses = [
            msg['text'] for msg in dataset_dict 
            if msg['message_id'] == parent_id and msg['role'] == 'assistant'
        ]
        if matching_responses:
            response = matching_responses[0]
    
    if response is None:
        return None
        
    formatted_text = f"### Instruction:\n{example['text']}\n\n### Response:\n{response}"
    return formatted_text

# Process and tokenize the dataset
def tokenize_function(examples):
    model_inputs = tokenizer(
        examples,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    # Create labels for causal language modeling
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# Convert dataset to list for easier processing
dataset_dict = list(dataset)

# Process the dataset
processed_dataset = dataset.filter(lambda x: x['role'] == 'prompter')
processed_dataset = processed_dataset.map(
    lambda x: {'text': format_conversation(x, dataset_dict)}
)
processed_dataset = processed_dataset.filter(lambda x: x['text'] is not None)
processed_dataset = processed_dataset.remove_columns(
    [col for col in processed_dataset.column_names if col != 'text']
)

# Split the dataset into train and validation
train_val_dataset = processed_dataset.train_test_split(test_size=0.1)

# Tokenize the datasets
tokenized_dataset = train_val_dataset.map(
    lambda x: tokenize_function(x['text']),
    batched=True,
    remove_columns=['text']
)

# Sample prompts for generation
sample_prompts = [
    "### Instruction:\nExplain what is machine learning in simple terms.\n\n### Response:",
    "### Instruction:\nWrite a short poem about a sunset.\n\n### Response:"
]

# Check for existing checkpoint
checkpoint_dir = "./phi2-finetuned"
resume_from_checkpoint = None

if os.path.exists(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint-')]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
        resume_from_checkpoint = os.path.join(checkpoint_dir, latest_checkpoint)
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")

# Training arguments
training_args = TrainingArguments(
    output_dir=checkpoint_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    save_steps=100,
    logging_steps=10,
    load_best_model_at_end=True,
    warmup_ratio=0.03,
    group_by_length=True,
    evaluation_strategy="steps",
    eval_steps=100,
    save_total_limit=3,  # Keep only the last 3 checkpoints
    logging_dir="./logs",  # Directory for tensorboard logs
)

# Initialize trainer with enhanced callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    callbacks=[DetailedTrainingCallback(tokenizer, sample_prompts)]
)

# Enhanced error handling and logging
try:
    logger.info("Starting training with configuration:")
    logger.info(f"Training Arguments: {training_args}")
    logger.info(f"Dataset size - Train: {len(tokenized_dataset['train'])}, Test: {len(tokenized_dataset['test'])}")
    
    if resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    logger.info("Training completed successfully!")

except Exception as e:
    logger.error(f"Training interrupted with error: {str(e)}", exc_info=True)
    logger.info("You can resume training from the last checkpoint")
    raise

finally:
    # Log final model save attempt
    try:
        trainer.save_model("./phi2-finetuned-final")
        logger.info("Final model saved successfully!")
    except Exception as e:
        logger.error(f"Error saving final model: {str(e)}", exc_info=True) 