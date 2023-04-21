'''!pip3 install sentencepiece datasets bitsandbytes accelerate git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git
'''

# Model Loading
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb

model = LlamaForCausalLM.from_pretrained(
    "huggyllama/llama-7b",
    load_in_8bit=True,
    device_map='auto',
)
tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer.pad_token = tokenizer.eos_token

# Post-Processing
for param in model.parameters():
    param.requires_grad = False  # freeze the model - train adapters later
    if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

# Applying LORA
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# Loading the dataset
import transformers
from datasets import load_dataset
data = load_dataset("Amirkid/reddit")
data = data.map(lambda samples: tokenizer(samples['text']), batched=True, remove_columns=['text'])

# Custom Trainer with loss threshold
from transformers import Trainer

class CustomTrainer(Trainer):
    def __init__(self, loss_threshold, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_threshold = loss_threshold

    def on_evaluate(self, args, state, control, **kwargs):
        logs = control.logs[0]  # Get logs of the current evaluation
        val_loss = logs.get("eval_loss")
        if val_loss is not None and val_loss <= self.loss_threshold:
            print(f"\nStopping training: validation loss reached {self.loss_threshold}")
            self.state.global_step = self.state.max_steps  # Set global_step to max_steps to stop training

# Training the model
loss_threshold = 0.7
trainer = CustomTrainer(
    model=model,
    loss_threshold=loss_threshold,
    train_dataset=data['train'],
    eval_dataset=data['validation'],  # Add the validation dataset
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,  # Add evaluation batch size
        gradient_accumulation_steps=4,
        warmup_steps=200,
        max_steps=300,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        evaluation_strategy="steps",  # Add evaluation strategy
        eval_steps=5,  # Add evaluation steps
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# Saving the Model on huggingface
token = "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"
model.push_to_hub("Amirkid/LlamaJokeGeneratorV2", use_auth_token=token)
