'''!pip3 install bitsandbytes datasets accelerate loralib sentencepiece datasets transformers git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git
'''



from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the tokenizer and model
model_name = "Amirkid/LlamaJokeGenerator"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a pipeline for text generation
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Input text
text = "What a beautiful day in london"

# Generate a completion
result = generator(text, max_length=50, do_sample=True, temperature=0.7)

# Print the result
print(result[0]['generated_text'])


