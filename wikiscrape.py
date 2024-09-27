import wikipediaapi, re, torch, shutil
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
from accelerate import Accelerator
from torch.utils.data import DataLoader

# Clear previous datasets and models if they exist
print("Deleting old dataset and model directories...")
shutil.rmtree('/workspaces/ml-stuffs/malaysia_history_dataset', ignore_errors=True)
shutil.rmtree('/workspaces/ml-stuffs/fine_tuned_llama', ignore_errors=True)

# Step 1: Extract Wikipedia articles with a custom User-Agent
user_agent = "ChatApp/1.0 (author@chatapp.com)"  # Customize your user agent
wiki_wiki = wikipediaapi.Wikipedia(language='en', user_agent=user_agent)
#pages = ["History of Malaysia", "Malaysian independence", "Geography of Malaysia", "Politics of Malaysia"]
pages = ["Anwar Ibrahim"]
def extract_wikipedia_page(page_title):
    print(f"Extracting page: {page_title}")
    page = wiki_wiki.page(page_title)
    return page.text if page.exists() else None

# Extract and store the text
print("Extracting Wikipedia pages...")
malaysia_text = {page: extract_wikipedia_page(page) for page in pages}

# Step 2: Clean the text
print("Cleaning the extracted text...")
def clean_text(text):
    text = re.sub(r"\[\d+\]", "", text)  # Remove references like [1], [2]
    text = re.sub(r"\n+", "\n", text)    # Replace multiple newlines with a single newline
    return text.strip()

malaysia_text = {title: clean_text(text) for title, text in malaysia_text.items() if text}

# Combine cleaned text
print("Combining cleaned text...")
combined_text = "\n\n".join(malaysia_text.values())
paragraphs = combined_text.split("\n\n")

# Step 3: Create the dataset
print("Creating the dataset...")
data_dict = {"text": paragraphs}
wikipedia_dataset = Dataset.from_dict(data_dict)
wikipedia_dataset.save_to_disk("malaysia_history_dataset")
print("Dataset created and saved to disk.")

# Step 4: Load dataset, model, and tokenizer for fine-tuning
print("Loading dataset and initializing model and tokenizer...")
dataset = Dataset.load_from_disk("malaysia_history_dataset")
tokenizer = AutoTokenizer.from_pretrained("/workspaces/Llama-3.2-1B-Instruct/")
model = AutoModelForCausalLM.from_pretrained("/workspaces/Llama-3.2-1B-Instruct/", use_cache=False)
model.gradient_checkpointing_enable()  # Enable gradient checkpointing

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

# Tokenization function that includes labels for causal language modeling
def tokenize_function(examples):
    print("Tokenizing dataset...")
    tokens = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
    
    # Create labels, set padding tokens to -100 so they are ignored in the loss calculation
    tokens["labels"] = tokens["input_ids"].copy()
    tokens["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in tokens["labels"]]
    
    return tokens

print("Applying tokenization to dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
print("Tokenization complete.")

# Step 5: Fine-tuning with Accelerate
def enable_accelerator(tokenized_dataset, model):
    print("Initializing Accelerator for training...")
    accelerator = Accelerator()

    # Prepare DataLoader for training (Data remains on CPU and will be moved to GPU in batches)
    data_loader = DataLoader(tokenized_dataset, batch_size=1, shuffle=True, num_workers=4)

    # Prepare model and optimizer for GPU training
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Use Accelerator to handle device placement
    return accelerator.prepare(model, optimizer, data_loader)

model, optimizer, data_loader = enable_accelerator(tokenized_dataset=tokenized_dataset, model=model)
print("Accelerator preparation complete. Model on:", model.device)


# Training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_llama",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,  # Reduced batch size
    per_device_eval_batch_size=4,   # Reduced eval batch size
    num_train_epochs=3,
    gradient_accumulation_steps=8,  # Gradient accumulation
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True  # Mixed precision to reduce memory usage
)

# Memory management callback
from transformers import TrainerCallback

class MemoryCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

# Initialize Trainer
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    optimizers=(optimizer, None),  # Optimizer is prepared with accelerate
    callbacks=[MemoryCallback()]   # Add memory clearing callback
)
print("Trainer initialized.")

# Step 6: Start fine-tuning
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete.")

# Step 7: Save the fine-tuned model
print("Saving the fine-tuned model and tokenizer...")
model.save_pretrained("./fine_tuned_llama")
tokenizer.save_pretrained("./fine_tuned_llama")
print("Model and tokenizer saved.")

# Step 8: Generate text using the fine-tuned model
def generate_text(prompt, model_path="./fine_tuned_llama"):
    print("Generating text from fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    result = text_generator(prompt, max_length=100)
    print("Text generation result:")
    print(result)

generate_text("Who is the current deputy prime minister of Malaysia.")
