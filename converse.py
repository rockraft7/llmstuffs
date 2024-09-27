import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import autocast

# Define the local path to your model
local_model_path = "/workspaces/Llama-3.2-1B-Instruct/"  # Replace with the actual path where your model is saved
model_name = local_model_path.replace('/workspaces/', '').replace('/', '')

# Load the model and tokenizer from the local path
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

# Use GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model_size_in_mb_per_token = 8  # Model size in MB per token
batch_size = 1  # Batch size for generation

# Function to calculate memory usage per token estimate (in MB)
def estimate_memory_usage(prompt_length, max_new_tokens, model_size_in_mb_per_token=8, batch_size=1):
    return (prompt_length + max_new_tokens) * model_size_in_mb_per_token * batch_size

def get_available_memory():
    # Get total GPU memory (in MB)
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    
    # Get currently allocated memory (in MB)
    allocated_memory = torch.cuda.memory_allocated() / (1024 * 1024)
    
    # Clear the cache and get a fresh read
    torch.cuda.empty_cache()

    # Calculate actual available memory based on allocation
    available_memory = total_memory - allocated_memory
    
    # Print debug information
    #print(f"Total Memory: {total_memory} MB")
    #print(f"Allocated Memory (after cache): {allocated_memory} MB")
    #print(f"Available Memory: {available_memory} MB")
    
    return available_memory

# Maintain conversation history to introduce context
conversation_history = []

# Function to generate a response
def generate_response(prompt, max_length=150):
    global conversation_history
    # Add the new prompt to the conversation history
    conversation_history.append(f"User: {prompt}")
    context = " ".join(conversation_history[-3:])  # Include the last 5 exchanges
    
    # Concatenate the conversation history into a single string
    context = " ".join(conversation_history)
    
    # Tokenize the context and calculate the number of tokens
    inputs = tokenizer(context, return_tensors="pt").to(device)
    prompt_length = len(inputs.input_ids[0])
    
    # Estimate the memory usage for the given prompt and max_new_tokens
    estimated_memory = estimate_memory_usage(prompt_length, max_length, model_size_in_mb_per_token, batch_size)
    
    # Get available GPU memory
    available_memory = get_available_memory()

    # Check if the estimated memory exceeds the available GPU memory
    if estimated_memory > available_memory:
        print(f"Warning: Estimated memory usage ({estimated_memory} MB) exceeds available memory ({available_memory} MB). Adjusting...")
        
        # Calculate a new max_new_tokens based on available memory
        max_tokens_that_fit = int(available_memory // (model_size_in_mb_per_token * batch_size))
        max_new_tokens = max(0, max_tokens_that_fit - prompt_length)  # Subtract prompt length
        max_length = min(max_length, max_new_tokens)  # Adjust the length to fit in memory
        
        if max_length == 0:
            print("Error: Not enough memory for even the smallest prompt. Please reduce batch size or try on a larger GPU.")
            max_length = 150
        else:
            print(f"Adjusting max_new_tokens to {max_length} to fit in available memory.")
    else:
        print(f"Estimated memory usage: {estimated_memory} MB (Available memory: {available_memory} MB)")

    attention_mask = inputs['attention_mask'].to(device)
    
    # Using autocast for mixed-precision if on GPU
    with autocast(device_type="cuda"):
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=attention_mask, 
            max_new_tokens=max_length,
            do_sample=True,           # Enable sampling for more human-like responses
            temperature=0.7,          # Adjust temperature for diversity in responses
            top_p=0.85,                # Top-p sampling for more natural conversations
            pad_token_id=tokenizer.eos_token_id  # Ensure padding uses end-of-sequence token
        )
    
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Add the response to the conversation history
    conversation_history.append(f"Chatbot: {response}")

    torch.cuda.empty_cache()

    return response

# Chatbot loop
def chat():
    global conversation_history
    print("Chatbot: Hello! I'm " + model_name + ". How can I assist you today?")
    
    while True:
        # Get user input
        user_input = input("You: ")

        # Exit condition
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        if user_input.lower() in ["clear", "reset", "start over"]:
            print("Chatbot: Conversation history cleared.")
            conversation_history = []
            continue

        # Generate and print the response
        response = generate_response(user_input)
        print(f"Chatbot: {response}")

# Start the chat
if __name__ == "__main__":
    chat()
