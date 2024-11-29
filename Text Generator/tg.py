import tkinter as tk
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time

# Load pre-trained GPT-2 model and tokenizer from HuggingFace
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the model to evaluation mode (no training)
model.eval()

# Function to generate text using GPT-2
def generate_text(seed_text, max_length=200):
    # Encode the seed text to tensor format
    input_ids = tokenizer.encode(seed_text, return_tensors="pt")

    # Generate text from the model
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)

    # Decode the generated text to readable string
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Create the Tkinter application window
class TextGeneratorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Text Generator using GPT-2")

        # Create the UI elements
        self.heading_label = tk.Label(self.master, text="Text Generator\nUsing GPT-2 Model", font=("Helvetica", 16, "bold"))
        self.heading_label.pack(pady=10)

        self.input_label = tk.Label(self.master, text="Enter seed text:")
        self.input_label.pack(pady=5)

        self.entry = tk.Entry(self.master, width=50)
        self.entry.pack(pady=10)

        self.generate_button = tk.Button(self.master, text="Generate Text", command=self.generate_text)
        self.generate_button.pack(pady=10)

        self.output_label = tk.Label(self.master, text="Generated Text:", font=("Helvetica", 14))
        self.output_label.pack(pady=10)

        self.output_text = tk.Text(self.master, height=10, width=50)
        self.output_text.pack(pady=10)

    def generate_text(self):
        seed_text = self.entry.get()

        if seed_text.strip() == "":
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Please enter a seed text!")
            return

        # Display a "Processing..." message in the terminal
        print("Processing... Please wait.")
        
        # Show processing window in the GUI while generating text
        self.show_processing_window()

        # Generate the text using GPT-2
        generated_text = generate_text(seed_text, max_length=100)

        # Close the processing window after generation is done
        self.processing_window.destroy()

        # Display the generated text in the output text box
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, generated_text)

    def show_processing_window(self):
        # Create a separate window for processing alert
        self.processing_window = tk.Toplevel(self.master)
        self.processing_window.title("Processing")
        label = tk.Label(self.processing_window, text="Processing... Please wait", font=("Helvetica", 14))
        label.pack(pady=50)
        self.processing_window.geometry("300x100")
        self.processing_window.protocol("WM_DELETE_WINDOW", self.on_processing_close)

    def on_processing_close(self):
        # Prevent the user from closing the processing window
        pass

# Run the Tkinter application
if __name__ == "__main__":
    # Display a processing message in the terminal before starting the Tkinter window
    print("Processing... Please wait.")
    
    root = tk.Tk()
    app = TextGeneratorApp(root)
    root.mainloop()





# Installation steps:
# 1. Install the required libraries:
# pip install transformers torch

# 2. Upgrade pip to avoid potential issues:
# python -m pip install --upgrade pip

# 3. Install additional PyTorch libraries (optional, for other uses):
# pip install torch torchvision torchaudio

# Verify installation by checking for installed packages:
# pip show transformers
# pip show torch
# pip show tk     