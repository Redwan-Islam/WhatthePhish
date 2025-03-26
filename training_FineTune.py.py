import os
import json
import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datetime import datetime

class PhishingEmailGenerator:
    def __init__(self, model_path="distilgpt2", base_email_file="base_emails.json"):  # Using smaller model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load or initialize base emails
        self.base_email_file = base_email_file
        if os.path.exists(base_email_file):
            with open(base_email_file, 'r') as f:
                self.base_emails = json.load(f)
        else:
            self.base_emails = {
                "examples": [
                    "Your account has been compromised. Secure it now by clicking here.",
                    "You have an unclaimed reward. Claim it before it expires.",
                    "We detected suspicious activity on your account. Verify your identity here."
                ]
            }
            with open(base_email_file, 'w') as f:
                json.dump(self.base_emails, f)
        
        # Initialize model
        config = GPT2Config.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path, config=config).to(self.device)
        
    def prepare_dataset(self, csv_file):
        # Load with explicit encoding handling
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, encoding='latin-1')
        
        # Use only first 100 samples for testing
        df = df.head(100)
        
        # Combine subject and body with a separator
        texts = [f"Subject: {row['Subject']}\nBody: {row['Body']}" for _, row in df.iterrows()]
        
        # Split into train and validation
        train_texts, val_texts = train_test_split(texts, test_size=0.1)
        
        # Save to temporary files
        with open("train_texts.txt", "w", encoding='utf-8') as f:
            f.write("\n".join(train_texts))
        with open("val_texts.txt", "w", encoding='utf-8') as f:
            f.write("\n".join(val_texts))
            
        train_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path="train_texts.txt",
            block_size=64  # Smaller block size for faster processing
        )
        
        val_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path="val_texts.txt",
            block_size=64
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        return train_dataset, val_dataset, data_collator
    
    def train(self, csv_file, output_dir="phishing_model", iterations=1, start_iter=0):
        # Prepare dataset
        train_dataset, val_dataset, data_collator = self.prepare_dataset(csv_file)
        
        # Training arguments - optimized for quick testing
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=1,  # Just 1 epoch
            max_steps=50,        # Stop after 50 steps (~2-5 minutes)
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            eval_steps=10,       # Evaluate every 10 steps
            save_steps=20,
            warmup_steps=5,
            prediction_loss_only=True,
            logging_dir='./logs',
            logging_steps=5,
            evaluation_strategy="steps",
            load_best_model_at_end=False,  # Disable for quick test
            save_total_limit=1,
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train the model
        print("Starting quick test training (should take ~5 minutes)...")
        trainer.train()
        
        # Save the model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Clean up temporary files
        if os.path.exists("train_texts.txt"):
            os.remove("train_texts.txt")
        if os.path.exists("val_texts.txt"):
            os.remove("val_texts.txt")
            
        # Quick accuracy check
        accuracy = self.evaluate_accuracy(num_samples=2)  # Check just 2 samples
        print(f"Quick test completed. Sample accuracy: {accuracy:.2f}%")
        
        # Save training log
        self.save_training_log(start_iter, iterations, accuracy)
    
    def evaluate_accuracy(self, num_samples=2):  # Reduced samples for quick test
        """Evaluate how well generated emails match base examples"""
        correct = 0
        total = 0
        
        for base_email in self.base_emails["examples"][:num_samples]:  # Only check first few
            prompt = base_email.split(".")[0] + "."
            generated = self.generate_email(prompt=prompt, max_length=50)
            
            if any(keyword in generated.lower() for keyword in ["click", "verify", "secure", "reward"]):
                correct += 1
            total += 1
            
            print(f"Base: {base_email}")
            print(f"Generated: {generated}")
            print("-" * 50)
        
        return (correct / total) * 100 if total > 0 else 0
    
    def generate_email(self, prompt="Your account", max_length=50, temperature=0.7):  # Reduced max_length
        """Generate a phishing email based on prompt"""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
    def save_training_log(self, start_iter, iterations, accuracy):
        """Save training progress to a log file"""
        log_entry = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "start_iteration": start_iter,
            "iterations": iterations,
            "end_iteration": start_iter + iterations,
            "accuracy": accuracy,
            "note": "QUICK TEST RUN"
        }
        
        log_file = "training_log.json"
        logs = []
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        
        logs.append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def load_model(self, model_dir):
        """Load a previously trained model"""
        self.model = GPT2LMHeadModel.from_pretrained(model_dir).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Phishing Email Generator - Quick Test")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file with training data")
    parser.add_argument("--start", type=int, default=0, help="Starting iteration number")
    parser.add_argument("--iter", type=int, default=1, help="Number of iterations (set to 1 for quick test)")
    parser.add_argument("--model", type=str, default="test_model", help="Output model directory")
    parser.add_argument("--load", action="store_true", help="Load existing model before training")
    
    args = parser.parse_args()
    
    print("=== Running in QUICK TEST mode (5-minute training) ===")
    generator = PhishingEmailGenerator()
    
    if args.load and os.path.exists(args.model):
        generator.load_model(args.model)
        print(f"Loaded existing model from {args.model}")
    
    generator.train(
        csv_file=args.csv,
        output_dir=args.model,
        iterations=args.iter,
        start_iter=args.start
    )
    
    print("\nSample generated emails from quick test:")
    for prompt in ["Urgent:", "Your account", "Claim your"]:
        generated = generator.generate_email(prompt=prompt)
        print(f"Prompt: '{prompt}'\nGenerated: {generated}\n")

if __name__ == "__main__":
    main()