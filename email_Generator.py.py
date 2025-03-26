import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import uuid
import re

class PhishingEmailGenerator:
    def __init__(self, model_dir="test_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_dir}...")
        
        # Load model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(model_dir).to(self.device)
        
        # Watermark configuration
        self.watermark_prefix = "<!-- SIMULATED_"
        self.watermark_suffix = " -->"
        self.watermark_enabled = True
        
        # Fix tokenizer warnings
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Model loaded successfully!")

    def _generate_watermark(self):
        """Create a unique invisible watermark"""
        return f"{self.watermark_prefix}{uuid.uuid4()}{self.watermark_suffix}"

    def _embed_watermark(self, text):
        """Insert watermark in the middle of text"""
        if not self.watermark_enabled:
            return text
            
        watermark = self._generate_watermark()
        words = text.split()
        insert_pos = len(words) // 2  # Middle position
        words.insert(insert_pos, watermark)
        return ' '.join(words)

    def generate_email(self, prompt="Urgent:", max_length=150, temperature=0.85):
        """Generate watermarked phishing email"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate with proper attention mask
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.92,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._embed_watermark(raw_output)

    def verify_watermark(self, text):
        """Check if text contains valid watermark"""
        pattern = re.compile(
            re.escape(self.watermark_prefix) + 
            r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}' +
            re.escape(self.watermark_suffix)
        )
        return bool(pattern.search(text))

def main():
    # Initialize generator with your trained model
    generator = PhishingEmailGenerator(model_dir="test_model")
    
    # Example generation
    prompts = [
        "URGENT: Your account"
    ]
    
    print("\n=== Generating Watermarked Emails ===")
    for prompt in prompts:
        email = generator.generate_email(prompt)
        
        print(f"\nPrompt: {prompt}")
        print("-"*60)
        print(email)
        print("-"*60)
        print(f"Watermark Verified: {generator.verify_watermark(email)}")
        print("="*60)

if __name__ == "__main__":
    main()