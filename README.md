Phishing Email Generator
This repository contains tools for generating and training AI models to create phishing emails for educational and research purposes. The system uses GPT-2 language model fine-tuned on phishing email datasets.

Contents
Email Generator: A script to generate watermarked phishing emails using a trained model

Training Script: A script to fine-tune GPT-2 on phishing email datasets

Sample Dataset: Example structure of the training data (shown below)

Dataset Structure
The training dataset should be a CSV file with the following structure:

Copy
Subject,Body
Exciting Job Opportunity for You!,Your account has been compromised. Secure it now by clicking here.
Important Security Alert,You have an unclaimed reward. Claim it before it expires.
Update Your Payment Information,"You have been selected as the winner of a $10,000 prize! Claim your reward now."
Your Account Has Been Compromised,"You have been selected as the winner of a $10,000 prize! Claim your reward now."
Congratulations! You've Won a Prize,We have detected unusual activity on your account. Please verify your account immediately.
Important Security Alert,Your account has been compromised. Secure it now by clicking here.
Installation
Clone this repository:

bash
Copy
git clone [repository-url]
cd [repository-name]
Install required Python packages:

bash
Copy
pip install torch transformers pandas scikit-learn
Download the base GPT-2 model (will be done automatically on first run)

Usage
Training the Model
To train the model with your dataset:

bash
Copy
python training_FineTune.py --csv your_dataset.csv --model output_model_dir
Options:

--csv: Path to your training CSV file

--model: Output directory for the trained model

--iter: Number of training iterations (default: 1)

--load: Load existing model before training

Generating Emails
To generate phishing emails using a trained model:

bash
Copy
python email_Generator.py
The generator will:

Load the model from test_model directory (or your specified directory)

Generate sample emails with invisible watermarks

Verify the watermarks are properly embedded

Features
Watermarking: All generated emails contain invisible watermarks for identification

Quick Training: Optimized for rapid testing with reduced dataset and training steps

Model Evaluation: Includes basic accuracy checking against known phishing patterns

Device Optimization: Automatically uses GPU if available

Important Notes
This tool is intended for educational and research purposes only

Generated emails contain visible watermarks indicating they are simulated

Always use responsibly and ethically

The sample training uses a very small subset of data for quick testing

Output Example
Sample generated email with watermark:

Copy
URGENT: Your account <!-- SIMULATED_550e8400-e29b-41d4-a716-446655440000 --> has been locked due to suspicious activity. Click here to verify your identity and restore access immediately.
The watermark can be verified using the built-in verification function.
