import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from src.config import settings

def create_llm():
    try:
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        print(f"Model loaded successfully on {device}")
        return model, tokenizer, device
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def generate_summary(model, tokenizer, text: str, device: str, custom_prompt: str = None) -> str:
    """Generate summary using BART model"""
    try:
        # Prepare the input text
        if custom_prompt:
            text = f"{custom_prompt}\n\n{text}"
        
        # Tokenize
        inputs = tokenizer(
            text,
            max_length=1024,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        
        # Move inputs to the same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=40,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        raise