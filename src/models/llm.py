import torch
import warnings
import re
from transformers import  AutoModelForSeq2SeqLM,AutoTokenizer 
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger()

def create_llm():
    try:
        warnings.filterwarnings("ignore", message="Some weights of PegasusForConditionalGeneration")

        tokenizer = AutoTokenizer.from_pretrained(
            settings.MODEL_ID,
            trust_remote_code=True,
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForSeq2SeqLM.from_pretrained(
            settings.MODEL_ID,
            torch_dtype=torch.float32,
        ).to(device)

        model.eval()

        logger.info(f"Model device: {next(model.parameters()).device}")

        return model, tokenizer, device

    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")

def format_prompt(text , custom_prompt=None):
    """Format the input text with specific constraints for better summarization"""
    if custom_prompt:
        formatted = (
            f"{custom_prompt}\n\n"
            f"Length Requirements:\n"
            f"- 3 to 4 sentences\n"
            f"- Between 100 and 200 words\n"
            f"- Maintain original writing style\n",
            f"- Include detailed elaboration\n\n"
            f"Article: {text}"
        )
    else:
        formatted = text
    return formatted

def format_summary(text:str) -> str:
    """Clean and format Summary"""

    text = re.sub(r'<n>', '\n', text)
    text = re.sub(r'\.(?=[A-Z])', '. ', text)

    if not text.endswith(('.' , '!' , '?')):
        text += '.'

    return text.strip()
    

def generate_summary(model, tokenizer, text, device, custom_prompt=None):
    try:
        
        prompt = format_prompt(text , custom_prompt)
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=settings.MAX_LENGTH,
            truncation=True,
            padding="longest"
        )

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,      
                min_length=100,       
                num_beams=8,         
                length_penalty=2.0,   
                no_repeat_ngram_size=3, 
                early_stopping=True,
                top_k=50,            
                top_p=0.95,          
                do_sample=True ,
                temperature=0.7,
                repetition_penalty=1.2
            )
        
        summary = tokenizer.decode(outputs[0] , skip_special_tokens=True)

        return format_summary(summary) 

    except Exception as e:
        logger.error(f"Error in generate_sumamry: {str(e)}")
        raise Exception(f"Summary Generation Failed: {str(e)}")
    
