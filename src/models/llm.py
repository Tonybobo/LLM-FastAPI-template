import torch
import warnings
from transformers import  AutoModelForSeq2SeqLM,AutoTokenizer 
from src.config import settings

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

        print(f"Model device: {next(model.parameters()).device}")

        return model, tokenizer, device

    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        raise

def generate_summary(model, tokenizer, text, device, custom_prompt=None):
    try:
        
        prompt = f"{custom_prompt}:{text}" if custom_prompt else text 
        
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
                max_length=128,      
                min_length=30,       
                num_beams=8,         
                length_penalty=1.2,   
                no_repeat_ngram_size=3, 
                early_stopping=True,
                top_k=50,            
                top_p=0.95,          
                do_sample=True 
            )
        
        summary = tokenizer.decode(outputs[0] , skip_special_tokens=True)

        return summary 

    except Exception as e:
        print(f"Error in generate_sumamry: {str(e)}")
        raise Exception(f"Summary Generation Failed: {str(e)}")
    
