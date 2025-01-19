import torch
from transformers import AutoModelForCausalLM ,AutoTokenizer , BitsAndBytesConfig
from src.config import settings

def create_llm():
    tokenizer = AutoTokenizer.from_pretrained(
        settings.MODEL_ID,
        token=settings.HF_TOKEN,
    )

    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype = torch.float16,
            bnb_4bit_use_double_quant= True,
            bnb_4bit_quant_type="nf4"
    )
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    model = AutoModelForCausalLM.from_pretrained(
        settings.MODEL_ID,
        token=settings.HF_TOKEN,
        device_map="auto",
        trust_remote_code=True,
        quantization_config= quantization_config
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer, device

def generate_summary(model, tokenizer, text, device, custom_prompt=None):
    try:
        cleaned_text = text.split("Text to summarize:")[-1].strip() if "Text to summarize:" in text else text

        instruction = custom_prompt if custom_prompt else "Summarize the article"
        
        prompt = f"<s>[INST] {instruction}:\n\n{cleaned_text}[/INST]"
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=settings.MAX_LENGTH,
            truncation=True,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                min_length=settings.MIN_LENGTH,
                do_sample=True,
                temperature=settings.TEMPERATURE,
                top_p=settings.TOP_P,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
        
        full_output = tokenizer.decode(outputs[0] , skip_special_tokens=True)

        summary = full_output.split('/[INST]')[-1].strip()

        return summary
    except Exception as e:
        print(f"Error in generate_sumamry: {str(e)}")
        raise Exception(f"Summary Generation Failed: {str(e)}")
    
