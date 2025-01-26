import torch
import warnings
import re
import os
from transformers import BartForConditionalGeneration, BartTokenizer 
from src.utils.config import settings
from src.utils.logger import Logger 
from src.utils.s3_handler import S3Handler

class BartModelManager:

    _instance = None
    _initialized = False

    def __new__(cls , logger = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self , logger: Logger):
        if not self._initialized:
            self.logger = logger
            self.s3 = S3Handler(logger)
            self.model_dir = settings.MODEL_LOCAL_DIR
            self.model = None 
            self.tokenizer = None
            self._initialized = True
    
    @classmethod
    def get_instance(cls , logger=None):
        if cls._instance is None:
            cls._instance = BartModelManager(logger)
        return cls._instance
    

    def sync_model_with_s3(self) -> bool:
        """Sync model with S3: upload if not in S3, download if not local"""
        try:
            s3_model_path = f"models/{settings.MODEL_ID}"
            
            # Check if model exists locally
            if os.path.exists(self.model_dir):
                # If exists locally but not in S3, upload it
                if not self.s3.check_file_exists(f"{s3_model_path}/pytorch_model.bin"):
                    logger.info("Uploading model to S3...")
                    return self.s3.upload_directory(self.model_dir, s3_model_path)
                return True
            else:
                # If exists in S3, download it
                if self.s3.check_file_exists(f"{s3_model_path}/pytorch_model.bin"):
                    logger.info("Downloading model from S3...")
                    return self.s3.download_directory(s3_model_path, self.model_dir)
                # If neither exists, download from HuggingFace
                return self.download_from_huggingface()
                
        except Exception as e:
            logger.error(f"Error syncing model with S3: {str(e)}")
            return False
            
    def download_from_huggingface(self) -> bool:
        """Download model from HuggingFace"""
        try:
            logger.info(f"Downloading model from HuggingFace: {settings.MODEL_ID}")
            
            # Download and save model locally
            tokenizer = BartTokenizer.from_pretrained(settings.MODEL_ID)
            model = BartForConditionalGeneration.from_pretrained(settings.MODEL_ID)
            
            os.makedirs(self.model_dir, exist_ok=True)
            tokenizer.save_pretrained(self.model_dir)
            model.save_pretrained(self.model_dir)
            
            # Upload to S3
            s3_model_path = f"models/{settings.MODEL_ID}"
            return self.s3.upload_directory(self.model_dir, s3_model_path)
            
        except Exception as e:
            logger.error(f"Error downloading from HuggingFace: {str(e)}")
            return False
    
    def load_model(self):
        """Load model either from local storage or S3"""
        try:
            # Ensure model is synced
            if not self.sync_model_with_s3():
                raise Exception("Failed to sync model with S3")
            
            # Load model and tokenizer
            self.tokenizer = BartTokenizer.from_pretrained(self.model_dir)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model = BartForConditionalGeneration.from_pretrained(
                self.model_dir,
                torch_dtype=torch.float32
            ).to(device)
            
            self.model.eval()
            logger.info(f"Model loaded successfully on {device}")
            
            return self.model, self.tokenizer, device
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


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
        

    def generate_summary(self,  text, custom_prompt=None):
        try:
            
            prompt = self.format_prompt(text , custom_prompt)
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=settings.MAX_LENGTH,
                truncation=True,
                padding="longest"
            )

            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=settings.MAX_SUMMARY_LENGTH,      
                    min_length=settings.MIN_SUMMARY_LENGTH,       
                    num_beams=settings.NUM_BEAMS,         
                    length_penalty=settings.LENGTH_PENALTY,   
                    no_repeat_ngram_size=3, 
                    early_stopping=True,
                    top_k=50,            
                    top_p=settings.TOP_P,          
                    do_sample=True ,
                    temperature=settings.TEMPERATURE,
                    repetition_penalty=1.2
                )
            
            summary = self.tokenizer.decode(outputs[0] , skip_special_tokens=True)

            return self.format_summary(summary) 

        except Exception as e:
            self.logger.error(f"Error in generate_sumamry: {str(e)}")
            raise Exception(f"Summary Generation Failed: {str(e)}")

def create_llm(logger:Logger) -> BartModelManager:
    return BartModelManager.get_instance(logger)
    
