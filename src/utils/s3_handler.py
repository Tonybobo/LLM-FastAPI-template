import boto3 
import os
from botocore.exceptions import ClientError
from src.utils.logger import Logger 
from src.utils.config import settings 


class S3Handler:
    def __init__(self , logger: Logger):
        self.s3 = boto3.client(
            's3' , 
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key= settings.AWS_SECRET_ACCESS_KEY,
            region_name= settings.AWS_REGION
        )
        self.bucket = settings.S3_BUCKET
        self.logger = logger
    
    def upload_files(self , local_path:str , s3_path:str) -> bool:
        """Upload File to S3"""
        try:
            self.s3.upload_files(local_path , self.bucket , s3_path)
            self.logger.info(f"Successfully upload to {local_path} to s3://{self.bucket}/{s3_path}")
            return True
        except ClientError as e:
            self.logger.error(f"Error uploading to S3: {str(e)}")
            return False
    
    def download_file(self , s3_path:str , local_path:str)->bool:
        """Download File from S3"""
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            self.s3.download_file(self.bucket , s3_path , local_path)
            self.logger.info(f"Successfully download s3://{self.bucket}/{s3_path} to {local_path}")

            return True;
            
        except ClientError as e:
            self.logger.error(f"Error downloading from S3: {str(e)}")
            return False
    
    def check_file_exists (self , s3_path:str)-> bool:
        """Check if File exist in S3"""
        try:
            self.s3.head_object(Bucket=self.bucket , Key=s3_path)
            return True
        except ClientError:
            return False
    
    def list_files(self, prefix: str = "") -> list:
        """List Files in S3 bucket with given prefix"""
        try:
            response = self.s3.list_objects_v2(Bucket= self.bucket , Prefix= prefix)
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except ClientError as e:
            self.logger.error(f"Error listing S3 files: {str(e)}")
            return []
    
    def upload_directory(self, local_dir:str , s3_prefix:str)-> bool:
        """Upload Directory to S3"""
        try:
            for root,_ , files in os.walk(local_dir):
                for file in files:
                    local_path = os.path.join(root , file)

                    relative_path = os.path.relpath(local_path , local_dir)
                    self.logger.info(f"Local Relative Path: {relative_path}")

                    s3_path = os.path.join(s3_prefix , relative_path).replace("\\" , "/")
                    self.logger.info(f"S3 Path : {s3_path}")
                    
                    success = self.upload_files(local_path , s3_path)
                    if not success:
                        return False
                    
            return True
        except Exception as e:
            self.logger.error(f"Error uploading directory to S3: {str(e)}")
            return False
            
    def download_directory(self , s3_prefix: str, local_dir:str) -> bool:
        """"Download entire directory from S3"""
        try:
            files = self.list_files(s3_prefix)
            for s3_path in files:
                relative_path = os.path.relpath(s3_path , s3_prefix)
                local_path= os.path.join(local_dir , relative_path)
                
                success = self.download_file(s3_path , local_path)
                if not success:
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Error downloading directory from S3: {str(e)}")
            return False
        

            


