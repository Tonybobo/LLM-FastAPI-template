from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import httpx
import pandas as pd
from typing import List
from langchain.docstore.document import Document
import tempfile

class WebLoader:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
    
    async def fetch_content(self, url: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
    
    def clean_html(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Handle tables
        tables = soup.find_all('table')
        for table in tables:
            try:
                df = pd.read_html(str(table))[0]
                markdown_table = df.to_markdown(index=False)
                table.replace_with(soup.new_string(f"\n{markdown_table}\n"))
            except:
                table.decompose()
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'iframe', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Get text content
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up extra whitespace
        lines = (line.strip() for line in text.splitlines())
        return '\n'.join(line for line in lines if line)
    
    async def load_and_process(self, url: str) -> List[Document]:
        # Fetch content
        html_content = await self.fetch_content(url)
        
        # Clean HTML
        cleaned_text = self.clean_html(html_content)
        
        # Create temporary file to use with TextLoader
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(cleaned_text)
            temp_file.flush()
            
            # Load text using LangChain's TextLoader
            loader = TextLoader(temp_file.name)
            documents = loader.load()
        
        # Split into chunks
        splits = self.text_splitter.split_documents(documents)
        
        return splits