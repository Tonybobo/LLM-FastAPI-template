from bs4 import BeautifulSoup
import httpx
from urllib.parse import urlparse
from abc import ABC,abstractmethod 
import re
from typing import List
from langchain.docstore.document import Document

class BaseWebLoader(ABC):
    def __init__(self):
        self.client = httpx.AsyncClient()

    def _clean_text(self, text:str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @abstractmethod
    def parse_html(self , soup: BeautifulSoup) -> str:
        """Abstract method to be implemented by specific new media parsers"""
        pass
    
    def _remove_common_elements(self , soup: BeautifulSoup , elements: list =None):
        """Remove unwanted elements from HTML"""
        default_elements = ['script' , 'style', 'nav' , 'header' , 'footer' , 'aside']
        elements_to_remove = default_elements + (elements if elements else []) 

        for element in soup.find_all(elements_to_remove):
            element.decompose()
    

class WebLoader(BaseWebLoader):
    def __init__(self):
        super().__init__()
        self.parsers = {}
        self._register_parsers()
    
    def _register_parsers(self):
        """Register all available parsers"""

        from src.loaders.media import(
            GenericParser,
            AsiaoneParser
        )

        self.parsers = {
            'asiaone.com': AsiaoneParser(),
            'generic' : GenericParser()
        }
    def parse_html(self, soup: BeautifulSoup) -> str:
        """Default implementation using generic parser if no available parser"""
        return self.parsers['generic'].parse_html(soup)
    
    async def load_and_process(self, url: str) -> List[Document]:
        try: 
            response = await self.client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text , 'html.parser')
            domain = urlparse(url).netloc.lower()

            parser = self.parsers.get(domain , self.parsers['generic'])
            text = parser.parse_html(soup)

            if not text:
                raise Exception("No content could be extracted from the webpage")

            document = Document(
                page_content=text,
                metadata={
                    "source": url,
                    "title": soup.title.string if soup.title else "",
                    "domain": domain
                }
            )

            return [document] if document else []
        
        except Exception as e: 
            raise Exception(f"Error loading URL {url} : {str(e)}")