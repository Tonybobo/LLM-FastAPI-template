from bs4 import BeautifulSoup
from src.loaders.web_loader import BaseWebLoader
import re

class GenericParser(BaseWebLoader):
    def parse_html(self , soup: BeautifulSoup) -> str:
        self._remove_common_elements(soup)

        main_content = (
            soup.find('article') or
            soup.find(class_=re.compile(r'article|content|story|post-content')) or 
            soup.find(role='main') or
            soup.find('main')
        )

        if(main_content):
            for unwanted in main_content.find_all(class_=re.compile(r'social|share|ad|comment|sidebar|related')):
                unwanted.decompose()

            text = main_content.get_text(separator=' ' , strip=True)
            return self._clean_text(text)

        return self._clean_text(soup.body.get_text(separator=' ' , strip=True))