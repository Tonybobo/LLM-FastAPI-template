from bs4 import BeautifulSoup
from src.loaders.web_loader import BaseWebLoader
import re

class AsiaoneParser(BaseWebLoader):
    def parse_html(self , soup:BeautifulSoup) -> str:
        try:
            self._remove_common_elements(soup)

            article = soup.find(class_='article_content')
            
            if article :
                for social in article.find_all('div',class_=re.compile(r'^dfp-')):
                    social.decompose()
                text = article.get_text(separator=' ' , strip=True)
                text_lines = text.split('\n')
                cleaned_lines = []
                skip_patterns = ['PHOTO:', 'PUBLISHED ON', '[[nid:', 'Share this article', 'This website is', 'embed']
        
                for line in text_lines:
                    if not any(pattern in line for pattern in skip_patterns) and line.strip():
                        cleaned_lines.append(line.strip())
                cleaned_text = ' '.join(cleaned_lines)

                return self._clean_text(cleaned_text)
            return ""
        except Exception as e:
            raise Exception(f"Error parsing HTML: {str(e)}")
