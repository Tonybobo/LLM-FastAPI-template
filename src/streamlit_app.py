import streamlit as st
from src.models.llm import create_llm , generate_summary
from loaders.web_loader import WebLoader
import asyncio
import httpx
import nest_asyncio

nest_asyncio.apply()

st.set_page_config(page_title="Article Summarizer", layout="wide")

@st.cache_resource
def initialize_components():
    try: 
        model, tokenizer, device = create_llm()
        loader = WebLoader()
        return model, tokenizer, loader, device
    except Exception as e:
        st.error(f"Error initializing component: {str(e)}")
        return None, None , None , None

async def validate_url(url: str) -> bool:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.head(url)
            return response.is_success
    except Exception:
        return False

async def process_article(loader , url):
    """Process article in async await manner"""
    try:
        documents = await loader.load_and_process(url)
        return documents
    except Exception as e:
        raise Exception(f"Error loading article : {str(e)}")

def main():
    st.title("Article Summarizer")
    
    components = initialize_components()
    if not all(components):
        st.error("Failed to initialize the required component. Please refresh")
        return

    model , tokenizer , loader , device = components
    
    url = st.text_input("Enter article URL:")
    custom_prompt = st.text_area(
        "Custom summarization prompt (optional):",
        placeholder="E.g., Summarize this article focusing on technical details..."
    )
    
    if st.button("Generate Summary"):
            if not url:
                st.error("Please enter a URL.")
                return
            
            with st.spinner("Validating URL..."):
                is_valid = asyncio.run(validate_url(url))
                if not is_valid:
                    st.error("Invalid or inaccessible URL. Please check the url and try again")
                    return
                
            with st.spinner("Processing article..."):
                try:
                    documents = asyncio.run(process_article(loader , url))

                    if not documents:
                        st.error("No content could be extracted from the article")
                    
                    full_text = " ".join([doc.page_content for doc in documents])
                    
                    with st.spinner("Generating Summary..."):
                    
                        summary = generate_summary(model, tokenizer, full_text, device, custom_prompt)
                        
                        st.subheader("Summary")
                        st.markdown(f"<p>{summary}</p>" , unsafe_allow_html=True)
                        
                        with st.expander("View Source Text"):
                            st.text(full_text)
                            
                except Exception as e:
                    st.error(f"Error processing the article: {str(e)}")
                    st.error(f"Error type: {type(e).__name__}")
                    if hasattr(e, '__cause__') and e.__cause__:
                        st.error(f"Caused by: {str(e.__cause__)}")

if __name__ == "__main__":
    main()




