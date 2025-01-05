import streamlit as st
from src.models.llm import create_llm, generate_summary
from src.loaders.web_loader import WebLoader
import asyncio
import httpx

st.set_page_config(page_title="Article Summarizer", layout="wide")

@st.cache_resource
def initialize_components():
    model, tokenizer, device = create_llm()
    loader = WebLoader()
    return model, tokenizer, loader, device

def validate_url(url: str) -> bool:
    try:
        response = httpx.head(url)
        return response.is_success
    except:
        return False

def main():
    st.title("Article Summarizer")
    
    model, tokenizer, loader, device = initialize_components()
    
    url = st.text_input("Enter article URL:")
    custom_prompt = st.text_area(
        "Custom summarization prompt (optional):",
        placeholder="E.g., Summarize this article focusing on technical details..."
    )
    
    if st.button("Generate Summary"):
        if url:
            if not validate_url(url):
                st.error("Invalid or inaccessible URL. Please check the URL and try again.")
                return
                
            with st.spinner("Processing article..."):
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    documents = loop.run_until_complete(
                        loader.load_and_process(url)
                    )
                    
                    # Combine all document chunks into one text
                    full_text = " ".join([doc.page_content for doc in documents])
                    
                    # Generate summary
                    summary = generate_summary(model, tokenizer, full_text, device, custom_prompt)
                    
                    st.subheader("Summary")
                    st.write(summary)
                    
                    with st.expander("View Source Text"):
                        st.text(full_text)
                            
                except Exception as e:
                    st.error(f"Error processing the article: {str(e)}")
                    st.error(f"Error details: {type(e).__name__}")  # Add more error details

if __name__ == "__main__":
    main()




