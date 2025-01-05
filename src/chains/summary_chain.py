from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

class SummaryChain:
    def __init__(self, llm):
        self.llm = llm
        self.default_template = """
        Write a comprehensive summary of the following text:
        "{text}"
        SUMMARY:
        """
        
    def create_chain(self, custom_prompt=None):
        if custom_prompt:
            prompt = PromptTemplate(
                template=custom_prompt + "\n{text}\n",
                input_variables=["text"]
            )
        else:
            prompt = PromptTemplate(
                template=self.default_template,
                input_variables=["text"]
            )
            
        return load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=prompt,
            combine_prompt=prompt
        )
    
    async def generate_summary(self, documents, custom_prompt=None):
        chain = self.create_chain(custom_prompt)
        return await chain.arun(documents)