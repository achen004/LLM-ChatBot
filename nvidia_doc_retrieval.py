from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import ArxivLoader
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from pydantic import BaseModel, Field
from typing import List
from IPython.display import clear_output
from functools import partial
from rich.console import Console
from rich.style import Style
from rich.theme import Theme

console = Console()
base_style = Style(color="#76B900", bold=True)
pprint = partial(console.print, style=base_style)

from langchain.text_splitter import RecursiveCharacterTextSplitter

documents = ArxivLoader(query="2404.16130").load()  ## GraphRAG
# documents = ArxivLoader(query="2404.03622").load()  ## Visualization-of-Thought
# documents = ArxivLoader(query="2404.19756").load()  ## KAN: Kolmogorov-Arnold Networks
# documents = ArxivLoader(query="2404.07143").load()  ## Infini-Attention
# documents = ArxivLoader(query="2210.03629").load()  ## ReAct

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", ",", " ", ""],
)

## Some nice custom preprocessing
# documents[0].page_content = documents[0].page_content.replace(". .", "")
docs_split = text_splitter.split_documents(documents)

# def include_doc(doc):
#     ## Some chunks will be overburdened with useless numerical data, so we'll filter it out
#     string = doc.page_content
#     if len([l for l in string if l.isalpha()]) < (len(string)//2):
#         return False
#     return True

# docs_split = [doc for doc in docs_split if include_doc(doc)]

print(len(docs_split))

def RPrint(preface="State: "):
    def print_and_return(x, preface=""):
        print(f"{preface}{x}")
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

def PPrint(preface="State: "):
    def print_and_return(x, preface=""):
        pprint(preface, x)
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))


class DocumentSummaryBase(BaseModel):
    running_summary: str = Field("", description="Running description of the document. Do not override; only update!")
    main_ideas: List[str] = Field([], description="Most important information from the document (max 3)")
    loose_ends: List[str] = Field([], description="Open questions that would be good to incorporate into summary, but that are yet unknown (max 3)")


summary_prompt = ChatPromptTemplate.from_template(
    "You are generating a running summary of the document. Make it readable by a technical user."
    " After this, the old knowledge base will be replaced by the new one. Make sure a reader can still understand everything."
    " Keep it short, but as dense and useful as possible! The information should flow from chunk to (loose ends or main ideas) to running_summary."
    " The updated knowledge base keep all of the information from running_summary here: {info_base}."
    "\n\n{format_instructions}. Follow the format precisely, including quotations and commas"
    "\n\nWithout losing any of the info, update the knowledge base with the following: {input}"
)

def RExtract(pydantic_class, llm, prompt):
    '''
    Runnable Extraction module
    Returns a knowledge dictionary populated by slot-filling extraction
    '''
    parser = PydanticOutputParser(pydantic_object=pydantic_class)
    instruct_merge = RunnableAssign({'format_instructions' : lambda x: parser.get_format_instructions()})
    def preparse(string):
        if '{' not in string: string = '{' + string
        if '}' not in string: string = string + '}'
        string = (string
            .replace("\\_", "_")
            .replace("\n", " ")
            .replace("\]", "]")
            .replace("\[", "[")
        )
        # print(string)  ## Good for diagnostics
        return string
    return instruct_merge | prompt | llm | preparse | parser


latest_summary = ""

## TODO: Use the techniques from the previous notebook to complete the exercise
def RSummarizer(knowledge, llm, prompt, verbose=False):
    '''
    Exercise: Create a chain that summarizes
    '''
    ###########################################################################################
    ## START TODO:

    def summarize_docs(docs):        
        ## TODO: Initialize the parse_chain appropriately; should include an RExtract instance.
        parser=RExtract(knowledge, llm, prompt)
        ## HINT: You can get a class using the <object>.__class__ attribute...
        parse_chain = RunnableAssign({'info_base' : parser})
        ## TODO: Initialize a valid starting state. Should be similar to notebook 4
        state = {'info_base':knowledge()}
        state['input']=docs
        global latest_summary  ## If your loop crashes, you can check out the latest_summary
        
        for i, doc in enumerate(docs):
            ## TODO: Update the state as appropriate using your parse_chain component
            assert 'info_base' in state 
            state=parse_chain.invoke(state)
            if verbose:
                print(f"Considered {i+1} documents")
                pprint(state['info_base'])
                latest_summary = state['info_base']
                clear_output(wait=True)

        return state['info_base']
        
    ## END TODO
    ###########################################################################################
    
    return RunnableLambda(summarize_docs)

# instruct_model = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1").bind(max_tokens=4096)
instruct_model = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1").bind(max_tokens=4096)
instruct_llm = instruct_model | StrOutputParser()

## Take the first 10 document chunks and accumulate a DocumentSummaryBase
doc_sum=DocumentSummaryBase(running_summary='Description of document',
                           main_ideas=['most important part of document'],
                           loose_ends=['why should someone read this?'])
summarizer = RSummarizer(DocumentSummaryBase, instruct_llm, summary_prompt, verbose=True)
summary = summarizer.invoke(docs_split[:10])