# %%capture
# %pip install -q langchain langchain-nvidia-ai-endpoints gradio rich
# %pip install -q arxiv pymupdf faiss-cpu

## If you encounter a typing-extensions issue, restart your runtime and try again
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
# ChatNVIDIA.get_available_models()

from functools import partial
from rich.console import Console
from rich.style import Style
from rich.theme import Theme
import json
import random
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import ArxivLoader
from langchain.document_transformers import LongContextReorder
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.schema.runnable.passthrough import RunnableAssign
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.vectorstores import FAISS
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from functools import partial
from operator import itemgetter
import gradio as gr

console = Console()
base_style = Style(color="#76B900", bold=True)
norm_style = Style(bold=True)
pprint = partial(console.print, style=base_style)
pprint2 = partial(console.print, style=norm_style)

## Utility Runnables/Methods
def RPrint(preface=""):
    """Simple passthrough "prints, then returns" chain"""
    def print_and_return(x, preface):
        if preface: print(preface, end="")
        pprint(x)
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string. Optional, but useful"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name:
            out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

## Optional; Reorders longer documents to center of output text
long_reorder = RunnableLambda(LongContextReorder().transform_documents)

# NVIDIAEmbeddings.get_available_models()
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")

# ChatNVIDIA.get_available_models()
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")

conversation = [  ## This conversation was generated partially by an AI system, and modified to exhibit desirable properties
    "[User]  Hello! My name is Beras, and I'm a big blue bear! Can you please tell me about the rocky mountains?",
    "[Agent] The Rocky Mountains are a beautiful and majestic range of mountains that stretch across North America",
    "[Beras] Wow, that sounds amazing! Ive never been to the Rocky Mountains before, but Ive heard many great things about them.",
    "[Agent] I hope you get to visit them someday, Beras! It would be a great adventure for you!"
    "[Beras] Thank you for the suggestion! Ill definitely keep it in mind for the future.",
    "[Agent] In the meantime, you can learn more about the Rocky Mountains by doing some research online or watching documentaries about them."
    "[Beras] I live in the arctic, so I'm not used to the warm climate there. I was just curious, ya know!",
    "[Agent] Absolutely! Lets continue the conversation and explore more about the Rocky Mountains and their significance!"
]

## Streamlined from_texts FAISS vectorstore construction from text list
convstore = FAISS.from_texts(conversation, embedding=embedder)
retriever = convstore.as_retriever()

#test retriever
pprint(retriever.invoke("What is your name?"))
pprint(retriever.invoke("Where are the Rocky Mountains?"))

context_prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context"
    "\n\nRetrieved Context: {context}"
    "\n\nUser Question: {question}"
    "\nAnswer the user conversationally. User is not aware of context."
)

chain = (
    {
        'context': convstore.as_retriever() | long_reorder | docs2str,
        'question': (lambda x:x)
    }
    | context_prompt
    # | RPrint()
    | instruct_llm
    | StrOutputParser()
)

pprint(chain.invoke("Where does Beras live?"))
pprint(chain.invoke("Where are the Rocky Mountains?"))
pprint(chain.invoke("Where are the Rocky Mountains? Are they close to California?"))
pprint(chain.invoke("How far away is Beras from the Rocky Mountains?"))

## Reset knowledge base and define what it means to add more messages.
convstore = FAISS.from_texts(conversation, embedding=embedder)

def save_memory_and_get_output(d, vstore):
    """Accepts 'input'/'output' dictionary and saves to convstore"""
    vstore.add_texts([f"User said {d.get('input')}", f"Agent said {d.get('output')}"])
    return d.get('output')

chat_prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context"
    "\n\nRetrieved Context: {context}"
    "\n\nUser Question: {input}"
    "\nAnswer the user conversationally. Make sure the conversation flows naturally.\n"
    "[Agent]"
)


conv_chain = (
    {
        'context': convstore.as_retriever() | long_reorder | docs2str,
        'input': (lambda x:x)
    }
    | RunnableAssign({'output' : chat_prompt | instruct_llm | StrOutputParser()})
    | partial(save_memory_and_get_output, vstore=convstore)
)

pprint(conv_chain.invoke("I'm glad you agree! I can't wait to get some ice cream there! It's such a good food!"))
print()
pprint(conv_chain.invoke("Can you guess what my favorite food is?"))
print()
pprint(conv_chain.invoke("Actually, my favorite is honey! Not sure where you got that idea?"))
print()
pprint(conv_chain.invoke("I see! Fair enough! Do you know my favorite food now?"))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", ",", " "],
)

print("Loading Documents")
docs = [
    ArxivLoader(query="1706.03762").load(),  ## Attention Is All You Need Paper
    ArxivLoader(query="1810.04805").load(),  ## BERT Paper
    ArxivLoader(query="2005.11401").load(),  ## RAG Paper
    ArxivLoader(query="2205.00445").load(),  ## MRKL Paper
    ArxivLoader(query="2310.06825").load(),  ## Mistral Paper
    ArxivLoader(query="2306.05685").load(),  ## LLM-as-a-Judge
    ArxivLoader(query="2505.03335").load(),  ## Absolute Zero: Reinforced Self-play Reasoning with Zero Data
    ArxivLoader(query="2507.16773").load(),  ## When LLMs Copy to Think: Uncovering Copy-Guided Attacks in Reasoning LLMs
    ArxivLoader(query="2507.16809").load(),  ## LingBench++: A Linguistically-Informed Benchmark and Reasoning Framework for Multi-Step and Cross-Cultural Inference with LLMs
    ## Some longer papers
    #https://arxiv.org/pdf/2505.03335
    # ArxivLoader(query="2210.03629").load(),  ## ReAct Paper
    # ArxivLoader(query="2112.10752").load(),  ## Latent Stable Diffusion Paper
    # ArxivLoader(query="2103.00020").load(),  ## CLIP Paper
]

for doc in docs:
    content = json.dumps(doc[0].page_content)
    if "References" in content:
        doc[0].page_content = content[:content.index("References")]

## Split the documents and also filter out stubs (overly short chunks)
print("Chunking Documents")
docs_chunks = [text_splitter.split_documents(doc) for doc in docs]
docs_chunks = [[c for c in dchunks if len(c.page_content) > 200] for dchunks in docs_chunks]

## Make some custom Chunks to give big-picture details
doc_string = "Available Documents:"
doc_metadata = []
for chunks in docs_chunks:
    metadata = getattr(chunks[0], 'metadata', {})
    doc_string += "\n - " + metadata.get('Title')
    doc_metadata += [str(metadata)]

extra_chunks = [doc_string] + doc_metadata

## Printing out some summary information for reference
pprint(doc_string, '\n')
for i, chunks in enumerate(docs_chunks):
    print(f"Document {i}")
    print(f" - # Chunks: {len(chunks)}")
    print(f" - Metadata: ")
    pprint(chunks[0].metadata)
    print()

print("Constructing Vector Stores")
vecstores = [FAISS.from_texts(extra_chunks, embedder)]
vecstores += [FAISS.from_documents(doc_chunks, embedder) for doc_chunks in docs_chunks]

embed_dims = len(embedder.embed_query("test"))
def default_FAISS():
    '''Useful utility for making an empty FAISS vectorstore'''
    return FAISS(
        embedding_function=embedder,
        index=IndexFlatL2(embed_dims),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )

def aggregate_vstores(vectorstores):
    ## Initialize an empty FAISS Index and merge others into it
    ## We'll use default_faiss for simplicity, though it's tied to your embedder by reference
    agg_vstore = default_FAISS()
    for vstore in vectorstores:
        agg_vstore.merge_from(vstore)
    return agg_vstore

## Unintuitive optimization; merge_from seems to optimize constituent vector stores away
docstore = aggregate_vstores(vecstores)

print(f"Constructed aggregate docstore with {len(docstore.docstore._dict)} chunks")

# NVIDIAEmbeddings.get_available_models()
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
# ChatNVIDIA.get_available_models()
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")
# instruct_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")

convstore = default_FAISS()

def save_memory_and_get_output(d, vstore):
    """Accepts 'input'/'output' dictionary and saves to convstore"""
    vstore.add_texts([
        f"User previously responded with {d.get('input')}",
        f"Agent previously responded with {d.get('output')}"
    ])
    return d.get('output')

initial_msg = (
    "Hello! I am a document chat agent here to help the user!"
    f" I have access to the following documents: {doc_string}\n\nHow can I help you?"
)

chat_prompt = ChatPromptTemplate.from_messages([("system",
    "You are a document chatbot. Help the user as they ask questions about documents."
    " User messaged just asked: {input}\n\n"
    " From this, we have retrieved the following potentially-useful info: "
    " Conversation History Retrieval:\n{history}\n\n"
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used. Make your response conversational.)"
), ('user', '{input}')])

stream_chain = chat_prompt| RPrint() | instruct_llm | StrOutputParser()

retrieval_chain = (
    {   'input': (lambda x: x)   }
    ## TODO: Make sure to retrieve history & context from convstore & docstore, respectively.
    | RunnableAssign({'history' : itemgetter('input') | convstore.as_retriever() | long_reorder | docs2str})
    | RunnableAssign({'context' : itemgetter('input') | docstore.as_retriever() | long_reorder | docs2str})
    | RPrint()
)

def chat_gen(message, history=[], return_buffer=True):
    buffer = ""
    ## First perform the retrieval based on the input message
    retrieval = retrieval_chain.invoke(message)
    line_buffer = ""

    ## Then, stream the results of the stream_chain
    for token in stream_chain.stream(retrieval):
        buffer += token
        ## If you're using standard print, keep line from getting too long
        yield buffer if return_buffer else token

    ## Lastly, save the chat exchange to the conversation memory buffer
    save_memory_and_get_output({'input':  message, 'output': buffer}, convstore)


## Start of Agent Event Loop
test_question = "Tell me about RAG!"  ## <- modify as desired

## Before you launch your gradio interface, make sure your thing works
for response in chat_gen(test_question, return_buffer=False):
    print(response, end='')

docstore.save_local("docstore_index")

import subprocess

# Run the tar command from Python
subprocess.run(['tar', 'czvf', 'docstore_index.tgz', 'docstore_index'], check=True)
subprocess.run(['rm', 'rf', 'docstore_index'], check=True)
subprocess.run(['tar', 'xzrf', 'docstore_index.tgz'], check=True)

new_db = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)
docs = new_db.similarity_search("Testing the index")
print(docs[0].page_content[:1000])

#RAG Chain evaluation
subprocess.run(['tar', 'xzvf', 'docstore_index.tgz'], check=True)
docstore = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)
docs = list(docstore.docstore._dict.values())

def format_chunk(doc):
    return (
        f"Paper: {doc.metadata.get('Title', 'unknown')}"
        f"\n\nSummary: {doc.metadata.get('Summary', 'unknown')}"
        f"\n\nPage Body: {doc.page_content}"
    )

## This printout just confirms that your store has been retrieved
pprint(f"Constructed aggregate docstore with {len(docstore.docstore._dict)} chunks")
pprint(f"Sample Chunk:")
#print(format_chunk(docs[len(docs)//2]))
print(format_chunk(docs[8]))

# NVIDIAEmbeddings.get_available_models()
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")

# ChatNVIDIA.get_available_models()
instruct_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
llm = instruct_llm | StrOutputParser()

def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string. Optional, but useful"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name: out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

chat_prompt = ChatPromptTemplate.from_template(
    "You are a document chatbot. Help the user as they ask questions about documents."
    " User messaged just asked you a question: {input}\n\n"
    " The following information may be useful for your response: "
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used. Make your response conversational, but remain objective)"
    "\n\nUser Question: {input}"
)

def output_puller(inputs):
    """"Output generator. Useful if your chain returns a dictionary with key 'output'"""
    if isinstance(inputs, dict):
        inputs = [inputs]
    for token in inputs:
        if token.get('output'):
            yield token.get('output')

# NVIDIAEmbeddings.get_available_models()
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
# ChatNVIDIA.get_available_models()
instruct_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
llm = instruct_llm | StrOutputParser()

def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string. Optional, but useful"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name: out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

chat_prompt = ChatPromptTemplate.from_template(
    "You are a document chatbot. Help the user as they ask questions about documents."
    " User messaged just asked you a question: {input}\n\n"
    " The following information may be useful for your response: "
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used. Make your response conversational, but remain objective)"
    "\n\nUser Question: {input}"
)

def output_puller(inputs):
    """"Output generator. Useful if your chain returns a dictionary with key 'output'"""
    if isinstance(inputs, dict):
        inputs = [inputs]
    for token in inputs:
        if token.get('output'):
            yield token.get('output')

#Pull in your desired RAG Chain. Memory not necessary
## Chain 1 Specs: "Hello World" -> retrieval_chain 
##   -> {'input': <str>, 'context' : <str>}
long_reorder = RunnableLambda(LongContextReorder().transform_documents)  
context_getter = itemgetter('input') | docstore.as_retriever() | long_reorder | docs2str
retrieval_chain = {'input' : (lambda x: x)} | RunnableAssign({'context' : context_getter})

## Chain 2 Specs: retrieval_chain -> generator_chain 
##   -> {"output" : <str>, ...} -> output_puller
generator_chain = chat_prompt | llm 
generator_chain = {'output' : generator_chain} | RunnableLambda(output_puller)  

rag_chain = retrieval_chain | generator_chain

# pprint(rag_chain.invoke("Tell me something interesting!"))
for token in rag_chain.stream("Tell me something interesting!"):
    print(token, end="")

#synthetic baseline question-answer pair
num_questions = 3
synth_questions = []
synth_answers = []

simple_prompt = ChatPromptTemplate.from_messages([('system', '{system}'), ('user', 'INPUT: {input}')])

for i in range(num_questions):
    doc1, doc2 = random.sample(docs, 2)
    sys_msg = (
        "Use the documents provided by the user to generate an interesting question-answer pair."
        " Try to use both documents if possible, and rely more on the document bodies than the summary."
        " Use the format:\nQuestion: (good question, 1-3 sentences, detailed)\n\nAnswer: (answer derived from the documents)"
        " DO NOT SAY: \"Here is an interesting question pair\" or similar. FOLLOW FORMAT!"
    )
    usr_msg = (
        f"Document1: {format_chunk(doc1)}\n\n"
        f"Document2: {format_chunk(doc2)}"
    )

    qa_pair = (simple_prompt | llm).invoke({'system': sys_msg, 'input': usr_msg})
    synth_questions += [qa_pair.split('\n\n')[0]]
    synth_answers += [qa_pair.split('\n\n')[1]]
    pprint2(f"QA Pair {i+1}")
    pprint2(synth_questions[-1])
    pprint(synth_answers[-1])
    print()

#generate synthetic answers
rag_answers = []
for i, q in enumerate(synth_questions):
    rag_answer = ""
    rag_answer=rag_chain.invoke(q)
    rag_answers += [rag_answer]
    pprint2(f"QA Pair {i+1}", q, "", sep="\n")
    pprint(f"RAG Answer: {rag_answer}", "", sep='\n')

## TODO: Adapt this prompt for whichever LLM you're actually interested in using. 
## If it's llama, maybe system message would be good?
eval_prompt = ChatPromptTemplate.from_template("""INSTRUCTION: 
Evaluate the following Question-Answer pair for human preference and consistency.
Assume the first answer is a ground truth answer and has to be correct.
Assume the second answer may or may not be true.
Take into account of sources from both answers. 
Score the answer with a firm 0 or 1 with the following are considerations:
[0] The second answer is better than the first and does not introduce any inconsistencies.
[1] The second answer lies, does not answer the question, or is inferior to the first answer.

Output Format:
[Score] Justification

{qa_trio}

EVALUATION: 
""")

pref_score = []

trio_gen = zip(synth_questions, synth_answers, rag_answers)
for i, (q, a_synth, a_rag) in enumerate(trio_gen):
    pprint2(f"Set {i+1}\n\nQuestion: {q}\n\n")

    qa_trio = f"Question: {q}\n\nAnswer 1 (Ground Truth): {a_synth}\n\n Answer 2 (New Answer): {a_rag}"
    pref_score += [(eval_prompt | llm).invoke({'qa_trio': qa_trio})]
    pprint(f"Synth Answer: {a_synth}\n\n")
    pprint(f"RAG Answer: {a_rag}\n\n")
    pprint2(f"Synth Evaluation: {pref_score[-1]}\n\n")

#DEBUGGING PURPOSES
# pprint(pref_score)
# print(len(pref_score))

# for score in pref_score:
#     if "[Score: 0]" in score:
#         print(score)

pref_score = sum(("[Score: 0]" in score) for score in pref_score) / len(pref_score)
print(f"Preference Score: {pref_score}")

#import py file and run following in jupyter notebook (retriever and generator app routes must be set up beforehand):
# %%js
# var url = 'http://'+window.location.host+':8090';
# element.innerHTML = '<a style="color:green;" target="_blank" href='+url+'><h1>< Link To Gradio Frontend ></h1></a>';