from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM # transformers >= 4.36.0.dev0
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import MergerRetriever
from langchain.vectorstores import Chroma
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
import openai
import torch
import os


# API_KEYS
OPENAI_API_KEY= os.environ['OPENAI_API_KEY']
COHERE_API_KEY = os.environ['COHERE_API_KEY']
openai.api_key = OPENAI_API_KEY


# TEMPLATES
TEMPLATE1 = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. However try to guess the closest answer as you can. Keep the answer as concise as possible. 
{context}

Question: {question}

Return the answer in this format. Use only the capital letters for the answer.
{{ "Answer": "A" , "Explanation": "Explanation about the option"}}

"""

TEMPLATE2 = """The following is a conversation with an AI Large Language Model. The AI has been trained to answer questions, provide recommendations, and help with decision making. The AI follows user requests. The AI thinks outside the box.

AI: How can I help you today?
You: ### Instruction:
1- Analyze the related information and the nature of the question.
2- Use those results to answer the question.
Use only the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

### Question: 
{question}

### Task:
Return the answer in JSON format. Use only the capital letters for the answer. 
Example: {{ "Explanation": "<explanation>", "Answer": "A" or "B" or "C" or "D"}}

### Output:
AI:
"""


TEMPLATE2_CoT = """The following is a conversation with an AI Large Language Model. The AI has been trained to answer questions, provide recommendations, and help with decision making. The AI follows user requests. The AI thinks outside the box.

AI: How can I help you today?
You: ### Instructions:
1- Analyze the related information and the nature of the question.
2- Use those results to answer the question.
Use only the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

### Question: 
{question}

To answer this question, let's think about it step by step.
AI:
"""

TEMPLATE_ZRS = """The following is a conversation with an AI Large Language Model. The AI has been trained to answer questions, provide recommendations, and help with decision making. The AI follows user requests. The AI thinks outside the box.

AI: How can I help you today?
You: ### Question: 
{question}

### Task:
Return the answer in JSON format. Use only the capital letters for the answer. 
Example: {{ "Explanation": "<explanation>", "Answer": "A" or "B" or "C" or "D"}}

### Output:
AI:
"""

###############################################################
#	RETRIEVERS
###############################################################
def load_retriever(
	embedding,
	persist_directory,
	k=30
):
	# Create embedding object
	#embedding = OpenAIEmbeddings(
	#	openai_api_key=OPENAI_API_KEY
	#) # TODO: replace this with an open source embedding

	# client = chromadb.PersistentClient(path="./chromadb")
	retriever1 = Chroma(
					collection_name="bcsc", # Name of the collection
					persist_directory=persist_directory,
					embedding_function=embedding,
				)
	retriever2 = Chroma(
					collection_name="shields", # Name of the collection
					persist_directory=persist_directory,
					embedding_function=embedding
				)

	# Assuming retriever1 and retriever2 need to be converted to retriever objects
	retriever1 = retriever1.as_retriever(
		search_type="similarity", 
		search_kwargs={"k": k, "include_metadata": True}
	)
	retriever2 = retriever2.as_retriever(
		search_type="similarity", 
		search_kwargs={"k": k, "include_metadata": True}
	)

	# Now initialize MergerRetriever with the correct retriever objects
	#lotr = MergerRetriever(retrievers=[retriever1, retriever2]) # TODO: modify -> a better way to select retrievers
	lotr = MergerRetriever(retrievers=[retriever1])
	return lotr

def load_cohere(
	lotr,
	n=5
):
	compressor = CohereRerank(cohere_api_key=COHERE_API_KEY, top_n = n) # type: ignore
	compression_retriever = ContextualCompressionRetriever(
		base_compressor=compressor, 
		base_retriever=lotr
	)
	return compression_retriever

###############################################################
#	LANGCHAIN
###############################################################
def load_qa_chain(llm, retriever, template, verbose=True):
	QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

	# Run chain
	qa_chain = RetrievalQA.from_chain_type(
		llm,
		retriever=retriever,  # Use the MergerRetriever object directly
		return_source_documents=True,
		chain_type="stuff",  # Ensure this is a valid argument
		chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
		verbose=verbose
	)
	return qa_chain


###############################################################
#	LLM
###############################################################
def load_llm(model_name='mistralai/Mistral-7B-Instruct-v0.1'):
	# Tokenizer
	#################################################################
	tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_side = "right"

	# bitsandbytes parameters
	#################################################################
	# Activate 4-bit precision base model loading
	use_4bit = True

	# Compute dtype for 4-bit base models
	bnb_4bit_compute_dtype = "float16"

	# Quantization type (fp4 or nf4)
	bnb_4bit_quant_type = "nf4"

	# Activate nested quantization for 4-bit base models (double quantization)
	use_nested_quant = False

	# Set up quantization config
	#################################################################
	compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

	bnb_config = BitsAndBytesConfig(
		load_in_4bit=use_4bit,
		bnb_4bit_quant_type=bnb_4bit_quant_type,
		bnb_4bit_compute_dtype=compute_dtype,
		bnb_4bit_use_double_quant=use_nested_quant,
	)
 
	# Load pre-trained config
	#################################################################
	model = AutoModelForCausalLM.from_pretrained(
		model_name,
		quantization_config=bnb_config,
		#do_sample=True
	)
 
	# LLM generation pipeline
	#################################################################
	text_generation_pipeline = pipeline(
		model=model,
		tokenizer=tokenizer,
		task="text-generation",
		temperature=0.2,
		repetition_penalty=1.1,
		return_full_text=True,
		max_new_tokens=300,
		do_sample=True,
		pad_token_id = 50256,
		#pad_token_id = 2
	)

	llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
	return llm