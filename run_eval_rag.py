"""
See more in `bcsc_mistral_cohere_v2.ipynb`

python run_eval_rag.py \
    --model_name_or_url='http://127.0.0.1:5000/v1/' \
    --filepath='data/ophthoquestions/oq_sampled.csv' \
    --cohere_n=5 \
    --chromadb_k=30 \
    --suffix='_mixtral8x7B_n_5_without_cohere' \
    --save_dir='results/sampled'
"""

import os
from src.setup import (
    load_retriever, load_cohere, load_qa_chain, 
    TEMPLATE2, TEMPLATE2_CoT, OPENAI_API_KEY
)
from tqdm import tqdm # progress bar
from langchain.chat_models import ChatOpenAI
import json
import pandas as pd
from ast import literal_eval
from datetime import datetime 
import fire
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from src.utils  import (
    save_result, is_url, postprocess_string, 
    is_legit, refine_result,
    OpensourceLLM, GPT, MODEL_LIST
)


OPENAI_KEY = os.environ["OPENAI_API_KEY"]

question_with_reasoning = '''The following is a conversation with an AI Large Language Model. The AI has been trained to answer questions, provide recommendations, and help with decision making. The AI follows user requests. The AI thinks outside the box.

AI: How can I help you today?
You: ### Question: 
{}

### Reasoning:
{}

### Task:
Return the answer in JSON format. Use only the capital letters for the answer. 
Example: {{ "Explanation": "<explanation>", "Answer": "A" or "B" or "C" or "D"}}

### Output:
AI:
'''

def main(
    filepath='data/ophthoquestions/oq_sampled.csv',
    model_name_or_url='http://127.0.0.1:5000/v1/',
    #model_name_or_url='gpt-4-turbo',
    chromadb_dir='./embeddings/v3/chromadb',
    temp=0.3,
    cohere_n=5,
    chromadb_k=30,
    suffix='test',
    save_dir='results/debug'
):
    
    # ========== Load models ==========
    # if model_name is provided, use model_name, otherwise use url
    if is_url(model_name_or_url): 
        # Open source LLM
        # RetrievalQA.from_chain_type only works with OpenAI-like API object
        llm = ChatOpenAI(
            openai_api_base=model_name_or_url,
            temperature=temp,
            openai_api_key='ramdom_stuff'
        )
    else: # OpenAI API
        llm = ChatOpenAI(
            model_name=model_name_or_url, 
            temperature=temp
        )

    emb_fnc = OpenAIEmbeddings(
        model="text-embedding-ada-002", # default model
        openai_api_key=OPENAI_API_KEY
    )
    # TODO: if zero-shot, does not load chromadb and cohere ranker

    # Retrievers 
    # NOTE: make sure the embedding function and the chromadb directory are compatible 
    # (i.e. the chromadb vectors were produced by the same embedding function)
    retriever = load_retriever(
        embedding=emb_fnc,
        persist_directory=chromadb_dir,
        k=chromadb_k
    )

    # Cohere
    if cohere_n:
        retriever = load_cohere(retriever, n=cohere_n)

    # QA chain
    if 'cot' in suffix:
        TEMPLATE = TEMPLATE2_CoT
    else:
        TEMPLATE = TEMPLATE2

    qa_chain = load_qa_chain(llm, retriever, TEMPLATE, verbose=False)
    #qa_chain = load_qa_chain(llm, lotr, TEMPLATE2, verbose=False)

    if model_name_or_url in MODEL_LIST:
        llm_refine = GPT(model_name=model_name_or_url, temp=temp)
    else:
        llm_refine = OpensourceLLM(
            model_name_or_path=os.path.join(model_name_or_url, 'completions/'), 
            temp=temp
        )


    # ========== prepare data ==========
    df = pd.read_csv(filepath)
    df.sort_values(by=['Section', 'Question Number'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ========== Run ==========
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = filepath.split('/')[-1].split('.')[0]
    data = []
    for i, row in tqdm(df.iterrows(), total=len(df)):

        qi = row['Question Number']
        question_full = row['Question Full']
        s = row['Section']

        start_time = datetime.now() 
        # get predicted answer from chatgpt
        result = qa_chain({"query": question_full})

        try:
            result_final = literal_eval(result['result'])
            answer = result_final['Answer']
            explanation = result_final['Explanation']
        except:
            if 'cot' in suffix:
                result_final = llm_refine.get_answer(question_with_reasoning.format(question_full, result['result']))
            else:
                result_final = llm_refine.get_answer(result['result'])
                #result_final = result['result']
            
            answer, explanation = refine_result(result_final, llm_refine.call)

        result_str = postprocess_result(result)

        time_elapsed = datetime.now() - start_time 

        data.append({
			'idx':i,
			'Section': s,
			'Question Number': qi,
			'Correct Answer': row['Correct Answer'],
			'Answer': answer,
			'Explanation': explanation,
            'Result Raw': result['result'], # raw from qa_chain
            'Result': str(result_final),    # after refining, dictionary {Answer, Explanation}
			'Result Full': str(result_str), # with query and retrieved documents
            'Question Full': row['Question Full'],
            'Cognitive Level': row['Cognitive Level'],
            'Difficulty Level': row['Difficulty Level'],
			'time': round(time_elapsed.total_seconds(), 3)
		})

        if i%30==0:
            save_result(data, filename, save_dir, suffix)

    save_result(data, filename, save_dir, suffix)

    print('>>> Done!')

    # print results
    df_results = pd.DataFrame(data)
    df_results['Correct'] = df_results['Correct Answer'] == df_results['Answer']
    print('>>> Overall Accuracy:', df_results['Correct'].mean())

    # accuracy by sections
    section_accuracy = df_results.groupby('Section')['Correct'].mean()
    print(section_accuracy)


def postprocess_result(r):
    '''
    Make sure result is in a json readable format
    '''
    # process Document class
    source_documents = []
    for s in r['source_documents']:
        source_documents.append(dict(s))
    r['source_documents'] = source_documents

    # make sure after converting to string, it is still a valid json
    return json.dumps(r)

def postprocess_string(s):
    s = s.replace('\n', ' ')
    if s.endswith('"}') or s.endswith('" }'):
        return s
    elif s.endswith('"') or s.endswith('" '):
        return s + '}'
    else:
        return s + '"}'

if __name__ == '__main__':
    fire.Fire(main)


    