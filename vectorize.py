#Import necessary libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from uuid import uuid4
import json
from PyPDF2 import PdfReader, PdfWriter
import chromadb
from chromadb.utils import embedding_functions
import fire
from tqdm import tqdm


'''
# TODO: implement open source embedding function
# embedding functions
#emb_model_name = "jinaai/jina-embeddings-v2-base-en"
#embedding = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=emb_model_name)
#embedding  = SentenceTransformerEmbeddings(model_name=emb_model_name)
#embedding = JinaEmbeddings(
#	jina_api_key="jina_*", model_name="jina-embeddings-v2-base-en"
#)

#emb_model_name = "Salesforce/SFR-Embedding-Mistral"
#embedding = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=emb_model_name)
#emb_fnc  = SentenceTransformerEmbeddings(model_name=emb_model_name)
'''


def split_books():
    #Load the JSON file that contains textbook list
    book_folder = "."
    f = open('{book_folder}/bcsc_books.json'.format(book_folder=book_folder))

    book_list = json.load(f)

    for book in book_list['books']:
        reader = PdfReader(book['path'])
        split_folder = book['id']
        print(split_folder)
        
        #Split the book into pages
        for page_num in range(1, book['pages']):
            writer = PdfWriter()
            page = reader.pages[page_num]

            # This is CPU intensive! It ZIPs the contents of the page
            # page.compress_content_streams()

            writer.add_page(page)
            file_name = "{book_folder}/{split_folder}/{page}.pdf".format(
                book_folder=book_folder,
                split_folder=split_folder,
                page=str(page_num)
            )
            print(file_name)
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            
            with open(file_name, "wb") as fh:
                writer.write(fh)


def vectorize(
    emb_func,
    book_folder='../embedding_bcsc/data',
    collection_name='bcsc',
    save_dir="./chromadb",
    chunk_size=2000,
    verbose=False
):
    #Load the textbook and Start embedding and upserting to Pinecone
    f = open('{book_folder}/bcsc_books.json'.format(book_folder=book_folder))
    book_list = json.load(f)

    # client = chromadb.PersistentClient(path="./chromadb")
    vectordb = Chroma(
                    collection_name=collection_name, # Name of the collection
                    persist_directory=save_dir,
                    embedding_function=emb_func
    )

    for book in tqdm(book_list['books'], desc='Books', position=0):
        sub_folder = book['id']
        
        pages = os.listdir("{book_folder}/{sub_folder}".format(
            book_folder=book_folder,
            sub_folder=sub_folder
        ))

        if verbose: print(f"Processing: {book['name']}\n{sub_folder}")

        for page in tqdm(pages, desc='Pages', position=1, leave=False):
            page_num = page.split('.')[0]
            
            file = "{book_folder}/{sub_folder}/{page}".format(
                book_folder=book_folder,
                sub_folder=sub_folder,
                page=page
            )

            if verbose: print(f"\tPage: {page_num}\t{file}")

            reader = PdfReader(file)
            data = reader.pages[0].extract_text()

            if (len(data) > 0):
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)

                texts = text_splitter.split_text(data)

                metadata = { #number #section: #type: question/answer
                    'id': book['id'],
                    'source': book['name'],
                    'page': page_num,
                    'year': book['year'],
                    'pages': book['pages'],
                    'author': book['authors'] 
                }

                ids = [str(uuid4()) for _ in range(len(texts))]

                metadatas = [metadata] * len(texts)

                vectordb.add_texts(
                    texts=texts,
                    ids=ids,
                    metadatas=metadatas
                )

                vectordb.persist()


if __name__ == "__main__":
    #Create embedding object
    #embed_model_name = 'text-embedding-ada-002'

    #embedding = OpenAIEmbeddings(
    #    openai_api_key=os.getenv('OPENAI_API_KEY')
    #)

    from langchain.embeddings import SentenceTransformerEmbeddings

    emb_model_name = "jinaai/jina-embeddings-v2-base-en"
    #emb_model_name = "Salesforce/SFR-Embedding-Mistral"
    #embedding = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=emb_model_name)
    embedding  = SentenceTransformerEmbeddings(model_name=emb_model_name)

    fire.Fire(vectorize(embedding))
    print("Done")