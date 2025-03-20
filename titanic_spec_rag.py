import json
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Read table
with open('data/titanic.csv', 'r') as f:
    DATA = f.read()

# Read metadata
with open('data/metadata.json') as f:
    METADATA = json.load(f)


SYSTEM_MESSAGE = """
# PERSONA
You are a help museum assistant about the Titanic victims. Take your user message and use the context to provide the best answer.

# CONTEXT INFORMATION
- The following is a snippet of text that provides context to answer your user question:
- {context}

## EXAMPLE
Example 1:
<user> how many people were in Titanic?
<assistant> There were ..... people in Titanic

Example 2:
<user> how many people survived Titanic?
<assistant> There were .... who survived

# RESPONSE FORMAT
- Always give friendly responses in a natural language format.
"""

prompt = ChatPromptTemplate(
    [
        ('system', SYSTEM_MESSAGE),
        ('placeholder', '{messages}')
    ]
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    task_type="retrieval_document"
)

raw_documents = TextLoader('data/titanic.csv').load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)

print('splitting documents...')
documents = text_splitter.split_documents(raw_documents)

print('embedding DB into a vector store...')
db = Chroma.from_documents(documents, embeddings)

agent = prompt | llm

if __name__ == '__main__':
    
    query = input('Q: ')
    
    while query != 'quit':

        docs = db.similarity_search(query)

        response = agent.invoke(
            input={
                "context": docs[0].page_content,
                "messages": [('user', query)],
            }
        )
        print('R:', response.content)
        print()
        query = input('>')