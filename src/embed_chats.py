import json
import datetime
import os
from utils import unicode_converter, hf_embedding
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from typing import List

persist_directory="./.db"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def create_embeddings(dir_path: str, data_source = 'messenger'):
    """
    Creates embeddings for all 'message_1.json' files in a directory
    dir_path: path to directory containing 'message_1.json' files
    """

    all_docs = []

    # Loop through root directory
    for root, _, files in os.walk(dir_path):
        # Loop through all files
        for file in files:
            # Check if file is 'message_1.json', Meta's conventional name for its chat history file
            if file == 'message_1.json':
                file_path = os.path.join(root, file)
                # Open and read json file
                with open(file_path, 'r', encoding="utf8") as f:
                    # convert latin-1 encoded strings to utf-8 encoded strings within the json object
                    decoded_data = unicode_converter(json.load(f))
                    documents = create_documents_from_data(decoded_data, file_path, data_source)
                    all_docs.extend(documents)
    
    embed_conversations(all_docs)


def format_message(row: dict) -> str:
    """Format each chat message into a readable string.
    row: one single message that looks like this:
    {
      "sender_name": "levi",
      "timestamp_ms": 1681354684281,
      "content": "Levi sent an attachment.",
      "share": {
        "link": "https://www.instagram.com/p/Cq57ZOZgr6y/?feed_type=reshare_chaining",
        "share_text": "Succession szn is back",
        "original_content_owner": "overheardonwallstreet"
      }
    }
    """
    sender = row["sender_name"]
    text = ''
    # `share_text` contains more information than just `content` "XX sent an attachment."
    if row.get('share') and isinstance(row['share'].get('share_text'), str):
        text = row['share']['share_text']
    else:
        text = row["content"]

    date = datetime.datetime.fromtimestamp(row["timestamp_ms"] / 1000).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    return f"{sender} on {date}: {text}\n\n"

def create_documents_from_data(structured_data: dict, file_path: str, data_source: str) -> List[Document]:
    """
    Creates a list of Langchain documents from the chat history data.
    Split the documents if necessary using HuggingFace's SentenceTransformersTokenTextSplitter.
    structured_data: the dict that contains the chat history data
    file_path: the path to the chat history file
    data_source: the chat platform that the chat history file is from
    """
    
    # store participants name as metadata
    convert_dict = lambda x: ', '.join(participant['name'] for participant in x)
    metadata = {
        'participants': convert_dict(structured_data['participants']),
        'source': file_path,
        'chat_platform': data_source,
        }

    # sort by timestamp so chat appears in chronological order
    messages = structured_data['messages']
    messages.sort(key=lambda x: x["timestamp_ms"])

    # combine messages into one string
    chat = "".join(
            [
                format_message(message)
                for message in messages
                if message.get('content') and isinstance(message.get('content'), str)
            ]
        )

    # initialize docs with just one document of the entire conversation
    docs = [Document(page_content=chat, metadata=metadata)]

    # split texts if they could go over the tokenizer's limit
    # see https://www.sbert.net/docs/pretrained_models.html for max token length
    # TODO: use an actual tokenizer instead of character length
    if len(chat) > 300:
        # text splitting
        text_splitter = SentenceTransformersTokenTextSplitter()
        texts = text_splitter.split_text(chat)
        docs = text_splitter.create_documents(texts, metadatas=[metadata] * len(texts))
    
    return docs

def embed_conversations(documents: List[Document]):
    """
    Embeds conversations and stores them in a database.
    documents: a list of Langchain documents
    """

    db = Chroma.from_documents(documents=documents, embedding=hf_embedding(), persist_directory=persist_directory)
    db.persist()

def main():
    create_embeddings('./messages')

if __name__ == "__main__":
    main()
