from datetime import date
from langchain import OpenAI, PromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from utils import hf_embedding
import warnings
import os
from dotenv import load_dotenv

# Load default environment variables (.env)
load_dotenv()
# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "\033[91m\033[1m" + "OPENAI_API_KEY environment variable is missing from .env" + "\033[0m\033[0m"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def create_qa_chain(prompt_context: dict):
    date_str = date.today().strftime("%Y-%m-%d")
    date_prompt = f"Today's date is {date_str}. "
    name_prompt = f"The question below is asked by {prompt_context['name']}." if prompt_context['name'] else ""
    prompt_template = (
            date_prompt
            + name_prompt
            + (
                """You're an expert at analyzing conversations. Use the following pieces of chat messages to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer. Remember to reference participant names when giving an answer. 

                {context}

                "Question: {question}
                "Answer:"""
            )
        )
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    persist_directory = './.db'

    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=hf_embedding())

    # check if db collection contains any documents
    if len(vectorstore.get()) == 0:
        warnings.warn("No collections found. Please run src/embed_messages.py first.")
        return

    chain_type_kwargs = {"prompt": PROMPT, "verbose": False}

    qa = RetrievalQA.from_chain_type(llm=OpenAI(max_tokens=1000, temperature=0.3),
                                     chain_type="stuff",
                                     chain_type_kwargs=chain_type_kwargs,
                                     retriever=vectorstore.as_retriever(search_kwargs={"k": 7}),
                                     )

    return qa

def main():
    name = input("What's your name?\n")
    qa = create_qa_chain({"name": name})
    while True:
        query = input("\nEnter a question:\n")
        if query == "exit":
            break
        answer = qa({'query': query})['result']
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()