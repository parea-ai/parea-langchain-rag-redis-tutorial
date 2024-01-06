import os

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Redis
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from parea import Parea
from parea.utils.trace_integrations.langchain import PareaAILangchainTracer

from evals.eval import run_evals
from rag.config import INDEX_NAME, INDEX_SCHEMA, REDIS_URL, EMBED_MODEL

load_dotenv()

# Need to instantiate Parea for tracing and evals
p = Parea(api_key=os.getenv("PAREA_API_KEY"))

# Init Tracer which will send logs to Parea AI
parea_tracer = PareaAILangchainTracer()

# Init Embeddings
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# Connect to preloaded vectorstore ( After you run ingest.py )
vectorstore = Redis.from_existing_index(
    embedding=embedder, index_name=INDEX_NAME, schema=INDEX_SCHEMA, redis_url=REDIS_URL
)

# Init retriever
retriever = vectorstore.as_retriever(search_type="mmr")

# Init global variable to store context from retriever
context_from_retriever = ""


def _format_docs(docs) -> str:
    """
    Format the docs retrieved from the retriever
    :param docs: list of documents retrieved from the retriever
    :return: str of concatenated documents
    """
    global context_from_retriever
    formatted_context = "\n\n".join(doc.page_content for doc in docs)
    context_from_retriever = formatted_context
    return formatted_context


# A sentinel value to indicate whether to add the source to the prompt
# When running evals such as exact match excluding the source could help the LLM judge the answer better
ADD_SOURCE = False
ADD_SOURCE_TEXT = """Include the 'source' and 'start_index from the metadata included in the context you used to 
answer the question"""

# Define our prompt template
template = """
Use the following pieces of context from Nike's financial 10k filings
dataset to answer the question. Do not make up an answer if there is no
context provided to help answer it. {ADD_SOURCE_TEXT}

Context:
---------
{context}

---------
Question: {question}
---------

Answer:
"""

prompt = ChatPromptTemplate.from_template(template).partial(
    ADD_SOURCE_TEXT=ADD_SOURCE_TEXT if ADD_SOURCE else ""
)


# RAG Chain
model = ChatOpenAI(model_name="gpt-3.5-turbo-16k")
chain = (
    RunnableParallel(
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
    )
    | prompt
    | model
    | StrOutputParser()
)


def invoke(question: str) -> tuple[str, str]:
    """
    Invoke the chain with the question
    :param question:
    :return:  response and trace_id
    """
    response = chain.invoke(question, config={"callbacks": [parea_tracer]})
    trace_id = parea_tracer.get_parent_trace_id()
    return response, str(trace_id)


def run_chain(question: str, target: str, run_eval: bool):
    """
    Run the chain with the question and target answer and optionally run evals
    :param question: question to ask
    :param target: target answer
    :param run_eval: whether to run evals

    :return: None
    """
    response, trace_id = invoke(question)
    print("Question: ", question, "\n")
    print("Response: ", response, "\n")

    if run_eval:
        print("Evals started in thread: \n")
        run_evals(
            trace_id=trace_id,
            question=question,
            context=context_from_retriever,
            response=response,
            target_answer=target,
        )
        print("Done")
