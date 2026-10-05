import os
import json
import time
import logging

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_docling.loader import ExportType
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from langchain_docling import DoclingLoader
from transformers import AutoTokenizer

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from config import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Load environment variables
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


QUESTION = "Which are the main AI models in Docling?"

# Chat Template to work with OpenAI compatible endpoint
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer the query using ONLY the provided context. "
     "If the answer isn't in the context, say so."),
    ("human",
     "Context information is below.\n"
     "---------------------\n{context}\n---------------------\n"
     "Query: {input}\n"
     "Answer:"),
])


# Create tokenizer which will be used to tokenize text
tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
    max_tokens=MAX_TOKENS,
)
logger.info("Embedding model loaded")

# Configure converter with external plugins enabled
pdf_pipeline_options = PdfPipelineOptions()
pdf_pipeline_options.allow_external_plugins = True

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
    }
)

start = time.perf_counter()
# Load the document in Docling
loader = DoclingLoader(
    file_path=FILE_PATH,
    export_type=EXPORT_TYPE,
    chunker=HybridChunker(tokenizer=tokenizer, repeat_table_header=True),
    converter=converter,
)

docs = loader.load()
end = time.perf_counter()
time_taken = end - start
logger.info(f"\n\nTime taken for DoclingLoader to execute is {time_taken} seconds\n\n")

if EXPORT_TYPE == ExportType.DOC_CHUNKS:
    splits = docs
elif EXPORT_TYPE == ExportType.MARKDOWN:
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ],
    )
    splits = [split for doc in docs for split in splitter.split_text(doc.page_content)]
else:
    raise ValueError(f"Unexpected export type: {EXPORT_TYPE}")

logger.info(f"\n\ndocling_langchain path is {MILVUS_URI}\n\n")

# Create Embeddings and Ingest the document in vector database (Milvus)
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

start = time.perf_counter()
vectorstore = Milvus.from_documents(
    documents=splits,
    embedding=embedding,
    collection_name="docling_langchain",
    connection_args={"uri": MILVUS_URI},
    index_params={
        "index_type": "FLAT",
        "metric_type": "COSINE"},
    drop_old=True,
)
end = time.perf_counter()
time_taken = end - start
logger.info(f"\n\nTime taken to create embeddings is {time_taken} seconds\n\n")

# RAG
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1", # llama-server running model as mentioned by the User
    api_key="not-needed",
    model="llama",
    temperature=0.1,
    max_tokens=1024,
)


def clip_text(text, threshold=100):
    return f"{text[:threshold]}..." if len(text) > threshold else text

question_answer_chain = create_stuff_documents_chain(llm, PROMPT)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
resp_dict = rag_chain.invoke({"input": QUESTION})

clipped_answer = clip_text(resp_dict["answer"], threshold=300)
print(f"Question:\n{resp_dict['input']}\n\nAnswer:\n{clipped_answer}")
for i, doc in enumerate(resp_dict["context"]):
    print()
    print(f"Source {i + 1}:")
    print(f"  text: {json.dumps(clip_text(doc.page_content, threshold=350))}")
    for key in doc.metadata:
        if key != "pk":
            val = doc.metadata.get(key)
            clipped_val = clip_text(val) if isinstance(val, str) else val
            print(f"  {key}: {clipped_val}")