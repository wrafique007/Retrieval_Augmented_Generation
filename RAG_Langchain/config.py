import os

from pathlib import Path
from tempfile import mkdtemp

from langchain_docling.loader import ExportType

def _get_env_from_colab_or_os(key):
    try:
        from google.colab import userdata

        try:
            return userdata.get(key)
        except userdata.SecretNotFoundError:
            pass
    except ImportError:
        pass
    return os.getenv(key)

HF_TOKEN = _get_env_from_colab_or_os("HF_TOKEN")
FILE_PATH = ["https://arxiv.org/pdf/2408.09869"]  # Docling Technical Report
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS=1024
EXPORT_TYPE = ExportType.DOC_CHUNKS
TOP_K = 3
MILVUS_URI = str(Path("docling_langchain.db").resolve())