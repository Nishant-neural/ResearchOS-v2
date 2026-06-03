# chunking.py

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from models import tokenizer

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


splitter = (
    RecursiveCharacterTextSplitter
    .from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
        ],
    )
)


def chunk_text(text):

    return splitter.split_text(text)