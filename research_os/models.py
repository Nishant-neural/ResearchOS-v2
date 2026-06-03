# models.py

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from sentence_transformers import (
    SentenceTransformer,
)

from config import (
    EMBEDDING_MODEL,
    GENERATION_MODEL,
)


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

tokenizer = AutoTokenizer.from_pretrained(
    GENERATION_MODEL
)

generation_model = (
    AutoModelForSeq2SeqLM
    .from_pretrained(
        GENERATION_MODEL
    )
)

generation_model.eval()


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

generation_model.to(device)