# generate.py

import torch

from models import (
    tokenizer,
    generation_model,
)

from qdrant_store import (
    hidden_search,
)


@torch.no_grad()
def generate_text_answer(
    query,
    chunks,
):

    context = "\n\n".join(chunks)

    prompt = f"""
    Question:
    {query}

    Context:
    {context}

    Answer:
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    outputs = generation_model.generate(
        **inputs,
        max_new_tokens=128,
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )


@torch.no_grad()
def generate_hidden_answer(
    query,
    chunk_ids,
):

    retrieved = hidden_search(
        chunk_ids
    )

    memory_tensors = []

    memory_masks = []

    query_prompt = f"""
    QUESTION:
    {query}

    RELEVANT LATENT MEMORY BEGINS
    """

    query_inputs = tokenizer(
        query_prompt,
        return_tensors="pt",
    )

    query_encoder = (
        generation_model.encoder(
            input_ids=query_inputs.input_ids,
            attention_mask=query_inputs.attention_mask,
        )
    )

    memory_tensors.append(
        query_encoder.last_hidden_state
    )

    memory_masks.append(
        query_inputs.attention_mask
    )

    for item in retrieved:

        memory_tensors.append(
            torch.tensor(
                item.payload[
                    "hidden_states"
                ]
            )
        )

        memory_masks.append(
            torch.tensor(
                item.payload[
                    "attention_mask"
                ]
            )
        )

    memory = torch.cat(
        memory_tensors,
        dim=1,
    )

    attention_mask = torch.cat(
        memory_masks,
        dim=1,
    )

    decoder_input_ids = tokenizer(
        f"""
        RELEVANT LATENT MEMORY ENDED

        Answer precisely:
        {query}
        """,
        return_tensors="pt",
    ).input_ids

    outputs = generation_model.generate(
        encoder_outputs=(memory,),
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        max_new_tokens=128,
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )