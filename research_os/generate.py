# generate.py

import torch

from transformers.modeling_outputs import (
    BaseModelOutput,
)

from models import (
    tokenizer,
    generation_model,
)

from qdrant_store import (
    hidden_search,
)


# -----------------------------------
# FORCE FIXED LATENT WIDTH
# -----------------------------------

TARGET_LEN = 256


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
@torch.no_grad()
def generate_hidden_answer(
    query,
    chunk_ids,
):

    retrieved = hidden_search(
        chunk_ids
    )

    memory_hidden_list = []
    memory_mask_list = []

    # -----------------------------------
    # QUERY FRAMING
    # -----------------------------------

    query_prefix = f"""
    QUESTION:
    {query}

    CONTEXT BEGINS
    """

    query_inputs = tokenizer(
        query_prefix,
        return_tensors="pt",
        truncation=True,
    )

    query_encoder = generation_model.encoder(
        input_ids=query_inputs.input_ids,
        attention_mask=query_inputs.attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    query_hidden = (
        query_encoder.last_hidden_state
    )

    # -----------------------------------
    # MEMORY RETRIEVAL
    # -----------------------------------

    DROP_FACTOR = 4

    for item in retrieved:

        hidden_states = torch.tensor(
            item.payload["hidden_states"]
        ).float()

        attention_mask = torch.tensor(
            item.payload["attention_mask"]
        )

        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)

        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        # -----------------------------------
        # TOKEN DROPPING
        # -----------------------------------

        hidden_states = hidden_states[
            :,
            ::DROP_FACTOR,
            :
        ]

        attention_mask = attention_mask[
            :,
            ::DROP_FACTOR
        ]

        memory_hidden_list.append(
            hidden_states
        )

        memory_mask_list.append(
            attention_mask
        )

    # -----------------------------------
    # ANSWER FRAMING
    # -----------------------------------

    answer_prompt = f"""
    CONTEXT ENDED

    using context,
    answer the question in your own words.

    Question:
    {query}

    Answer:
    """

    answer_inputs = tokenizer(
        answer_prompt,
        return_tensors="pt",
        truncation=True,
    )

    answer_encoder = generation_model.encoder(
        input_ids=answer_inputs.input_ids,
        attention_mask=answer_inputs.attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    answer_hidden = (
        answer_encoder.last_hidden_state
    )

    # -----------------------------------
    # CONCATENATION ONLY
    # -----------------------------------

    hidden_parts = [query_hidden]
    mask_parts = [query_inputs.attention_mask]

    hidden_parts.extend(
        memory_hidden_list
    )

    mask_parts.extend(
        memory_mask_list
    )

    hidden_parts.append(
        answer_hidden
    )

    mask_parts.append(
        answer_inputs.attention_mask
    )

    memory = torch.cat(
        hidden_parts,
        dim=1,
    )

    attention_mask = torch.cat(
        mask_parts,
        dim=1,
    )

    encoder_outputs = BaseModelOutput(
        last_hidden_state=memory
    )

    # -----------------------------------
    # DEBUG
    # -----------------------------------

    print("\n===== LATENT DEBUG =====")

    print(
        "hidden_states.shape:",
        memory.shape
    )

    print(
        "attention_mask.shape:",
        attention_mask.shape
    )

    print(
        "attention_mask.sum():",
        attention_mask.sum().item()
    )

    print(
        "hidden_states.mean():",
        memory.mean().item()
    )

    print(
        "hidden_states.std():",
        memory.std().item()
    )

    print(
        "avg token norm:",
        memory.norm(
            dim=-1
        ).mean().item()
    )

    print(
        "max abs value:",
        memory.abs().max().item()
    )

    print("========================\n")

    outputs = generation_model.generate(
        encoder_outputs=encoder_outputs,
        attention_mask=attention_mask,
        max_new_tokens=128,
        do_sample=False,
        num_beams = 1,
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )