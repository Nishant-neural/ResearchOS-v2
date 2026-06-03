# generate.py

import torch

from transformers.modeling_outputs import (
    BaseModelOutput,
)

from models import (
    tokenizer,
    generation_model,
)

from config import (
    ENCODER_LAYER_INDEX,
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
    )

    query_encoder = (
        generation_model.encoder(
            input_ids=query_inputs.input_ids,
            attention_mask=query_inputs.attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    )

    query_hidden = (
       query_encoder.last_hidden_state
    )

    memory_tensors.append(
        query_hidden
    )

    memory_masks.append(
        query_inputs.attention_mask
    )

    # -----------------------------------
    # CACHED MEMORY
    # -----------------------------------

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

        memory_tensors.append(
            hidden_states
        )

        memory_masks.append(
            attention_mask
        )

    # -----------------------------------
    # ANSWER INSTRUCTION
    # -----------------------------------

    answer_prompt = f"""
    CONTEXT ENDED

    using context,
   answer the question.

    Question:
    {query}

    Answer:
    """

    answer_inputs = tokenizer(
        answer_prompt,
        return_tensors="pt",
    )

    answer_encoder = (
        generation_model.encoder(
            input_ids=answer_inputs.input_ids,
            attention_mask=answer_inputs.attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    )

    answer_hidden = (
        answer_encoder.last_hidden_state
    )

    memory_tensors.append(
        answer_hidden
    )

    memory_masks.append(
        answer_inputs.attention_mask
    )

    # -----------------------------------
    # CONCAT ALL LATENT STATES
    # -----------------------------------

    memory = torch.cat(
        memory_tensors,
        dim=1,
    )

    attention_mask = torch.cat(
        memory_masks,
        dim=1,
    )

    encoder_outputs = BaseModelOutput(
        last_hidden_state=memory
    )

    # -----------------------------------
    # GENERATION
    # -----------------------------------

    outputs = generation_model.generate(
        encoder_outputs=encoder_outputs,
        attention_mask=attention_mask,
        max_new_tokens=128,
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )