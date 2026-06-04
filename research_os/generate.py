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
def generate_hidden_answer(
    query,
    chunk_ids,
):

    retrieved = hidden_search(
        chunk_ids
    )

    cached_memories = []

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

        # -----------------------------------
        # FORCE FIXED SHAPE
        # -----------------------------------

        current_len = hidden_states.shape[1]

        if current_len < TARGET_LEN:

            pad_amount = (
                TARGET_LEN - current_len
            )

            hidden_states = (
                torch.nn.functional.pad(
                    hidden_states,
                    (0, 0, 0, pad_amount),
                )
            )

            attention_mask = (
                torch.nn.functional.pad(
                    attention_mask,
                    (0, pad_amount),
                )
            )

        elif current_len > TARGET_LEN:

            hidden_states = hidden_states[
                :,
                :TARGET_LEN,
                :
            ]

            attention_mask = attention_mask[
                :,
                :TARGET_LEN
            ]

        cached_memories.append(
            hidden_states
        )

    # -----------------------------------
    # ANSWER INSTRUCTION
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

    # -----------------------------------
    # LATENT SUPERPOSITION
    # -----------------------------------

    if len(cached_memories) == 0:

        superposed_memory = torch.zeros(
            (
                1,
                TARGET_LEN,
                generation_model.config.d_model,
            )
        )

    else:

        superposed_memory = (
            cached_memories[0].clone()
        )

        print(
            "\n===== MEMORY SUPERPOSITION ====="
        )

        for idx, memory in enumerate(
            cached_memories[1:]
        ):

            # -----------------------------------
            # WEIGHTED ADDITION
            # -----------------------------------

            superposed_memory += (
                0.25 * memory
            )

            print(
                f"\nAfter cached memory {idx + 1}"
            )

            print(
                "mean:",
                superposed_memory.mean().item()
            )

            print(
                "std:",
                superposed_memory.std().item()
            )

            print(
                "avg norm:",
                superposed_memory.norm(
                    dim=-1
                ).mean().item()
            )

            print(
                "max abs:",
                superposed_memory.abs().max().item()
            )

        print(
            "================================\n"
        )

    # -----------------------------------
    # FINAL MEMORY LAYOUT
    # -----------------------------------

    memory = torch.cat(
        [
            query_hidden,
            superposed_memory,
            answer_hidden,
        ],
        dim=1,
    )

    # -----------------------------------
    # ATTENTION MASKS
    # -----------------------------------

    query_mask = (
        query_inputs.attention_mask
    )

    memory_mask = torch.ones(
        (
            superposed_memory.shape[0],
            superposed_memory.shape[1],
        ),
        dtype=torch.long,
    )

    answer_mask = (
        answer_inputs.attention_mask
    )

    attention_mask = torch.cat(
        [
            query_mask,
            memory_mask,
            answer_mask,
        ],
        dim=1,
    )

    encoder_outputs = BaseModelOutput(
        last_hidden_state=memory
    )

    # -----------------------------------
    # FINAL DEBUG
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

    print(
        "dtype:",
        memory.dtype
    )

    print(
        "device:",
        memory.device
    )

    print("========================\n")

    # -----------------------------------
    # GENERATION
    # -----------------------------------

    outputs = generation_model.generate(
        encoder_outputs=encoder_outputs,
        attention_mask=attention_mask,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )