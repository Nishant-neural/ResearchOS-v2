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


def debug_hidden(name, hidden):

    print(f"\n===== {name} =====")

    print("shape:", hidden.shape)
    print("mean:", hidden.mean().item())
    print("std:", hidden.std().item())
    print(
        "avg token norm:",
        hidden.norm(dim=-1).mean().item()
    )
    print(
        "max abs:",
        hidden.abs().max().item()
    )

    print("========================\n")


@torch.no_grad()
def inspect_text_encoding(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    encoder_out = generation_model.encoder(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        return_dict=True,
    )

    debug_hidden(
        "FRESH TEXT ENCODING",
        encoder_out.last_hidden_state,
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

    query_hidden = (
    query_hidden
    /
    query_hidden.norm(
        dim=-1,
        keepdim=True
    ).clamp(min=1e-6)
)

    query_target_norm = (
    query_hidden.norm(
        dim=-1
    ).mean()
)
    debug_hidden(
        "QUERY ENCODER OUTPUT",
        query_hidden,
    )

    # -----------------------------------
    # CACHED MEMORY
    # -----------------------------------

    for idx, item in enumerate(retrieved):

        hidden_states = torch.tensor(
            item.payload["hidden_states"]
        ).float()

        hidden_states = (
    hidden_states
    /
    hidden_states.norm(
        dim=-1,
        keepdim=True
    ).clamp(min=1e-6)
)

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

        debug_hidden(
            f"CACHED MEMORY {idx}",
            hidden_states,
        )

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

    answer_hidden = (
    answer_hidden
    /
    answer_hidden.norm(
        dim=-1,
        keepdim=True
    ).clamp(min=1e-6)
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

        print(
        "\n===== MEMORY SUPERPOSITION ====="
    )

    # -----------------------------------
    # TRUE SUPERPOSITION
    # -----------------------------------

        superposed_memory = torch.zeros_like(
        cached_memories[0]
    )

        for memory in cached_memories:

            superposed_memory += memory

    # -----------------------------------
    # MATCH DECODER EXPECTED SCALE
    # -----------------------------------

        target_norm = (
        query_hidden.norm(
            dim=-1
        ).mean()
    )

        current_norm = (
        superposed_memory.norm(
            dim=-1
        ).mean()
    )

        superposed_memory = (
        superposed_memory
        * (
            target_norm
            / (
                current_norm + 1e-8
            )
        )
    )

        print(
        "target_norm:",
        target_norm.item()
    )

        print(
        "scaled_norm:",
        superposed_memory.norm(
            dim=-1
        ).mean().item()
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
# PAD QUERY TO TARGET_LEN
# -----------------------------------

    query_len = query_hidden.shape[1]

    if query_len < TARGET_LEN:

        query_hidden = torch.nn.functional.pad(
        query_hidden,
        (
            0,
            0,
            0,
            TARGET_LEN - query_len,
        ),
    )

    else:

        query_hidden = query_hidden[
        :,
        :TARGET_LEN,
        :
    ]

# -----------------------------------
# PAD ANSWER TO TARGET_LEN
# -----------------------------------

    answer_len = answer_hidden.shape[1]

    if answer_len < TARGET_LEN:

        answer_hidden = torch.nn.functional.pad(
        answer_hidden,
        (
            0,
            0,
            0,
            TARGET_LEN - answer_len,
        ),
    )

    else:

        answer_hidden = answer_hidden[
        :,
        :TARGET_LEN,
        :
    ]

# -----------------------------------
# FULL LATENT SUPERPOSITION
# -----------------------------------

    memory = (
    query_hidden
    + superposed_memory
    + answer_hidden
)

# -----------------------------------
# MATCH QUERY SCALE
# -----------------------------------
    target_norm = query_target_norm

    current_norm = (
    memory.norm(
        dim=-1
    ).mean()
)

    memory = (
    memory
    * (
        target_norm
        / (
            current_norm + 1e-8
        )
    )
)

# -----------------------------------
# SIMPLE ATTENTION MASK
# -----------------------------------

    attention_mask = torch.ones(
    (
        memory.shape[0],
        memory.shape[1],
    ),
    dtype=torch.long,
)

    print(
    "FULL SUPERPOSED NORM:",
    memory.norm(
        dim=-1
    ).mean().item()
)

    print(
    "FULL SUPERPOSED STD:",
    memory.std().item()
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
        do_sample=False,
        
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )