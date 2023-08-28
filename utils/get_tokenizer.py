from transformers import AutoTokenizer


def get_tokenizer(model_name, num_tokens):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"cls_token": "<|image|>"})  # add special image token to tokenizer
    
    visual_token_idx = []
    # Add [IMG] tokens to the vocabulary.
    for i in range(num_tokens):
        tokenizer.add_tokens(f'[IMG{i}]')
        token_idx = tokenizer(f'[IMG{i}]', add_special_tokens=False).input_ids
        visual_token_idx.append(token_idx[0])

    return tokenizer, visual_token_idx


