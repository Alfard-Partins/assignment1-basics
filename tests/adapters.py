from cs336_basics.tokenizer import train_bpe

    return train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        **kwargs,
    )


