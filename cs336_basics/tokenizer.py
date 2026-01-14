def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    verbose: bool = False,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_merges = vocab_size - 256 - (len(special_tokens) if special_tokens else 0)
    vocab: dict[int, bytes] = init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    # 1. Pre-tokenization
    # 1.1 Find chunk boundaries
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(
            f, desired_num_chunks=kwargs.get("desired_num_chunks", NUM_PROCESSES), split_special_token=b"\n"
        )

    if verbose:
        print_color(f"Identified {len(chunk_boundaries) - 1} chunks for pre-tokenization.")

    # 1.2 Count word frequencies across chunks using multiprocessing
    manager = Manager()
    queue = manager.Queue()
    processes: list[Process] = []

    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        p = Process(
            target=pre_tokenize_string_worker,
            args=(input_path, special_tokens, queue, start, end, False),
        )
        processes.append(p)
        p.start()

    if verbose:
        print_color("Pre-tokenization processes completed. Aggregating results...")

    word_counter = Counter()
    for _ in range(len(processes)):
        try:
            partial_counter = queue.get(timeout=10)
            word_counter.update(partial_counter)
        except Empty:
            continue
    for p in processes:
        p.join()

    if verbose:
        print_color(f"Completed pre-tokenization. Vocabulary size: {len(word_counter)} unique tokens.")

    pairs_counter = Counter()
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
    for word in word_counter:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_to_words[pair].add(word)
            pairs_counter[pair] += word_counter[word]

    # 2. BPE Core Loop
    pair_heap = build_pair_heap(pairs_counter, vocab)

    for i in trange(num_merges):
        most_frequent_pair = pop_most_frequent_pair(pair_heap, pairs_counter)
        new_id = update_vocab(vocab, most_frequent_pair)

        word_counter, pairs_counter, pair_heap, pair_to_words = merge_pairs_with_heap_index(
            word_counter, pairs_counter, most_frequent_pair, new_id, vocab, pair_heap, pair_to_words
        )

        merges.append((vocab[most_frequent_pair[0]], vocab[most_frequent_pair[1]]))

    if kwargs.get("save_path"):
        save_vocab_and_merges(vocab, merges, kwargs["save_path"])
        with open(os.path.join(kwargs["save_path"], "special_tokens.txt"), "w", encoding="utf-8") as f:
            if special_tokens:
                for token in special_tokens:
                    f.write(f"{token}\n")

    return vocab, merges
