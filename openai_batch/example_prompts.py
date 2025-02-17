"""
Create some simple example prompts.
"""

import argparse
import random

ACTIONS = [
    "Tell me a funny joke about",
    "Tell me a funny dad joke about",
    "Tell me two funny dad jokes about",
    "Write two paragraph short story about",
    "Write a poem about",
]

SUBJECTS = [
    "anything",
    "dogs",
    "life, the universe, and everything",
    "large language models",
]

ABSTRACTS = [
    "We study empirical scaling laws for language model performance on the cross-entropy loss. The loss scales as a power-law with model size, dataset size, and the amount of compute used for training, with some trends spanning more than seven orders of magnitude. Other architectural details such as network width or depth have minimal effects within a wide range. Simple equations govern the dependence of overfitting on model/dataset size and the dependence of training speed on model size. These relationships allow us to determine the optimal allocation of a fixed compute budget. Larger models are significantly more sample-efficient, such that optimally compute-efficient training involves training very large models on a relatively modest amount of data and stopping significantly before convergence.",
    "High throughput serving of large language models (LLMs) requires batching sufficiently many requests at a time. However, existing systems struggle because the key-value cache (KV cache) memory for each request is huge and grows and shrinks dynamically. When managed inefficiently, this memory can be significantly wasted by fragmentation and redundant duplication, limiting the batch size. To address this problem, we propose PagedAttention, an attention algorithm inspired by the classical virtual memory and paging techniques in operating systems. On top of it, we build vLLM, an LLM serving system that achieves (1) near-zero waste in KV cache memory and (2) flexible sharing of KV cache within and across requests to further reduce memory usage. Our evaluations show that vLLM improves the throughput of popular LLMs by 2-4× with the same level of latency compared to the state-of-the-art systems, such as FasterTransformer and Orca. The improvement is more pronounced with longer sequences, larger models, and more complex decoding algorithms.",
    "Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length. Approximate attention methods have attempted to address this problem by trading off model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a missing principle is making attention algorithms IOaware—accounting for reads and writes between levels of GPU memory. We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity of FlashAttention, showing that it requires fewer HBM accesses than standard attention, and is optimal for a range of SRAM sizes. We also extend FlashAttention to block-sparse attention, yielding an approximate attention algorithm that is faster than any existing approximate attention method. FlashAttention trains Transformers faster than existing baselines: 15% end-to-end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record, 3\u0002 speedup on GPT-2 (seq. length 1K), and 2.4\u0002 speedup on long-range arena (seq. length 1K-4K). FlashAttention and block-sparse FlashAttention enable longer context in Transformers, yielding higher quality models (0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document classification) and entirely new capabilities: the first Transformers to achieve better-than-chance performance on the Path-X challenge (seq. length 16K, 61.4% accuracy) and Path-256 (seq. length 64K, 63.1% accuracy).",
]


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "output",
        nargs="?",
        default="-",
        type=argparse.FileType("w", encoding="utf-8"),
        help="Where to write example prompts.",
    )

    parser.add_argument(
        "-n",
        default=1000,
        type=int,
        help="Number of example prompts to generate.",
    )

    parser.add_argument(
        "--embedding",
        "-e",
        help="Generate prompts appropriate for an embedding model.",
        default=False,
        action="store_true",
    )

    return parser


def main(args=None):
    args = get_parser().parse_args(args)

    for i in range(args.n):
        if args.embedding:
            prompt = random.choice(ABSTRACTS)
        else:
            prompt = f"{random.choice(ACTIONS)} {random.choice(SUBJECTS)}."

        args.output.write(prompt + "\n")
    args.output.flush()


if __name__ == "__main__":
    main()
