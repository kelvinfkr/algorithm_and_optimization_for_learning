"""
快速下载 FineWeb-Edu - 并行 tokenize + 直接写入文件

用法：
    python download_data_v2.py --output ./data/fineweb --num_tokens 10B --workers 32
"""

import os
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='./data/fineweb')
    parser.add_argument('--num_tokens', type=str, default='10B')
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析目标
    num_tokens = args.num_tokens
    if num_tokens.endswith("B"):
        target = int(float(num_tokens[:-1]) * 1e9)
    elif num_tokens.endswith("M"):
        target = int(float(num_tokens[:-1]) * 1e6)
    else:
        target = int(num_tokens)
    
    # 估算需要多少文档（平均每文档约 1000 tokens）
    est_docs = int(target / 800) + 100000  # 多拿一点
    
    print(f"Target: {target:,} tokens")
    print(f"Estimated docs needed: {est_docs:,}")
    print(f"Workers: {args.workers}")
    
    # 非流式加载一部分
    print("\n[1/3] Loading dataset...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", 
        "sample-10BT", 
        split=f"train[:{est_docs}]",
        num_proc=args.workers
    )
    
    print(f"Loaded {len(ds):,} documents")
    
    # 并行 tokenize，直接返回 token 数组
    print("\n[2/3] Tokenizing (parallel)...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    def tokenize(example):
        return {'tokens': tokenizer.encode(example['text'])}
    
    ds = ds.map(
        tokenize, 
        num_proc=args.workers,
        remove_columns=ds.column_names,
        desc="Tokenizing"
    )
    
    # 拼接 tokens 并保存
    print(f"\n[3/3] Concatenating and saving...")
    token_file = output_dir / "tokens.bin"
    
    # 用 pyarrow list_flatten（C++ 向量化，很快）
    import pyarrow.compute as pc
    
    print("  Flattening tokens (pyarrow)...")
    table = ds.data  # 底层 pyarrow Table
    
    # list_flatten 是 C++ 实现，一次性展平
    tokens_flat = pc.list_flatten(table.column('tokens'))
    print(f"  Flattened: {len(tokens_flat):,} tokens")
    
    # 转 numpy 并截断
    print("  Converting to numpy...")
    all_tokens = tokens_flat.to_numpy(zero_copy_only=False).astype(np.uint16)
    
    if len(all_tokens) > target:
        all_tokens = all_tokens[:target]
    
    # 写入文件
    print(f"  Writing {len(all_tokens):,} tokens to disk...")
    all_tokens.tofile(token_file)
    
    # 写 meta
    with open(output_dir / "meta.txt", 'w') as f:
        f.write(f"total_tokens={len(all_tokens)}\n")
        f.write(f"dtype=uint16\n")
    
    print(f"\nDone! {token_file} ({token_file.stat().st_size/1e9:.2f} GB, {len(all_tokens):,} tokens)")


if __name__ == "__main__":
    main()