
import json
from typing import List

# === 分词器定义 ===
# 这个 BPETokenizer 类是你提供的、适配你的 vocab.json 和 merges.txt 的版本。
class BPETokenizer:
    """加载你训练好的 BPE 分词器"""
    def __init__(self, vocab_path: str, merges_path: str):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        self.vocab = {}
        self.id2token = {}
        for id_str, byte_list in vocab_data.items():
            token_bytes = bytes(byte_list)
            token_id = int(id_str)
            self.vocab[token_bytes] = token_id
            self.id2token[token_id] = token_bytes
        
        self.merges = []
        with open(merges_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        bytes_list = [int(x) for x in parts]
                        mid = len(bytes_list) // 2
                        pair1, pair2 = bytes(bytes_list[:mid]), bytes(bytes_list[mid:])
                        self.merges.append((pair1, pair2))
                    except ValueError: continue
        
        self.vocab_size = len(self.vocab)
        self.pad_token_id = 0
        # 使用换行符作为结束标记，如果词汇表中没有，则默认为1
        self.eos_token_id = self.vocab.get(b'\n', 1)
    
    def encode(self, text: str) -> List[int]:
        """编码文本为 token IDs"""
        text_bytes = text.encode('utf-8')
        tokens = [bytes([b]) for b in text_bytes]
        
        for pair1, pair2 in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair1 and tokens[i+1] == pair2:
                    tokens[i] = pair1 + pair2
                    tokens.pop(i+1)
                else: i += 1
        
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                for byte in token:
                    byte_token = bytes([byte])
                    token_ids.append(self.vocab.get(byte_token, self.pad_token_id))
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """解码 token IDs 为文本"""
        tokens = [self.id2token.get(tid, b'') for tid in token_ids]
        text_bytes = b''.join(tokens)
        return text_bytes.decode('utf-8', errors='ignore')

