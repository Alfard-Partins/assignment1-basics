
# ==============================================================================
#                      AI 小故事生成器 (主运行脚本)
#
# 功能: 加载一个预训练的 TinyStoriesTransformer 模型和 BPE 分词器，
#      然后根据用户输入的开头，自动生成小故事。
# ==============================================================================

# === 1. 导入必要的库 ===
import torch

# === 关键: 从你创建的模块中导入类 ===
from cs336_basics.bpe import BPETokenizer
from cs336_basics.model import TinyStoriesTransformer

# === 2. 主程序：加载模型并生成故事 ===
def main():
    # --- 用户配置区 (请根据你的文件路径修改) ---
    model_path = '/Users/saileisi/Downloads/点头/tinystories_checkpoints/best_model.pt'
    vocab_path = '/Users/saileisi/Downloads/点头/bpe_tokenizer_tiny/vocab.json'
    merges_path = '/Users/saileisi/Downloads/点头/bpe_tokenizer_tiny/merges.txt'
    
    print("--- AI 小故事生成器启动 ---")
    
    # --- 步骤 1: 加载模型和配置 ---
    print(f"⏳ 正在从 '{model_path}' 加载模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # --- 步骤 2: 初始化分词器 ---
    tokenizer = BPETokenizer(vocab_path=vocab_path, merges_path=merges_path)
    eos_token_id = tokenizer.eos_token_id
    print(f"✅ 分词器加载完成 (词汇表: {tokenizer.vocab_size}, 结束ID: {eos_token_id})")

    # --- 步骤 3: 初始化模型并加载权重 ---
    model = TinyStoriesTransformer(
        vocab_size=config['vocab_size'], d_model=config['d_model'],
        num_layers=config['num_layers'], num_heads=config['num_heads'],
        d_ff=config['d_ff'], max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() # 切换到评估模式
    
    num_params = sum(p.numel() for p in model.parameters())/1e6
    print(f"✅ 模型加载完成 (参数量: {num_params:.2f}M, 设备: {device})")
    
    # --- 步骤 4: 进入交互式生成循环 ---
    print("\n" + "="*50)
    print("✍️  开始生成故事... (输入 'quit' 或 'exit' 退出)")
    print("="*50)
    
    while True:
        prompt = input(">>> 请输入故事的开头: ")
        if prompt.lower() in ['quit', 'exit']:
            print("再见！")
            break
        
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        print("\n...模型正在创作中，请稍候...\n")
        
        generated_ids = model.generate(
            input_tensor,
            max_new_tokens=800,    # 增加最大长度，给模型足够空间
            temperature=0.9,      # 控制创造性，越小越保守
            top_k=100,              # 只在概率最高的100个词中选择
            top_p=0.9,             # 核心采样，进一步筛选
            eos_token_id=eos_token_id # 告诉模型遇到这个ID就停止
        )
        
        generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
        
        # 这是一个安全措施，以防模型生成了 <|endoftext|> 这个特殊字符串
        generated_text = generated_text.split('<|endoftext|>')[0]
        
        print("--- 📖 你的小故事 📖 ---")
        print(generated_text)
        print("\n" + "-"*30 + "\n")

# === 3. 脚本入口 ===
if __name__ == "__main__":
    main()
