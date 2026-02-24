import torch
from model import Language

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ 加载模型 ============
checkpoint = torch.load('poetry_checkpoint.pth', map_location=DEVICE)

word_dict = checkpoint['word_dict']
tag_dict = checkpoint['tag_dict']
len_words = len(word_dict)

model = Language(
    len_feature=checkpoint['len_feature'],
    len_words=len_words,
    len_hidden=checkpoint['len_hidden'],
    num_to_word=tag_dict,
    word_to_num=word_dict,
    strategy=checkpoint['strategy'],
    drop_out=checkpoint['drop_out']
).to(DEVICE)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"模型加载完成 | 词表大小: {len_words} | 设备: {DEVICE}")
print("=" * 50)
print("输入第一句诗，模型将为你续写剩余诗句")
print("输入 q 退出")
print("=" * 50)

while True:
    first_line = input("\n请输入第一句诗: ").strip()
    if first_line.lower() == 'q':
        print("再见！")
        break
    if not first_line:
        print("输入不能为空，请重新输入。")
        continue

    with torch.no_grad():
        poem = model.continue_poem(first_line, num_sentence=4)
        poem_str = '\n'.join([''.join(s) for s in poem])
        print(f"\n{poem_str}")
