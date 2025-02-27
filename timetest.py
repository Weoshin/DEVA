import torch
from transformers import BertTokenizer, BertModel
import time

# 初始化tokenizer和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 准备输入数据
sentence = "The speaker made such an expression: raise upper lid, raise inner brow, raise outer brow , part lip."
inputs = tokenizer(sentence, return_tensors="pt", max_length=50, truncation=True, padding="max_length")

# 将模型设置为评估模式
model.eval()

# 开始计时
start_time = time.time()

# 使用BERT模型编码句子
with torch.no_grad():  # 禁用梯度计算，减少内存和计算资源的使用
    outputs = model(**inputs)

# 结束计时
end_time = time.time()

# 计算并打印编码所需时间
encoding_time = end_time - start_time
print(f"Encoding time: {encoding_time:.4f} seconds")

# 如果需要，可以访问编码结果
# encodings = outputs.last_hidden_state