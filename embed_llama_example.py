import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModel

from huggingface_hub import login, snapshot_download
login(token="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")

# 步驟 1: 加載模型和 tokenizer
model_id = "lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

# 確保模型在 eval 模式
model.eval()

# 步驟 2: 輸入文本並進行編碼
text = """將法律條文轉換為可以匹配日常問題的表示方式。
        需要考慮：
        1. 將專業法律用語對應到一般民眾的日常用語
        2. 識別法條中描述的情境和實際案例的關聯
        3. 保留法條的核心概念和關鍵要素
        4. 連結相似的法律概念和情境
        請轉換以下法條：
        家事事件之調解，就離婚、終止收養關係、分割遺產或其他得處分之事項，經當事人合意，並記載於調解筆錄時成立。但離婚及終止收養關係之調解，須經當事人本人表明合意，始得成立。 2   前項調解成立者，與確定裁判有同一之效力。 3   因調解成立有關身分之事項，依法應辦理登記者，法院應依職權通知該管戶政機關。 4   調解成立者，原當事人得於調解成立之日起三個月內，聲請退還已繳裁判費三分之二。
        """
inputs = tokenizer(text, return_tensors="pt")

# 步驟 3: 使用模型生成詞嵌入
with torch.no_grad():
    outputs = model(**inputs)

# 提取嵌入 (通常取最後一層隱藏狀態)
embeddings = outputs.last_hidden_state

# 如果需要平均池化來獲得句子嵌入
sentence_embedding = torch.mean(embeddings, dim=1).squeeze()

print("詞嵌入形狀:", embeddings.shape)
print("句子嵌入:", sentence_embedding)