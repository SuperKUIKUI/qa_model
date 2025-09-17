import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# 1. 读取文档并切割（保持原来的按标题切割）
with open("DeepinWiki.txt", "r", encoding="utf-8") as f:
    text = f.read()

segments = re.split(r"标题：", text)
segments = [seg.strip() for seg in segments if seg.strip()]

docs = []
for seg in segments:
    lines = seg.splitlines()
    title = lines[0]
    content = "\n".join(lines[1:])
    docs.append({"title": title, "content": content})

# 2. 加载 sentence-transformers 模型
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 3. 构建所有层级的索引
# 3.1 文档级索引（标题+内容）
doc_texts = [d["title"] + " " + d["content"] for d in docs]
doc_embeddings = embedder.encode(doc_texts, convert_to_numpy=True)
doc_index = faiss.IndexFlatL2(doc_embeddings.shape[1])
doc_index.add(doc_embeddings)

# 3.2 预构建段落级索引（为每个文档建立段落索引）
paragraph_indices = {}
paragraph_data = {}

for doc_idx, doc in enumerate(docs):
    # 按段落分割（多种分隔方式）
    paragraphs = re.split(r'\n\n+|\n•\s*|\n\d+[\.\)]\s*', doc["content"])
    paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]
    
    if not paragraphs:
        paragraphs = [doc["content"]]
    
    para_embeddings = embedder.encode(paragraphs, convert_to_numpy=True)
    para_index = faiss.IndexFlatL2(para_embeddings.shape[1])
    para_index.add(para_embeddings)
    
    paragraph_indices[doc_idx] = para_index
    paragraph_data[doc_idx] = paragraphs

# 4. 加载 QA 模型
model_path = "./tim_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0)

def find_best_sentences(question, paragraph, top_k=3):
    """句子级检索：从段落中找出最相关的几个句子"""
    # 分句
    sentences = re.split(r'[.!?。！？]+', paragraph)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    if len(sentences) <= 1:
        return paragraph
    
    # 句子向量化
    sent_embeddings = embedder.encode(sentences, convert_to_numpy=True)
    sent_index = faiss.IndexFlatL2(sent_embeddings.shape[1])
    sent_index.add(sent_embeddings)
    
    # 检索最相关句子
    q_embedding = embedder.encode([question], convert_to_numpy=True)
    D, I = sent_index.search(q_embedding, min(top_k, len(sentences)))
    
    # 按相关性排序并组合
    best_sentences = [sentences[i] for i in I[0]]
    return " ".join(best_sentences)

def multi_level_retrieval(question, top_k_docs=3, top_k_paras=2, top_k_sents=3):
    q_embedding = embedder.encode([question], convert_to_numpy=True)
    D, I = doc_index.search(q_embedding, min(top_k_docs, len(docs)))
    
    best_contexts = []
    for doc_idx in I[0]:
        doc = docs[doc_idx]
        para_index = paragraph_indices[doc_idx]
        paragraphs = paragraph_data[doc_idx]

        D_para, I_para = para_index.search(q_embedding, min(top_k_paras, len(paragraphs)))

        for rank, para_idx in enumerate(I_para[0]):  # 用 rank 对齐
            paragraph = paragraphs[para_idx]
            best_sentences = find_best_sentences(question, paragraph, top_k_sents)

            best_contexts.append({
                'title': doc['title'],
                'content': best_sentences,
                'score': float(D_para[0][rank])  # 用 rank，而不是 para_idx
            })

    best_contexts.sort(key=lambda x: x['score'])
    return best_contexts


def ask(question):
    """改进的QA函数，使用多级检索"""
    # 多级检索获取最佳上下文
    contexts = multi_level_retrieval(question)
    
    if not contexts:
        return "抱歉，没有找到相关信息。"
    
    # 尝试在前几个最佳上下文中寻找答案
    best_answer = None
    best_score = 0
    
    for context in contexts[:3]:  # 尝试前3个最佳上下文
        try:
            result = qa_pipeline(question=question, context=context['content'])
            
            # 选择置信度最高的答案
            if result['score'] > best_score:
                best_score = result['score']
                best_answer = {
                    'answer': result['answer'],
                    'score': result['score'],
                    'title': context['title'],
                    'context': context['content']
                }
        except:
            continue
    
    if best_answer and best_score > 0.1:  # 设置置信度阈值
        return f"【{best_answer['title']}】 → {best_answer['answer']} (得分: {best_answer['score']:.3f})"
    else:
        # 如果QA模型没找到答案，返回最相关的上下文
        return f"【{contexts[0]['title']}】 → 未找到确切答案，相关上下文：{contexts[0]['content'][:200]}..."

# 交互部分保持不变
if __name__ == "__main__":
    print("改进版多级检索问答系统已启动，输入 exit/quit/q 退出")
    print("-" * 50)
    
    while True:
        q = input("你: ").strip()
        if q.lower() in ["exit", "quit", "q"]:
            break
        if not q:
            continue
            
        try:
            answer = ask(q)
            print("机器人:", answer)
        except Exception as e:
            print(f"机器人: 出错了 - {str(e)}")
        
        print()  # 空行