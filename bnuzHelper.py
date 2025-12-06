import os

# 设置环境变量 (放在最上面没问题，配置项通常可以全局)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_community.document_loaders import TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
# === 配置区域 ===
# 建议：API_KEY 最好也放在函数里读，或者用环境变量，不过这里为了简单先放这
try:
    with open("api.txt", "r", encoding="utf-8") as f:
        API_KEY = f.read().strip()
except FileNotFoundError:
    API_KEY = "你的KEY"  # 防止报错

DEEPSEEK_API_KEY = API_KEY
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DB_PATH = "faiss_index_store"
FORCE_REBUILD = False


# === 核心函数：构建并返回 QA 链 ===
def get_qa_chain():
    # 1. 加载 Embedding 模型 (把打印和加载移到函数内)
    print("⬇ [bnuzHelper] 正在加载 Embedding 模型...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. 数据库加载/构建逻辑
    db = None
    if os.path.exists(DB_PATH) and not FORCE_REBUILD:
        print(f"⚡ [bnuzHelper] 发现本地向量数据库: {DB_PATH}，直接加载...")
        db = FAISS.load_local(
            DB_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
    else:
        print("🔄 [bnuzHelper] 未发现数据库或强制重构，正在处理数据...")
        # --- 数据处理 ---
        # 建议使用 Markdown 分割 (兼容你之前的需求)
        try:
            with open("bnuz_helper.txt", "r", encoding="utf-8") as f:
                file_content = f.read()

            headers_to_split_on = [("#", "Category"), ("##", "Topic")]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            md_header_splits = markdown_splitter.split_text(file_content)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_documents(md_header_splits)

            db = FAISS.from_documents(texts, embedding_model)
            db.save_local(DB_PATH)
            print("✅ [bnuzHelper] 数据库构建完毕！")
        except Exception as e:
            print(f"❌ 数据处理出错: {e}")
            return None

    # 3. 创建检索器
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 4. 初始化 LLM
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_BASE_URL,
        temperature=0.3,
        max_tokens=1024
    )

    # 5. 定义 Prompt (解决上下文丢失问题)
    template = """你是一个北师珠（BNUZ）的新生助手。请根据下方的【参考资料】回答用户问题。

    注意：
    1. 参考资料可能包含多人的评价，请综合正反面观点。
    2. 如果资料里没有相关内容，请诚实说不知道，不要编造。

    【参考资料】：
    {context}

    用户问题：{question}
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # 6. 构建链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    return qa_chain


# === 测试代码 ===
# 加上这就只有直接运行这个文件时才跑，被 import 时不跑
if __name__ == "__main__":
    print("正在进行单元测试...")
    chain = get_qa_chain()
    if chain:
        res = chain.invoke({"query": "刘凯老师怎么样？"})
        print(res['result'])