
import os

from openai import api_key

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_community.document_loaders import TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 建议：API_KEY 最好也放在函数里读，或者用环境变量，不过这里为了简单先放这
import streamlit as st

def get_api_key():
    # 优先读 Secrets (云端)
    if "DEEPSEEK_API_KEY" in st.secrets:
        return st.secrets["DEEPSEEK_API_KEY"]
    # 其次读系统文件
    if(os.environ.get("DEEPSEEK_API_KEY")!=None):
        return os.environ.get("DEEPSEEK_API_KEY")
    #读本地文件
    try:
        with open("api.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except:
        return None

DEEPSEEK_API_KEY = get_api_key()
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DB_PATH = "faiss_index_store"
FORCE_REBUILD = True#控制是否重新构建数faiss数据库的变量，true代表无论文件是否存在，重新embeding，重新构建rag数据库


def get_qa_chain():
    # 检查 Key
    api_key = get_api_key()
    if not api_key:
        st.error("未找到 API Key。")
        return None
    # 加载 Embedding 模型
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
            print(" [bnuzHelper] 数据库构建完毕！")
        except Exception as e:
            print(f" 数据处理出错: {e}")
            return None


    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 初始化 LLM
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_BASE_URL,
        temperature=0.3,
        max_tokens=1024
    )

    # 5. 定义 Prompt
    template = """你是一个北师珠（BNUZ）的新生助手。请根据下方的【参考资料】回答用户问题。

    注意：
    1. 参考资料可能包含多人的评价，请综合正反面观点。
    2. 如果资料里没有相关内容，请诚实说不知道，不要编造。

    【参考资料】：
    {context}

    用户问题：{question}
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    return qa_chain

if __name__ == "__main__":
    print("正在进行单元测试...")
    chain = get_qa_chain()
    if chain:
        res = chain.invoke({"query": "刘凯老师怎么样？"})
        print(res['result'])