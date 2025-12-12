import os
import torch
from openai import api_key
from modelscope import snapshot_download

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_community.document_loaders import TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

import streamlit as st


def get_api_key():
    # 优先读 Secrets (云端)
    # if "DEEPSEEK_API_KEY" in st.secrets:
    #     return st.secrets["DEEPSEEK_API_KEY"]
    # 其次读系统环境变量
    if (os.environ.get("DEEPSEEK_API_KEY") != None):
        return os.environ.get("DEEPSEEK_API_KEY")
    # 最后读本地文件
    try:
        with open("api.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except:
        return None


DEEPSEEK_API_KEY = get_api_key()
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DB_PATH = "faiss_index_store"
FORCE_REBUILD = True


def inject_metadata_to_content(docs):
    for doc in docs:
        header_path = []
        if "Category" in doc.metadata: header_path.append(doc.metadata["Category"])
        if "Topic" in doc.metadata: header_path.append(doc.metadata["Topic"])
        if "SubTopic" in doc.metadata: header_path.append(doc.metadata["SubTopic"])
        if "Item" in doc.metadata: header_path.append(doc.metadata["Item"])

        path_str = " > ".join(header_path)
        doc.page_content = f"【当前主题：{path_str}】\n{doc.page_content}"
    return docs


def get_qa_chain():
    # --- A. 初始化环境 ---
    api_key = get_api_key()
    if not api_key:
        st.error("❌ 未找到 API Key！请配置 DEEPSEEK_API_KEY")
        return None

    # --- B. 加载 Embedding (智能硬件检测 + 魔搭下载) ---
    print("⬇ [bnuzHelper] 正在准备 Embedding 模型...")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("🚀 检测到 NVIDIA 显卡，已启用 CUDA 加速")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("🍎 检测到 Mac M芯片，已启用 MPS 加速")

    embedding_model = None  # 初始化变量
    try:
        # 使用魔搭下载
        print("⬇ [bnuzHelper] 正在从魔搭社区加载模型...")
        model_dir = snapshot_download('AI-ModelScope/bge-small-zh-v1.5')
        print(f"✅ 模型已加载到本地路径: {model_dir}")

        embedding_model = HuggingFaceEmbeddings(
            model_name=model_dir,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        st.error(f"❌ Embedding 模型加载失败: {e}")
        return None

    # --- C. 数据库构建/加载逻辑 ---
    db = None
    if FORCE_REBUILD or not os.path.exists(DB_PATH):
        print("🔄 [bnuzHelper] 开始处理数据并构建知识库...")

        if not os.path.exists("bnuz_helper.txt"):
            st.error("❌ 找不到源数据 bnuz_helper.txt！")
            return None

        try:
            with open("bnuz_helper.txt", "r", encoding="utf-8") as f:
                file_content = f.read()

            headers_to_split_on = [
                ("#", "Category"),
                ("##", "Topic"),
                ("###", "SubTopic"),
                ("####", "Item")
            ]

            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            md_header_splits = markdown_splitter.split_text(file_content)

            # 注入元数据
            md_header_splits = inject_metadata_to_content(md_header_splits)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_documents(md_header_splits)

            print(f"⚡ 正在向量化 {len(texts)} 个片段...")
            db = FAISS.from_documents(texts, embedding_model)

            try:
                db.save_local(DB_PATH)
                print("💾 数据库已保存")
            except Exception as e:
                print(f"⚠ 保存失败 (不影响运行): {e}")

        except Exception as e:
            st.error(f"❌ 数据处理出错: {e}")
            return None
    else:
        print(f"⚡ [bnuzHelper] 直接加载本地数据库: {DB_PATH}")
        try:
            db = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"❌ 加载失败: {e}")
            return None

    # --- D. 构建问答链 ---
    retriever = db.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base=DEEPSEEK_BASE_URL,
        temperature=0.4,
        max_tokens=2048
    )

    template = """你是一个北师珠（BNUZ）的资深学长/学姐助手。
       你现在的任务是整理并回答关于学校的问题。

       【参考资料】中可能包含了来自不同学生的【多条评价】（用列表形式呈现）。
       请你充当“舆情分析师”的角色：
       1. 不要只机械地复述某一个人的话。
       2. 如果大家观点一致，就总结这个共识。
       3. 如果观点冲突，请客观地把两边的观点都列出来。
       4. 保持回答的条理性，分点作答。

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
    # 注意：直接运行这个文件时，Streamlit 的 Secrets 是读不到的
    # 除非你在本地有 api.txt 或者设置了环境变量
    chain = get_qa_chain()
    if chain:
        print("正在提问...")
        res = chain.invoke({"query": "刘凯老师怎么样？"})
        print("回答结果：")
        print(res['result'])