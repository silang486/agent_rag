
import os
import torch
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
FORCE_REBUILD = False#控制是否重新构建数faiss数据库的变量，true代表无论文件是否存在，重新embeding，重新构建rag数据库


def inject_metadata_to_content(docs):
    for doc in docs:
        header_path = []
        # 获取各级标题
        if "Category" in doc.metadata: header_path.append(doc.metadata["Category"])  # 对应 #
        if "Topic" in doc.metadata: header_path.append(doc.metadata["Topic"])  # 对应 ##
        if "SubTopic" in doc.metadata: header_path.append(doc.metadata["SubTopic"])  # 对应 ###
        if "Item" in doc.metadata: header_path.append(doc.metadata["Item"])  # 对应 ####

        # 拼接成字符串，例如： "专业建议 > 人工智能 > 老师锐评 > 刘凯"
        path_str = " > ".join(header_path)
        doc.page_content = f"【当前主题：{path_str}】\n{doc.page_content}"
    return docs


def get_qa_chain():
    # --- A. 初始化环境 ---
    api_key = get_api_key()
    if not api_key:
        st.error("❌ 未找到 API Key！请配置 DEEPSEEK_API_KEY")
        return None

    # --- B. 加载 Embedding (智能硬件检测) ---
    print(" [bnuzHelper] 正在加载模型...")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        #print("检测到 NVIDIA 显卡，已启用 CUDA 加速")
    elif torch.backends.mps.is_available():
        device = "mps"
        #print("检测到 Mac M芯片，已启用 MPS 加速")
    # else:
    #     print("未检测到硬件，使用 CPU ")

    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        st.error(f" Embedding 模型加载失败: {e}")
        return None
    # try:
    #     # 使用魔搭下载
    #     print("⬇ [bnuzHelper] 正在从魔搭社区加载模型...")
    #     model_dir = snapshot_download('AI-ModelScope/bge-small-zh-v1.5')
    #     print(f"✅ 模型已加载到本地路径: {model_dir}")
    #
    #     embedding_model = HuggingFaceEmbeddings(
    #         model_name=model_dir,
    #         model_kwargs={'device': device},
    #         encode_kwargs={'normalize_embeddings': True}
    #     )
    # except Exception as e:
    #     st.error(f"❌ Embedding 模型加载失败: {e}")
    #     return None

    # --- C. 数据库构建/加载逻辑 ---
    db = None
    # 检查是否需要从头构建
    # 如果强制重构(True) 或者 本地没有数据库文件夹，则进入构建流程
    if FORCE_REBUILD or not os.path.exists(DB_PATH):
        print("🔄 [bnuzHelper] 开始处理数据并构建知识库...")

        if not os.path.exists("bnuz_helper.txt"):
            st.error(" 找不到源数据 bnuz_helper.txt，无法构建数据库desuwa！")
            return None

        try:
            with open("bnuz_helper.txt", "r", encoding="utf-8") as f:
                file_content = f.read()
            # 必须和你的文档结构 (#, ##, ###, ####) 对应
            headers_to_split_on = [
                ("#", "Category"),  # 一级标题
                ("##", "Topic"),  # 二级标题
                ("###", "SubTopic"),  # 三级标题
                ("####", "Item")  # 四级标题 (如：刘凯)
            ]


            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            md_header_splits = markdown_splitter.split_text(file_content)

            # 【关键修改 2】调用元数据注入函数
            # 这一步至关重要，否则 LLM 不知道这一段话是在评价谁
            md_header_splits = inject_metadata_to_content(md_header_splits)

            # 2. 字符级二次切分 (防止某一节内容过长)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_documents(md_header_splits)

            #print(f"✂ [bnuzHelper] 共切分为 {len(texts)} 个片段，正在向量化...")

            # 3. 生成向量数据库
            db = FAISS.from_documents(texts, embedding_model)

            # 尝试保存 (魔搭社区某些环境可能是只读的，加个 try 保护)
            try:
                db.save_local(DB_PATH)
                print("💾 [bnuzHelper] 数据库已保存到本地")
            except Exception as e:
                print(f"⚠数据库保存失败(可能是权限问题，不影响本次运行): {e}")

        except Exception as e:
            st.error(f"数据处理过程出错: {e}")
            return None
    else:
        # 直接加载旧数据
        print(f"⚡ [bnuzHelper] 发现本地数据库 {DB_PATH}，直接加载...")
        try:
            db = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"❌ 加载本地数据库失败，建议将 FORCE_REBUILD 设为 True 重试。错误: {e}")
            return None

    # --- D. 构建问答链 ---
    # k=4 保证能检索到多条不同同学的评价
    retriever = db.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base=DEEPSEEK_BASE_URL,
        temperature=0.4,  # 稍微提高一点创造力，让总结更顺滑
        max_tokens=2048,
        streaming = True
    )

    # 【关键修改 3】使用“舆情分析师”提示词
    template = """你是一个北师大（BNU）的资深学长/学姐助手。
       你现在的任务是整理并回答关于学校的问题。

       【参考资料】中可能包含了来自不同学生的【多条评价】（用列表形式呈现）。
       请你充当“舆情分析师”的角色：
       1. 不要只机械地复述某一个人的话。
       2. 如果大家观点一致，就总结这个共识。
       3. 如果观点冲突（例如有人说好，有人说坏），请客观地把两边的观点都列出来（例如：“关于给分，有同学觉得...但也有同学认为...”）。
       4. 保持回答的条理性，分点作答。
       5. 遇到不知道的问题时请直接回答不知道，并且对现在知识库有限（目前只有开发者提供了基于他个人的知识构建的知识库，只有ai专业和他口味的食堂的部分信息）并询问是否愿意在侧边栏留下反馈；当遇到与学校生活无关的问题时，请重复一遍自己的职能后委婉表达拒绝回答
       6. 回答不要超出原本问题的内容。不要回答一些无关的信息。例如提问“你好”，简单回复即可，或者推荐几个问题。不要回答一些无关的信息
       7. 如果有北京校区的同学来提问，或者有关于北京校区的提问，请委婉的表示由于开发者来自珠海校区，知识和UI设计都以BNUZ先入为主，关于北京校区的知识比较匮乏。并且邀请回答北京校区相关的问题，帮助更新知识库。
        
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