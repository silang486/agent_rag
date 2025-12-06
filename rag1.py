import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from langchain_community.document_loaders import TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
with open("api.txt", "r", encoding="utf-8") as f:
    API_KEY=f.read()

DEEPSEEK_API_KEY = API_KEY
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DB_PATH = "faiss_index_store"
# 是否强制重新构建数据库？
# 如果你修改了 ai_tutorial.txt，请把这里改为 True 运行一次，然后改回 False
FORCE_REBUILD = False
print("⬇ 正在加载 Embedding 模型...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
# 判断本地是否已经有数据库文件
if os.path.exists(DB_PATH) and not FORCE_REBUILD:
    print(f" 发现本地向量数据库: {DB_PATH}")
    print("⚡ 正在直接加载，跳过数据处理...")
    # 【关键】allow_dangerous_deserialization=True 是必须的
    # 因为 pickle 文件理论上不安全，但这是我们自己生成的，所以可以信任
    db = FAISS.load_local(
        DB_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    print("✅ 本地数据库加载成功！")
else:
    print(" 未发现本地数据库 或 要求强制重构，开始处理数据...")
    # --- 原有的数据处理流程 ---
    try:
        loader = TextLoader("bnuz_helper.txt", encoding="utf-8")
        documents = loader.load()
    except Exception as e:
        print(f"⚠️ 读取文件失败: {e}，使用测试数据")
        from langchain_core.documents import Document

        documents = [Document(page_content="Agent的核心架构...")]  # (此处省略长文本)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)

    print(f"⚡ 正在计算向量并构建索引 (共 {len(texts)} 个片段)...")
    db = FAISS.from_documents(texts, embedding_model)

    # --- 保存到本地 ---
    print(f"💾 正在保存数据库到: {DB_PATH} ...")
    db.save_local(DB_PATH)
    print("✅ 数据库构建并保存完毕！")

# 创建检索器
retriever = db.as_retriever(search_kwargs={"k": 2})

# 初始化 DeepSeek 模型

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    temperature=0.1, # 教学场景温度低一点，保持准确
    max_tokens=1024
)
def ask_question(question):
    # 构建问答链并测试
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True # 返回引用的素材来源
    )
    question = question

    print(f"\n向 DeepSeek 提问: {question}")
    response = qa_chain.invoke({"query": question})
    print("DeepSeek 回答:")
    print(response['result'])
    print(" 参考的素材片段:")
    for doc in response['source_documents']:
        print(f"[内容]: {doc.page_content}...")
question1="北师珠的人工智能专业如何？"
question2="刘凯老师讲课如何？考试严格吗？"
print('请输入提问：')
question=input()
ask_question(question)