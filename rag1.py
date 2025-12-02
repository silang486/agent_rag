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
print("处理数据，存入向量数据库...")
loader = TextLoader("bnuz_helper.txt", encoding="utf-8")
documents = loader.load()
# 切分文本 (如果是大书，chunk_size 设置为 500-1000 左右)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True} # BGE模型建议开启归一化
)

db = FAISS.from_documents(texts, embedding_model)
retriever = db.as_retriever(search_kwargs={"k": 2})
print("向量数据库构建完成")

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
ask_question(question2)