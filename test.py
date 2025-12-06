from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
with open("bnuzHelper/api.txt", "r", encoding="utf-8") as f:
    API_KEY=f.read()
print(API_KEY)
# ==========================================
# 1. 配置 DeepSeek API
# ==========================================
DEEPSEEK_API_KEY = API_KEY
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# --- 数据准备阶段 (ETL) ---

# 2. 加载文档
loader = TextLoader("knowledge.txt", encoding="utf-8")
documents = loader.load()

# 3. 文本切分 (Chunking)
# 将长文档切分成 1000 字符的块，重叠 200 字符以保持上下文连贯
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# 4. 向量化并存储 (Embedding & Vector Store)
# 使用 OpenAI 的 embedding 模型，数据存存在本地 ./chroma_db 文件夹
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# --- 检索与问答阶段 (RAG) ---

# 5. 创建检索器
retriever = vectorstore.as_retriever()

# 6. 定义 Prompt 模板
template = """你是一个智能助手。请根据下面的上下文信息回答用户的问题。
如果你不知道答案，就直说不知道，不要编造。

上下文信息：
{context}

用户问题：
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 7. 定义 LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 8. 构建 RAG 链 (LangChain Expression Language - LCEL)
# 逻辑：获取问题 -> 检索上下文 -> 拼装Prompt -> 喂给LLM -> 解析输出
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 9. 运行测试
query = "公司的请假流程是怎样的？"
print(f"问题: {query}")
response = rag_chain.invoke(query)
print(f"回答: {response}")