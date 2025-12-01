import os

from langchain_core.prompts import ChatPromptTemplate

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
loader = TextLoader("ai_tutorial.txt", encoding="utf-8")
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

llm_tutor = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    temperature=0.2, # 稍微有一点创造力用于举例
    max_tokens=2000
)

# 检查者：非常严厉，温度设为0
llm_checker = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    temperature=0.1, # 绝对理智，只看事实
    max_tokens=2000
)
tutor_prompt = ChatPromptTemplate.from_template("""
你是一名专业的AI课程辅导老师。请根据下方的【参考教材】回答学生的【问题】。
要求：
1. 引用教材中的概念进行解释。
2. 语言通俗易懂，适当举例说明。
3. 如果教材中没有相关内容，请诚实说“教材中未提及”，不要编造。

【参考教材】：
{context}

【学生问题】：
{question}
""")

# --- 角色 B: 检查者 (Checker) ---
checker_prompt = ChatPromptTemplate.from_template("""
你是一名严厉的教务主任。你的任务是审核老师给学生的回答。
请对比【参考教材】和【老师回答】，进行以下检查：
1. 准确性：老师的回答是否与教材内容一致？有无幻觉？
2. 完整性：是否遗漏了核心知识点？
3. 清晰度：解释是否容易理解？

输出要求：
- 如果回答完美，请直接输出“【通过】”。
- 如果有错误或不足，请输出“【驳回】”，并列出具体的修改建议和正确内容。

【参考教材】：
{context}

【学生问题】：
{question}

【老师回答】：
{tutor_answer}
""")


# ==========================================
# 6. 编排协作流程 (Agent Workflow)
# ==========================================

def run_collaboration(question):
    print(f"\n 新任务: {question}")

    print(" 1. 正在检索知识库...")
    docs = retriever.invoke(question)
    # 把检索到的文档合并成一个字符串
    context_text = "\n\n".join([d.page_content for d in docs])

    if not context_text:
        print("未找到相关教材内容。")
        return

    # --- 第2步: 回答者生成答案 (Tutor Agent) ---
    print("2. 回答者(Tutor)正在思考...")
    # 填充提示词
    tutor_messages = tutor_prompt.format_messages(
        context=context_text,
        question=question
    )
    # 调用大模型
    tutor_response = llm_tutor.invoke(tutor_messages).content
    print(f"\n[Tutor 初稿]:\n{tutor_response}\n")

    # --- 第3步: 检查者审核 (Checker Agent) ---
    print(" 3. 检查者(Checker)正在审核...")
    checker_messages = checker_prompt.format_messages(
        context=context_text,
        question=question,
        tutor_answer=tutor_response
    )
    checker_response = llm_checker.invoke(checker_messages).content

    print(f"\n[Checker 审核结果]:\n{checker_response}\n")

    # --- (进阶) 第4步: 简单的判断逻辑 ---
    if "【通过】" in checker_response:
        print(" 最终结果: 答案已发布给学生。")
    else:
        print(" 最终结果: 答案被驳回，需要人工或自动修正。")



# 测试一个教材里有的问题
run_collaboration("Agent的核心架构是什么？")

# 测试一个可能让它胡说八道的问题（你可以故意问一个教材里没有的）
run_collaboration("怎么煮红烧肉？")