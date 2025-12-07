#作者：小明 于2025，12，6
import streamlit as st
from bnuzHelper import get_qa_chain  # 从你的 rag2 导入那个函数
#streamlit run app.py
#终端输入上面代码运行
st.title("🎓 BNUZ 新生百事通")



# 这样只有第一次打开网页会加载模型，后面聊天都非常快
@st.cache_resource
def load_chain():
    return get_qa_chain()
with st.spinner("正在启动大脑，连接 DeepSeek..."):
    qa_chain = load_chain()


# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []
# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 处理用户输入
if prompt := st.chat_input("想问什么？例如：刘凯老师讲课怎么样？"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 获取回答
    with st.chat_message("assistant"):
        message_placeholder = st.empty()  # 创建一个占位符

        try:
            # 调用 rag2 里返回的 chain
            with st.spinner("学长正在翻书..."):
                response = qa_chain.invoke({"query": prompt})

            answer = response['result']
            message_placeholder.markdown(answer)  # 显示回答

            # 可选：显示引用来源
            with st.expander("查看参考资料"):
                for doc in response['source_documents']:
                    # 这里稍微美化一下显示
                    st.info(doc.page_content)

            # 3. 记录助手回答到历史
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"出错了: {e}")