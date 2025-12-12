#作者：小明 于2025，12，6
#streamlit run app.py
#终端输入上面代码运行
import streamlit as st
from bnuzHelper import get_qa_chain
import csv
import os
import time
import pandas as pd # 如果没有安装，记得在 requirements.txt 里加 pandas
st.title("🎓 BNU 百事通")
from langchain_community.callbacks import StreamlitCallbackHandler
# 这样只有第一次打开网页会加载模型，后面聊天都非常快
@st.cache_resource
def load_chain():
    return get_qa_chain()
with st.spinner("正在启动..."):
    qa_chain = load_chain()


# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []
# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 处理用户输入
if prompt := st.chat_input("想问什么？"):
    # 1. 显示用户输入
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. 获取回答 (流式版)
    with st.chat_message("assistant"):
        try:
            # 创建一个 Streamlit 的回调处理器
            # 它会自动在界面上显示 "Thinking..." 和打字机效果
            st_callback = StreamlitCallbackHandler(st.container())

            # 调用 invoke 时，把 callback 传进去
            with st.spinner("学长正在翻书..."):
                # 调用 invoke 时，把 callback 传进去
                response = qa_chain.invoke(
                    {"query": prompt},
                    {"callbacks": [st_callback]}
                )
            answer = response['result']
            # 注意：StreamlitCallbackHandler 已经自动打印了过程，
            # 但为了保证历史记录格式统一，我们还是拿到了最终结果。
            # 为了防止重复打印，这里不需要再 st.markdown(answer)，
            # 因为回调里已经显示过了。

            # 显示引用来源 (保持不变)
            with st.expander("查看参考资料"):
                for doc in response['source_documents']:
                    st.info(doc.page_content)

            # 3. 记录助手回答到历史
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"出错了: {e}")
# 定义反馈文件路径
FEEDBACK_CSV = "feedback_data.csv"

def save_feedback(user_input, comments):
    # 如果文件不存在，写入表头
    file_exists = os.path.isfile(FEEDBACK_CSV)

    with open(FEEDBACK_CSV, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["时间", "用户问题", "补充建议/反馈"])  # 表头

        # 写入时间、问题、反馈
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        writer.writerow([timestamp, user_input, comments])
# --- 侧边栏：反馈通道 ---
with st.sidebar:
    st.markdown("---")
    st.header("📝 没找到答案？")
    st.write("如果你发现助手答不上来，或者有更好的建议，请告诉学长！")

    # 使用 form 表单，防止每输入一个字就刷新
    with st.form("feedback_form"):
        # 既然是反馈，最好能引用刚才用户问的问题，或者让用户自己写
        feedback_q = st.text_input("未获得答案的问题描述（选填）")
        feedback_c = st.text_area("问题的详细描述/相关建议/正确答案", height=100)

        submitted = st.form_submit_button("📤 提交给学长")

        if submitted:
            if not feedback_c:
                st.warning("写点内容再提交嘛！")
            else:
                save_feedback(feedback_q, feedback_c)
                st.success("收到！感谢你的投喂，我们会尽快更新知识库！❤️")

    # --- 管理员后门 (平时折叠起来) ---
    st.markdown("---")
    with st.expander("🔐 管理员入口 (获取反馈)"):
        # 简单做个密码验证，防止同学乱点（你可以设个简单的密码，比如 bnuz666）
        admin_pwd = st.text_input("输入管理员密码", type="password")

        if admin_pwd == "bnuz666":  # <--- 修改这里的密码
            if os.path.exists(FEEDBACK_CSV):
                df = pd.read_csv(FEEDBACK_CSV)
                st.dataframe(df)  # 预览数据

                # 下载按钮
                st.download_button(
                    label="📥 下载反馈数据(CSV)",
                    data=df.to_csv(index=False).encode('utf-8-sig'),  # utf-8-sig 防止Excel乱码
                    file_name=f"feedback_{time.strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

                # 清空按钮
                if st.button("🗑️ 清空所有反馈"):
                    os.remove(FEEDBACK_CSV)
                    st.rerun()
            else:
                st.info("暂时还没有收到反馈哦~")
        elif admin_pwd:
            st.error("密码错误")