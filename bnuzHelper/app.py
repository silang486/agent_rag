#作者：小明 于2025，12，6

'''
streamlit run app.py
终端输入这段代码运行
'''

import streamlit as st
import csv
import os
import time
import pandas as pd
from langchain_community.callbacks import StreamlitCallbackHandler
from bnuzHelper import get_qa_chain




class BotEngine:
    """负责 RAG 模型的加载和推理"""

    @staticmethod
    @st.cache_resource
    def load_brain():
        """加载问答链 (带缓存)"""
        return get_qa_chain()

    @staticmethod
    def generate_response(qa_chain, prompt, container):
        """生成回答"""
        # 创建回调处理器
        st_callback = StreamlitCallbackHandler(container)
        # 调用链
        response = qa_chain.invoke(
            {"query": prompt},
            {"callbacks": [st_callback]}
        )
        return response


class FeedbackService:
    """负责反馈数据的持久化 (CSV操作)"""
    FILE_PATH = "feedback_data.csv"

    @classmethod
    def save(cls, question, comments):
        """保存反馈"""
        file_exists = os.path.isfile(cls.FILE_PATH)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        try:
            with open(cls.FILE_PATH, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["时间", "用户问题", "补充建议/反馈"])
                writer.writerow([timestamp, question, comments])
            return True
        except Exception as e:
            st.error(f"保存失败: {e}")
            return False

    @classmethod
    def load_dataframe(cls):
        """加载反馈数据用于预览"""
        if os.path.exists(cls.FILE_PATH):
            return pd.read_csv(cls.FILE_PATH)
        return None

    @classmethod
    def clear_data(cls):
        """清空数据"""
        if os.path.exists(cls.FILE_PATH):
            os.remove(cls.FILE_PATH)


class UIComponents:
    """界面组件集合"""

    @staticmethod
    def init_session_state():
        if "messages" not in st.session_state:
            st.session_state.messages = []

    @staticmethod
    def render_chat_history():
        """渲染历史消息"""
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                # === [修改点 1] 渲染历史记录里的参考资料 ===
                # 检查这条消息里有没有 'sources' 字段，且不为空
                if "sources" in msg and msg["sources"]:
                    with st.expander("查看参考资料 (历史)"):
                        for doc_content in msg["sources"]:
                            st.info(doc_content)

    @staticmethod
    def render_feedback_sidebar():
        # ... (这部分代码保持不变，太长省略) ...
        # 请保留你原有的 render_feedback_sidebar 代码
        with st.sidebar:
            st.markdown("---")
            st.header("📝 没找到答案？")
            st.write("如果你发现助手答不上来，或者有更好的建议，请告诉我们！")

            with st.form("feedback_form"):
                feedback_q = st.text_input("未获得答案的问题描述（选填）")
                feedback_c = st.text_area("问题的详细描述/相关建议/正确答案", height=100)
                submitted = st.form_submit_button("📤 提交给学长")

                if submitted:
                    if not feedback_c:
                        st.warning("写点内容再提交嘛！")
                    elif FeedbackService.save(feedback_q, feedback_c):
                        st.success("收到！感谢你的投喂，我们会尽快更新知识库！❤️")

            st.markdown("---")
            with st.expander("🔐 反馈信箱"):
                admin_pwd = st.text_input("输入管理员密码", type="password")
                if admin_pwd == "bnuz666":
                    df = FeedbackService.load_dataframe()
                    if df is not None:
                        st.dataframe(df)
                        st.download_button(
                            label="📥 下载 CSV",
                            data=df.to_csv(index=False).encode('utf-8-sig'),
                            file_name=f"feedback_{time.strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                        if st.button("🗑️ 清空所有反馈"):
                            FeedbackService.clear_data()
                            st.rerun()
                    else:
                        st.info("暂无反馈数据")
                elif admin_pwd:
                    st.error("密码错误")

    @staticmethod
    def render_invite_footer():
        """底部邀请标签"""
        with st.container():
            st.markdown("---")
            st.caption("愿意填写问卷，给我们反馈，为我们提供知识，让它更好吗？")
            st.code("https://v.wjx.cn/vm/elbP0yB.aspx", language="text")


def main():
    st.set_page_config(page_title="BNU 百事通", page_icon="🎓")
    st.title("🎓 BNU 百事通")

    # 1. 初始化
    UIComponents.init_session_state()

    # 2. 加载模型
    with st.spinner("正在启动..."):
        qa_chain = BotEngine.load_brain()

    # 3. 渲染侧边栏
    UIComponents.render_feedback_sidebar()

    # 4. 渲染聊天历史 (现在它能显示历史资料了)
    UIComponents.render_chat_history()

    # 5. 渲染底部邀请
    if len(st.session_state.messages) > 0:
        UIComponents.render_invite_footer()

    # 6. 处理用户输入
    if prompt := st.chat_input("想问什么？（请点击右侧箭头发送提问，暂时不支持回车发送提问）"):

        with st.chat_message("user"):
            st.markdown(prompt)
        # 注意：这里先不要 append user message，等最后一起 append 也可以，或者像现在这样
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            try:
                with st.spinner("学长正在翻书..."):
                    container = st.container()
                    response = BotEngine.generate_response(qa_chain, prompt, container)

                answer = response['result']

                # === [修改点 2] 提取参考资料内容 ===
                source_contents = []
                if 'source_documents' in response:
                    for doc in response['source_documents']:
                        source_contents.append(doc.page_content)

                # 显示本次回答的引用源 (让用户在刷新前能看到一眼)
                if source_contents:
                    with st.expander("查看参考资料"):
                        for content in source_contents:
                            st.info(content)

                # === [修改点 3] 将资料存入 session_state ===
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": source_contents  # 关键：把资料存进去！
                })

                # 强制刷新 (刷新后，render_chat_history 会负责把上面的 sources 画出来)
                st.rerun()

            except Exception as e:
                st.error(f"出错了: {e}")


if __name__ == "__main__":
    main()