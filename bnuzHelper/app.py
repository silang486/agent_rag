#作者：小明 于2025，12，6
#streamlit run app.py 终端输入这段代码运行
import streamlit as st
import csv
import os
import time
import pandas as pd
from langchain_community.callbacks import StreamlitCallbackHandler
from bnuzHelper import get_qa_chain


# ==========================================
# 1. 核心逻辑层 (Core Logic Layer)
# 负责数据处理、模型加载，不包含任何 UI 代码
# ==========================================

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


# ==========================================
# 2. UI 组件层 (UI Component Layer)
# 负责界面渲染，不处理底层逻辑
# ==========================================

class UIComponents:
    """界面组件集合"""

    @staticmethod
    def init_session_state():
        """初始化会话状态"""
        if "messages" not in st.session_state:
            st.session_state.messages = []

    @staticmethod
    def render_chat_history():
        """渲染历史消息"""
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    @staticmethod
    def render_feedback_sidebar():
        """渲染侧边栏反馈功能"""
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

            # 管理员入口
            st.markdown("---")
            with st.expander("🔐 反馈信箱入口"):
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
        """
        [新增功能] 底部邀请标签
        功能：显示一条邀请信息，并提供一键复制的问卷链接
        设计理念：独立组件，不干扰聊天流
        """
        # 使用 container 稍微隔离一下视觉
        with st.container():
            st.markdown("---")  # 分割线
            st.caption("愿意填写问卷，给我们反馈，让它更好吗？")
            # st.code 用于显示文本，自带右上角的“复制”按钮，非常适合分享链接
            st.code("https://wj.qq.com/s2/your-survey-id", language="text")


# ==========================================
# 3. 主程序入口 (Main Controller)
# 负责组装各个模块
# ==========================================

def main():
    st.set_page_config(page_title="BNU 百事通", page_icon="🎓")
    st.title("🎓 BNU 百事通")

    # 1. 初始化
    UIComponents.init_session_state()

    # 2. 加载模型 (带 Spinner)
    with st.spinner("正在启动..."):
        qa_chain = BotEngine.load_brain()

    # 3. 渲染侧边栏
    UIComponents.render_feedback_sidebar()

    # 4. 渲染聊天历史
    UIComponents.render_chat_history()

    # 5. [新增] 渲染底部邀请 (放在输入框上方，历史记录下方)
    # 如果希望它始终在最底部，Streamlit 的机制决定了它只能在聊天流的末尾
    if len(st.session_state.messages) > 0:
        UIComponents.render_invite_footer()

    # 6. 处理用户输入循环
    if prompt := st.chat_input("想问什么？"):
        # 显示用户提问
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 生成回答
        with st.chat_message("assistant"):
            try:
                with st.spinner("学长正在翻书..."):
                    container = st.container()
                    response = BotEngine.generate_response(qa_chain, prompt, container)

                answer = response['result']

                # 显示引用源
                with st.expander("查看参考资料"):
                    for doc in response['source_documents']:
                        st.info(doc.page_content)

                # 记录历史
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # 强制刷新以更新底部的邀请栏位置
                st.rerun()

            except Exception as e:
                st.error(f"出错了: {e}")


if __name__ == "__main__":
    main()