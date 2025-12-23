import re
import os
def clean_txt(input_path,output_path):
    def clean_qa_keep_labels(input_path, output_path):
        print(f"🧹 正在读取文件: {input_path}")

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print("❌ 找不到源文件，请确认文件名！")
            return

        cleaned_lines = []

        # 定义分类关键词，自动加二级标题 ##
        categories = ["基建类", "学业类", "娱乐类", "食宿类", "其他"]

        for line in lines:
            original_line = line.strip() # 去除首尾空格

            if not original_line:
                cleaned_lines.append("") # 保留空行
                continue

            # 1. 处理分类标题 (基建类 -> ## 基建类)
            # 只要行首是以这些词开头，就加 ##
            is_category = False
            for cat in categories:
                if original_line.startswith(cat):
                    cleaned_lines.append(f"\n## {original_line}")
                    is_category = True
                    break
            if is_category:
                continue

            # 2. 处理问题 Q (Q1: xxx -> #### Q1: xxx)
            # 匹配逻辑：以 Q 开头，后面跟数字或冒号
            if re.match(r'^Q\d*[:：\s]', original_line):
                cleaned_lines.append(f"#### {original_line}")

            # 3. 处理回答 A (A1: xxx -> - A1: xxx)
            # 匹配逻辑：以 A 开头，后面跟数字、引号或冒号
            # 【关键修改】：这里保留了 original_line，只是在前面加 "- "
            elif re.match(r'^A\d*[\'\"’]*[:：\s]', original_line):
                cleaned_lines.append(f"- {original_line}")

            # 4. 处理补充 (补充：xxx -> - 补充：xxx)
            elif re.match(r'^(补充|补@充)', original_line):
                cleaned_lines.append(f"- {original_line}")

            # 5. 其他文本，原样保留
            else:
                cleaned_lines.append(original_line)

        # 写入文件
        final_content = "# BNUZ 2024 新生答疑汇总\n\n" + "\n".join(cleaned_lines)

        # 最后再做一次空行整理，防止空行太多
        final_content = re.sub(r'\n{3,}', '\n\n', final_content)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)

        print(f"✅ 转换完成！保留了编号格式。已保存为: {output_path}")


def clean_course_data(input_file, output_file):
    print(f"🧹 正在读取课程表格: {input_file}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("❌ 找不到源文件！请把那一长串文本保存为 raw_courses.txt")
        return

    cleaned_content = []

    # 定义要过滤的无意义词汇
    FILTER_KEYWORDS = ["蹲", "蹲蹲", "蹲一个", "蹲蹲蹲"]

    # 假设第一行是标题，跳过
    # 如果数据里包含标题行 "序号	首字母...", 可以在循环里判断

    current_category = ""

    for line in lines:
        line = line.strip()
        if not line: continue

        # 1. 尝试按制表符(Tab)或多个空格分割
        # 你的数据看起来是从 Excel 复制的，通常是 Tab 分隔
        parts = re.split(r'\t+', line)

        # 如果分割失败，尝试用大空格
        if len(parts) < 3:
            parts = re.split(r'\s{2,}', line)

        # 简单的格式校验：至少要有 [序号, 字母, 课程名, 老师]
        if len(parts) < 4:
            continue

        # 2. 提取字段
        # 序号 = parts[0] (不重要)
        # 字母 = parts[1] (不重要)
        course_name = parts[2].strip()
        teacher_name = parts[3].strip()

        # 3. 提取评价 (从第4列开始都是评价)
        comments = parts[4:]

        # 4. 过滤无效评价 & 清洗
        valid_comments = []
        for c in comments:
            c = c.strip()
            # 过滤空字符串
            if not c: continue
            # 过滤 "蹲"
            if any(k in c for k in FILTER_KEYWORDS) and len(c) < 5:
                continue  # 如果只是个“蹲”，跳过；如果是“蹲一个好老师”，保留

            valid_comments.append(c)

        # 如果没有有效评价，也跳过（或者只保留课程名）
        # 这里我们选择保留，哪怕没评价，至少知道有这门课

        # 5. 生成 Markdown 格式
        # 格式：#### 课程名 (老师名)
        #       - 评价1
        #       - 评价2

        # 稍微处理一下课程名里的特殊符号
        course_name = course_name.replace("《", "").replace("》", "")

        entry = f"#### {course_name} ({teacher_name})\n"

        if valid_comments:
            for comment in valid_comments:
                entry += f"- {comment}\n"
        else:
            entry += "- (暂无详细评价)\n"

        cleaned_content.append(entry)

    # 写入文件
    final_content = "# BNUZ 选课避雷指南 (清洗版)\n\n## 课程评价汇总\n\n" + "\n".join(cleaned_content)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_content)

    print(f"✅ 转换完成！已保存为: {output_file}")



if __name__ == "__main__":
    # 记得把你那一大段文本保存为 raw_qa.txt
    #clean_txt("raw_qa.txt", "bnuz_qa_v2.txt")
    clean_course_data("raw_courese_xiaoqu.txt", "bnuz_courses_fenxiao_cleaned.txt")