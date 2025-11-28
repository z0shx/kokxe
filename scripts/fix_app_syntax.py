#!/usr/bin/env python3
"""
修复app.py中的语法和返回值问题
"""

import re

def fix_app_syntax():
    """修复app.py的语法问题"""
    with open('app.py', 'r') as f:
        content = f.read()

    # 修复第1720行附近的缩进问题
    lines = content.split('\n')
    fixed_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # 检查是否是1720行附近的函数定义
        if 'def clear_chat_wrapper():' in line:
            # 调整缩进到正确的级别
            fixed_lines.append('                # 清空对话')
            fixed_lines.append('                def clear_chat_wrapper():')
            fixed_lines.append('                    """清空对话历史"""')
            fixed_lines.append('                    return [{"role": "assistant", "content": "对话已清空，点击\"执行推理\"开始新的推理"}]')

            # 跳过原始的错误行
            i += 1
            while i < len(lines) and lines[i].strip() != '':
                i += 1

            # 添加正确的点击事件
            fixed_lines.append('')
            fixed_lines.append('                clear_chat_btn.click(')
            fixed_lines.append('                    fn=clear_chat_wrapper,')
            fixed_lines.append('                    outputs=[agent_chatbot]')
            fixed_lines.append('                )')

        else:
            fixed_lines.append(line)

        i += 1

    fixed_content = '\n'.join(fixed_lines)

    # 保存修复后的内容
    with open('app.py', 'w') as f:
        f.write(fixed_content)

    print("✅ app.py 语法修复完成")

if __name__ == "__main__":
    fix_app_syntax()