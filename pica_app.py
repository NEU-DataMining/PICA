# -*- coding: utf-8 -*-
# @Author      : 高正杰
# @File        : pica_app.py
# @Email       : gaozhengj@foxmail.com
# @Date        : 2023/7/22 15:14
# @Description :


from transformers import AutoTokenizer, AutoModel, AutoConfig
import os
import torch
import streamlit as st
from streamlit_chat import message
import json
import re

st.set_page_config(
    page_title="PICA-V1模型",
    page_icon="👩‍🏫",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """
        -   版本：PICA-V1模型
        -   模型开发者：东北大学数据挖掘实验室
        -   本界面开发者：高正杰
        """
    }
)
st.header("PICA-V1模型")
with st.expander("ℹ️ - 关于我们", expanded=False):
    st.write(
        """
        -   版本：PICA-V1模型
        -   模型开发者：东北大学数据挖掘实验室
        -   本界面开发者：高正杰
        """)


def answer(user_history, bot_history, sample=True, top_p=0.75, temperature=0.95):
    """

    :param user_history: 用户输入的历史文本
    :param bot_history: 机器生成的历史文本
    :param sample: 是否抽样。生成任务，可以设置为True;
    :param top_p: 0-1之间，生成的内容越多样
    :param temperature:
    :return:
    """
    if len(bot_history) > 0:
        dialog_turn = 5  # 设置历史对话轮数
        if len(bot_history) > dialog_turn:
            bot_history = bot_history[-dialog_turn:]
            user_history = user_history[-(dialog_turn + 1):]

        context = "\n".join(
            [f"[Round {i + 1}]\n\n问：{user_history[i]}\n\n答：{bot_history[i]}" for i in range(len(bot_history))])
        input_text = context + f"[Round {len(bot_history) + 1}]\n\n问：" + user_history[-1] + "\n\n答："
    else:
        input_text = "[Round 1]\n\n问：" + user_history[-1] + "\n\n答："

    print(input_text)
    if not sample:
        response, history = model.chat(tokenizer, query=input_text, history=None, max_length=2048, num_beams=1,
                                       do_sample=False, top_p=top_p, temperature=temperature, logits_processor=None)
    else:
        response, history = model.chat(tokenizer, query=input_text, history=None, max_length=2048, num_beams=1,
                                       do_sample=True, top_p=top_p, temperature=temperature, logits_processor=None)

    print("模型原始输出：\n", response)
    # 规则校验，这里可以增加校验文本的规则
    response = re.sub("\n+", "\n", response)
    print('答: ' + response)
    return response


@st.cache_resource
def load_model():
    config = AutoConfig.from_pretrained("/hy-tmp/chatglm2-6b", trust_remote_code=True, pre_seq_len=128)
    model = AutoModel.from_pretrained("/hy-tmp/chatglm2-6b", config=config, trust_remote_code=True).half().quantize(4)
    CHECKPOINT_PATH = '/hy-tmp/PICA-V1'
    prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    model.to(device)
    print('Model Load done!')
    return model


@st.cache_resource
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("/hy-tmp/chatglm2-6b", trust_remote_code=True)
    print('Tokenizer Load done!')
    return tokenizer


def get_text():
    user_input = st.session_state.user_input
    if 'id' not in st.session_state:
        if not os.path.exists("./history"):
            # 创建保存用户聊天记录的目录
            os.makedirs("./history")
        json_files = os.listdir("./history")
        id = len(json_files)
        st.session_state['id'] = id

    if user_input:
        st.session_state["past"].append(user_input)
        output = answer(st.session_state.past, st.session_state.generated)
        try:
            st.session_state.generated.append(output)
        except KeyError as e:
            print("Asd")
        # 将对话历史保存成json文件
        dialog_history = {
            'user': st.session_state.past,
            'bot': st.session_state.generated
        }
        with open(os.path.join("./history", str(st.session_state['id']) + '.json'), "w", encoding="utf-8") as f:
            json.dump(dialog_history, f, indent=4, ensure_ascii=False)

    if st.session_state.generated:
        for i in range(len(st.session_state.generated)):
            # 显示用户的输入
            message(st.session_state.past[i], is_user=True, key=str(i) + '_user', avatar_style="avataaars", seed=26)
            # 显示机器人的回复
            message(st.session_state.generated[i], is_user=False, key=str(i), avatar_style="avataaars", seed=5)
    st.session_state["user_input"] = ""


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model()
    tokenizer = load_tokenizer()

    if 'generated' not in st.session_state:
        st.session_state.generated = []

    if 'past' not in st.session_state:
        st.session_state.past = []

    with st.container():
        st.text_area(label="请在下列文本框输入您的咨询内容：", value="",
                     placeholder="请输入您的求助内容，并且点击Ctrl+Enter发送信息", key="user_input", on_change=get_text)

    if st.button("清理对话缓存"):
        st.session_state.generated = []
        st.session_state.past = []
