import json
import os
import time
from datetime import datetime

import random
# 3.13 -> 3.52
import gradio as gr
import torch
from peft import PeftModel, prepare_model_for_int8_training
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline, set_seed

model_path = '/datas/huggingface/llama/llama-7b'
load_in_8bit = False
device_map = "auto"
torch_dtype = torch.float16

do_lora = True
lora_path = '/datas/zyq/research/emo-llm/checkpoints/llama-7b-emo-llm-12k'

model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    load_in_8bit = load_in_8bit,
    torch_dtype = torch_dtype,
    device_map = device_map,
)

model.tie_weights()

if do_lora:
    model = PeftModel.from_pretrained(
        model = model,
        model_id = lora_path,
        is_trainable = False,
        torch_dtype = torch_dtype,
    )

if load_in_8bit:
    model = prepare_model_for_int8_training(model)
else:
    model.half()

tokenizer = LlamaTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path,
    add_eos_token = True
)
tokenizer.eos_token_id = 2
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.padding_side = 'left'


generation_kwargs = {
    "do_sample"           : False,
    "num_beams"           : 4,
    "num_beam_groups"     : 2,
    "temperature"         : 0.3,
    "top_p"               : 0.75,
    "top_k"               : 40,
    "max_new_tokens"      : 50,
    "num_return_sequences": 1,
    "return_full_text"    : False,
    "stop_sequence"       : ["\n"],
    "repetition_penalty"  : 1.3,
    "diversity_penalty"   : 0.1,
    "early_stopping"      : True,
    "prefix"              : "",
}

text_generator = pipeline(
    task = "text-generation",
    model = model,
    tokenizer = tokenizer,
    pad_token_id = tokenizer.pad_token_id,
    device_map = device_map
)

def gen_process(emotion, situation, act, personas):
    if len(emotion) == 0:
        emotion = "None"
    if len(situation) == 0:
        situation = "None"
    if len(act) == 0:
        act = "None"
    if len(personas) == 0:
        personas = "None"

    gen_prompt = f"The following is a conversation between an AI assistant called [ai] and a human user called [user]. The [ai] should comply with the following control information. If control information is none, please omit it.\n\n## personas:\n{personas}\n\n## emotion:\n{emotion}\n\n## situation:\n{situation}\n\n## act:\n{act}\n\n## dialog:\n"

    return gen_prompt

def gen_history(history, input, prefix):
    history = history + prefix + input + '\n'
    return history

def get_response(inputs):
    response = text_generator(text_inputs=inputs, **generation_kwargs)
    response = response[0]['generated_text']
    response = response.split("\n")[0]
    return response

def clear_process(chatbots, emotion, situation, act, personas):
    chatbots = []
    history = gen_process(emotion, situation, act, personas)
    return chatbots, history

def save_chathistory(history, id):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if id is None:
        id = "None"

    chathistory = {
        "name"   : id,
        "model"  : model_path,
        "lora"   : lora_path,
        "time"   : current_time,
        "history": history
    }
    with open('chathistory_test_v1.json', 'a+') as fp:
        json.dump(chathistory, fp, indent=4)

    print("save successful!")

def respond(message, chatbot, history):
    print(history)

    history = gen_history(history, message, prefix='[user]: ')
    response = get_response(inputs=history)
    history = gen_history(history=history, input=response, prefix='')
    response = response.replace("[ai]: ", "")

    chatbot.append((message, response))

    return "", chatbot, history
    
def create_text_generation_menus():
    gr.Markdown(f"\n # 使用指南 \n 当前模型{lora_path.split('/')[-1]} \n 请填写控制信息：personas, situation, emotion, act, 如果不填写则默认为None \n 填写完控制信息后请务必点击Generate prompt按钮 \n 在inputs按钮中与模型对话，按回车/send按钮发送 \n 点击清除可以清空聊天记录并重置prompt! \n 点击save会保存聊天记录，请尽量保存！")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                emotion = gr.Textbox(value="None", label="emotion")
                situation = gr.Textbox(value="None", label="situation")
                act = gr.Textbox(value="None", label="act")
                personas = gr.Textbox(value="None", label="personas")
                gen_button = gr.Button("Generate prompt")

                history = gr.State(value = "")

            with gr.Row():
                chatbot = gr.Chatbot(label=title, min_width=600, height=600)
                with gr.Column():
                    msg = gr.Textbox(placeholder="Enter text and press enter", label="inputs")
                    send_button = gr.Button("Send")
                    clear_button = gr.Button("Clear")
                    id_input = gr.Textbox(label="id", placeholder="fill your id")
                    save_button = gr.Button("Save")

        gen_button.click(fn=gen_process, inputs=[emotion, situation, act, personas], outputs=[history])

        msg.submit(fn=respond, inputs = [msg, chatbot, history], outputs = [msg, chatbot, history])

        send_button.click(fn=respond, inputs = [msg, chatbot, history], outputs = [msg, chatbot, history])

        save_button.click(fn=save_chathistory, inputs = [history, id_input])

        clear_button.click(fn=clear_process,inputs=[chatbot, emotion, situation, act, personas], outputs=[chatbot, history])

if __name__ == '__main__':
    # set_seed(-1)
    # emotion = 'happy'
    # situation = 'at home'
    # personas = 'i like playing basketball'
    # act = 'None'
    # history = gen_process(emotion=emotion, situation = situation, act = act, personas=personas)
    # print('# prompt: \n\n', history)
    # try:
    #     while True:
    #         inputs = input("[user]: ")
    #         history = gen_history(history, inputs, prefix='[user]: ')
    #         response = get_response(inputs=history)

    #         print(response)

    #         history = gen_history(history, response, prefix='')
    # except KeyboardInterrupt:
    #     print("\nfinish!")
    title = 'PICA demo'
    with gr.Blocks(title=title) as demo:
        with gr.Tabs():
            with gr.TabItem("Text Generation"):
                create_text_generation_menus()
            with gr.TabItem("Text emotion classification"):
                gr.Markdown("开发中...")
            with gr.TabItem("Dialog emotion classification"):
                gr.Markdown("开发中...")

    demo.queue()
    demo.launch(share=True, inbrowser=True)
    # save_chathistory(history)

# hello, how are you?