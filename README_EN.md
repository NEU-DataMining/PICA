[**中文**](./README.md) | [**English**](./README_EN.md)

<p align="center" width="100%">
<a href="https://github.com/NEU-DataMining/Emo-LLM" target="_blank"><img src="logo.png" alt="EMO_LLM" style="width: 35%; min-width: 300px; display: block; margin: auto;"></a>
</p>

# PICA: A comprehensive language model with satisfying empathetical ability

This project collected more than 30 data sets from five fields, including text sentiment analysis, dialogue sentiment analysis, personality dialogue generation, empathic dialogue, and emotion-reason pairs extraction. After comparison and screening, 23 of them were retained. Meanwhile, multi-task transformation was carried out on the data sets and 35 different versions of the data sets were finally obtained, which were used for training in five different tasks: Text emotion classification, dialogue emotion classification, emotion cause extraction, controllable dialogue generation, dialogue situation reasoning. And then with... as the base, pass the integrated dataset to the training phaze. Concretely, ... . The empathy task... Among other tasks,...
We named the resulted modal PICA, which means magpie in Latin, because we hope the model can use empathic utterances to sweep out the haze over users, as if they come across inspiring magpies. This is what prompts us to pursuit our work.  

### Update

&#x1F5D3; **2023.7.11** The data of a model finely tuned from a small number of self-built empathy datasets (Chinese) are published.

&#x1F43E; **2023.6.29** The model obtained by the joint fine-tuning of multi-task data sets is realized, and the model effect is tested internally.

### Quick Start

Please follow the ensuing steps

```
bash emo_llm
ssh victory/of/NEU-Data_Mining
```
### Case Study

1️⃣ test on consult of abstract problems

###### user：
> 我总是对自己的能力感到怀疑，做什么事情都会犹豫，然后失去最好的去做的机会，之后又非常懊悔，你说该怎么办？
###### PICA:
> 感到犹豫和拖延是很正常的事，特别是在面对重要决策时。首先，你可以尝试分解任务，将它们分解成为更小的步骤，这样可以帮助你更容易地完成任务。
###### user：
> 哦，我明白了，通过分解成更小的步骤，然后再努力去做，不要拖延，是吗？可是有时候，我会觉得就算做了也没有用，对自己未来不抱希望了，没有做事的动力，该怎么办呢？
###### PICA:
>有时候，我们可能会对自己没有动力而感到沮丧，但是请相信，你所做的每一个步骤都会为未来做好准备。同时，你可以寻找支持和鼓励的人，这样可以让你更有动力去完成任务。
###### user：
>谢谢你的帮助。当看到身边的人很优秀而自己比较落后，该怎么才能继续坚持努力，不要打乱自己的节奏呢？
###### PICA:
>每个人的人生轨迹不同，没有必要与人比较。重要的是，你能够实现自己的目标，而不是只是与他人的差距。记住，每个人都有自己的节奏和目标，你的努力也会得到回报

The model can identify the emotion accurately and give positive guidance.

2️⃣ test on consult of concrete problems

###### user：
>我现在处于高二的暑假，需要补课。我的妈妈要我同时参加家教和集体补课，我的爸爸说这样就没有整理复习的时间了，家教针对性强些，只参加家教课，老师说集体补课学习氛围更好，只需要集体补课就够了。我自己一点注意也没有，你说该怎么办？
###### PICA:
>我明白你的困惑和纠结。 在这种情况下，你可以考虑与你的妈妈和爸爸进行沟通，告诉他们你的困惑和需要调整的学习计划。你可以提出你希望专注于学习的内容，并提供一些具体的建议。同时，你也可以寻求其他学生或老师的帮助，获取更多的学习资源和支持。
###### user：
>其实我是很想两个都参加的，可是怕自己给家庭造成经济负担，然后还是没学好让父母失望。
###### PICA:
>这是一个很好的问题！你可以尝试与你的妈妈和爸爸讨论一下，看看他们是否能够理解你的困境，并提供一些解决方案。同时，你也可以考虑与同学或老师建立良好的合作关系，互相帮助和支持。这样，你可以更好地平衡学习和家庭的责任。
###### user：
>你能给我些鼓励，让我坚持自己的想法，并且为即将到来的高考好好拼搏一番吗？
###### PICA:
>当然可以！你已经迈出了第一步，相信自己能够实现目标。记住，成功需要坚持和努力。相信自己的能力，相信你所做出的决定，继续朝着自己的目标努力前进。相信自己，你一定能够取得好成绩！

The model's understanding of events is not enough, and it remains at the level of word understanding (such as mechanical capture of the word economic burden, and then use the expression of "responsibility" to reply), there is a certain sense of violation, but it can highlight the characteristics of problem solving and empathy.

### Presentation

The following video shows the performance of our emo_LLM in application.

### Dataset

The data sets used include public data sets from various aspects in the field of sentiment analysis, and the final data set is obtained after screening, task-oriented transformation and format integration.

#### Task-oriented Transformation

The transformation details and the instructions.

#### Integration

All data sets are integrated into the following format.
* 'task type' : type of training task to which they belong 
* 'dataset' : name of the dataset used 
* 'instruction': instruction
* ' instances' : training sample 
* 'is classification' : distinguish between generation tasks and classification tasks 
* 'name' : the name of the task done by the specific data set

### Task Description

Emo-LLM is mainly trained in the following five tasks, for inspiration of language knowledge and empathic capacity:

#### Text emotion classification

In this task, given a text and candidate emotion labels, the model chooses the appropriate emotion label to get the ability to classify the emotion in the text.

#### Dialogue emotion classification

This task gives the historical conversation, the current conversation, the candidate emotion labels, and lets the model choose the appropriate emotion label so that the model can understand and recognize the emotion in the conversation.

#### Emotion cause extraction

In this task, given a text or dialogue, find the cause of emotion in it, help the model deeply understand the emotion in the sentence and its cause, and further improve the model's emotion analysis ability.

#### Controllable dialogue generation

This task gives a conversation history, gives control information, generates replies, and enables the model to generate the required conversations based on the control information.

#### Dialogue situation inference

This task gives a dialogue and deduces the background of the dialogue, so that the model can deduce the information of the dialogue based on the dialogue content and context, so as to improve the intelligent interaction effect of the dialogue system.

### Training

#### Training architecture and details

#### Features

* **Modular design of output:**
* **Emotion and personality adaptation in iteration:**

### Evaluation

#### Baseline

#### Result and analyse

### Participants

### Acknowledge

### Citation

```
@misc{    ,
      title={   },
      author={   },
      year={2023},
      eprint={   },
      archivePrefix={arXiv},
      primaryClass={    }
}
```
