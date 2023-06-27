[**中文**](./README.md) | [**English**](./README_EN.md)

<p align="center" width="100%">
<a href="https://github.com/NEU-DataMining/Emo-LLM" target="_blank"><img src="this_one.png" alt="EMO_LLM" style="width: 60%; min-width: 300px; display: block; margin: auto;"></a>
</p>

# Emo-LLM: A comprehensive language model with satisfying empathetical ability

This project collected more than 30 data sets from five fields, including text sentiment analysis, dialogue sentiment analysis, personality dialogue generation, empathic dialogue, and emotion-reason pairs extraction. After comparison and screening, 23 of them were retained. Meanwhile, multi-task transformation was carried out on the data sets and 35 different versions of the data sets were finally obtained, which were used for training in five different tasks: Text emotion classification, dialogue emotion classification, emotion cause extraction, controllable dialogue generation, dialogue situation reasoning. And then with... as the base, pass the integrated dataset to the training phaze. Concretely, ... . The empathy task... Among other tasks,...

### Update

No update for now.

### Quick Start

Please follow the ensuing steps

```
bash emo_llm
ssh victory/of/NEU-Data_Mining
```

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
