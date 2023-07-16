import os

class Evaluator:
    def __init__(self, type) -> None:
        assert type in ['consistency', 'fluency', 'empathy'], "error type! please check it!"
        self.type = type
        template_path = os.path.join('prompt', f"{type}.txt")
        self.template = open(template_path).read()

        self.ai_prefix = "AI: "
        self.user_prefix = "USER: "
        self.del_prefix = "你是PICA，来自东北大学数据挖掘实验室，一个具备共情能力的聊天机器人。凭借先进的深度学习算法，你能够理解人类细腻的情感，在专业领域给予需要帮助的人类专业的指导，并以同情和理解的方式回应人类的情绪，你正在与对话者进行共情对话。\n\n"

    def make_queries(self, prompts, responses, histories):
        def make_query(prompt, response, history):
            conv = ""
            for his in history:
                conv += self.user_prefix + his[0] + '\n' + self.ai_prefix + his[1] + '\n'
            conv += self.user_prefix + prompt

            template = self.template.replace(
                '{{conversation history}}', conv
            ).replace(
                '{{response}}', self.ai_prefix + response
            ).replace(
                self.del_prefix, ""
            )

            return template
        
        queries = []
        for history, prompt, response in zip(histories, prompts, responses):
            query = make_query(
                prompt=prompt,
                response=response,
                history=history
            )

            queries.append(query)
        
        return queries
    
