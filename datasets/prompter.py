import json
import os

class Prompter():
    def __init__(self, template_name: str, verbose: bool=False):
        self._verbose = verbose
        if not template_name:
            template_name = "txt_clf"
        
        file_name = os.path.join('templates', f"{template_name}.json")
        if not os.path.exists(file_name):
            raise ValueError(f"can't read {file_name}")
        
        with open(file_name, 'r') as fp:
            self.template = json.load(fp)
        
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )
    def dia_inf_generate(self, instruction, dialog, type, output):
        res = self.template["prompt_input"].format(
            instruction = instruction,
            dialog      = dialog,
            type        = type,
            output      = output
        )
        if self._verbose:
            print(res)
        return res
    def dia_gen_generate(self, dialog, personas, emotion, situation, act):
        res = self.template["prompt_input"].format(
            personas  = personas,
            dialog    = dialog,
            emotion   = emotion,
            situation = situation,
            act       = act
        )
        if self._verbose:
            print(res)
        return res
    def er_abs_generate(self, instruction, input, emotion, cause):
        res = self.template["prompt_input"].format(
            instruction = instruction,
            input       = input,
            emotion     = emotion,
            cause       = cause
        )
        if self._verbose:
            print(res)
        return res
    def dia_clf_generate(self, instruction, dialog, emotion):
        res = self.template["prompt_input"].format(
            instruction = instruction,
            dialog      = dialog,
            emotion     = emotion
        )
        if self._verbose:
            print(res)
        return res
    def txt_clf_generate(self, instruction, input, emotion):
        res = self.template["prompt_input"].format(
            instruction = instruction,
            input       = input,
            emotion     = emotion
        )
        if self._verbose:
            print(res)
        return res
    
if __name__ == '__main__':
    prompter = Prompter('dia_gen', verbose=True)

    prompt = prompter.dia_gen_generate(
        personas = "123",
        emotion = "none",
        situation = "789",
        act = "1111",
        dialog = "[user]: 123\n[ai]: 456"
    )
