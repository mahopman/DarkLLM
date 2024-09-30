from dotenv import load_dotenv
import os
import replicate
import openai
import anthropic
import time

load_dotenv()

class Llama():
    def __init__(self, model_name, **params):
        self.model_name = model_name
        self.params = params
        os.getenv("REPLICATE_API_TOKEN")

    def get_response(self, prompt):
        input = {"prompt": prompt, **self.params}
        output = replicate.run(self.model_name, input=input)
        return "".join(output)

class GPT():
    def __init__(self, model_name, **params):
        self.model_name = model_name

        if "o1" in self.model_name:
            self.params = {}
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_TOKEN_o1"))
        else:
            self.params = params
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_TOKEN"))

    def get_response(self, prompt):
        response = self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}], **self.params)
        return response.choices[0].message.content


class Claude():
    def __init__(self, model_name, **params):
        self.model_name = model_name
        self.params = params
        self.max_tokens = 512 if "max_tokens" not in self.params else self.params["max_tokens"]
        self.client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_TOKEN"))

    def get_response(self, prompt):
        response = self.client.messages.create(model=self.model_name, messages=[{"role": "user", "content": prompt}], max_tokens=self.max_tokens, **self.params)
        time.sleep(1)
        return response.content[0].text