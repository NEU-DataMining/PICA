import openai
import tenacity
import re

@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=200),
    stop=tenacity.stop_after_attempt(3),
    reraise=True)
def get_azure_response(
    url: str,
    apikey: str,
    content: str,
    _verbose: bool = False,
    temperature: float = 2,
    max_tokens: int = 5,
    top_p: float= 1,
    frequency_penalty: float=0,
    presence_penalty: float=0,
    n: int=5
):
    openai.api_type    = "azure"
    openai.api_base    = url
    openai.api_version = "2023-03-15-preview"
    openai.api_key     = apikey

    if _verbose:
        print(content)

    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages = [
            {
                "role"   : "system",
                "content": "You are an AI assistant that helps people find information."
            },
            {
                "role"   : "user",
                "content": content,
            }
        ],
        temperature       = temperature,
        max_tokens        = max_tokens,
        top_p             = top_p,
        frequency_penalty = frequency_penalty,
        presence_penalty  = presence_penalty,
        n                 = n,
    )

    all_responses = [response['choices'][i]['message']['content']
        for i in range(len(response['choices']))
    ]

    return all_responses

def parse_output(output):
    matched = re.search("^ ?([\d\.]+)", output)
    if (matched):
        try:
            score = float(matched.group(1))
        except:
            score = -1
    else:
        score = -1
    return score