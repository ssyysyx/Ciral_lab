import copy
from tqdm import tqdm
import time
import json


class OpenaiClient:
    def __init__(self, keys=None, start_id=None, proxy=None):
        import openai
        from openai import OpenAI
        
        if isinstance(keys, str):
            keys = [keys]
        if keys is None:
            raise "Please provide OpenAI Key."

        self.key = keys
        self.key_id = start_id or 0
        self.key_id = self.key_id % len(self.key)
        self.api_key = self.key[self.key_id % len(self.key)]
        # 下面这一行base_url="https://api.gpts.vin/v1"是我自己加的
        # self.client = OpenAI(base_url="https://uiuiapi.com/v1", api_key=self.api_key)
        self.client = OpenAI(api_key=self.api_key)

    def chat(self, *args, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                completion = self.client.chat.completions.create(*args, **kwargs, timeout=30)
                break
            except Exception as e:
                print(str(e))
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                time.sleep(0.1)
        if return_text:
            completion = completion.choices[0].message.content
        return completion

    def text(self, *args, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                completion = self.client.completions.create(
                    *args, **kwargs
                )
                break
            except Exception as e:
                print(e)
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                time.sleep(0.1)
        if return_text:
            completion = completion.choices[0].text
        return completion


class ClaudeClient:
    def __init__(self, keys):
        from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
        self.anthropic = Anthropic(api_key=keys)

    def chat(self, messages, return_text=True, max_tokens=300, *args, **kwargs):
        system = ' '.join([turn['content'] for turn in messages if turn['role'] == 'system'])
        messages = [turn for turn in messages if turn['role'] != 'system']
        if len(system) == 0:
            system = None
        completion = self.anthropic.messages.create(messages=messages, system=system, max_tokens=max_tokens, *args, **kwargs)
        if return_text:
            completion = completion.content[0].text
        return completion

    def text(self, max_tokens=None, return_text=True, *args, **kwargs):
        completion = self.anthropic.messages.create(max_tokens_to_sample=max_tokens, *args, **kwargs)
        if return_text:
            completion = completion.completion
        return completion


class LitellmClient:
    #  https://github.com/BerriAI/litellm
    def __init__(self, keys=None):
        self.api_key = keys

    def chat(self, return_text=True, *args, **kwargs):
        from litellm import completion
        response = completion(api_key=self.api_key, *args, **kwargs)
        if return_text:
            response = response.choices[0].message.content
        return response

class GeminiClient:
    # chat用于对话，text用于文本补全
    def __init__(self, keys=None):
        self.api_key = keys

    def chat(self, messages, return_text=True, *args, **kwargs):
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        response = model.generate_content(
            convert_messages_to_prompt(messages),
            generation_config = genai.GenerationConfig(
                temperature = 0,
            )    
        )
        return response.text

class OpenaiClient_o1:
    def __init__(self, keys=None, start_id=None, proxy=None):
        import openai
        from openai import OpenAI
        
        if isinstance(keys, str):
            keys = [keys]
        if keys is None:
            raise "Please provide OpenAI Key."

        self.key = keys
        self.key_id = start_id or 0
        self.key_id = self.key_id % len(self.key)
        self.api_key = self.key[self.key_id % len(self.key)]
        # 下面这一行base_url="https://api.gpts.vin/v1"是我自己加的
        # self.client = OpenAI(base_url="https://uiuiapi.com/v1", api_key=self.api_key)
        self.client = OpenAI(api_key=self.api_key)

    def chat(self, *args, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                completion = self.client.chat.completions.create(*args, **kwargs, timeout=30)
                break
            except Exception as e:
                print(str(e))
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                time.sleep(0.1)
        if return_text:
            completion = completion.choices[0].message.content
        return completion

    def text(self, *args, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                completion = self.client.completions.create(
                    *args, **kwargs
                )
                break
            except Exception as e:
                print(e)
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                time.sleep(0.1)
        if return_text:
            completion = completion.choices[0].text
        return completion


def convert_messages_to_prompt(messages):
    #  convert chat message into a single prompt; used for completion model (eg davinci)
    prompt = ''
    for turn in messages:
        if turn['role'] == 'system':
            prompt += f"{turn['content']}\n\n"
        elif turn['role'] == 'user':
            prompt += f"{turn['content']}\n\n"
        else:  # 'assistant'
            pass
    prompt += "The ranking results of the 20 passages (only identifiers) is:"
    return prompt


def run_retriever(topics, searcher, qrels=None, k=100, qid=None):
    ranks = []
    if isinstance(topics, str):
        hits = searcher.search(topics, k=k)
        ranks.append({'query': topics, 'hits': []})
        rank = 0
        for hit in hits:
            rank += 1
            content = json.loads(searcher.doc(hit.docid).raw())
            if 'title' in content:
                content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
            else:
                content = content['contents']
            content = ' '.join(content.split())
            ranks[-1]['hits'].append({
                'content': content,
                'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
        return ranks[-1]

    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]['title']
            ranks.append({'query': query, 'hits': []})
            hits = searcher.search(query, k=k)
            rank = 0
            for hit in hits:
                rank += 1
                content = json.loads(searcher.doc(hit.docid).raw())
                if 'title' in content:
                    content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
                else:
                    content = content['contents']
                content = ' '.join(content.split())
                ranks[-1]['hits'].append({
                    'content': content,
                    'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
    return ranks


def get_prefix_prompt(query, num):
    return [{'role': 'system', # used to be system
             'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]

# 关键部分
def get_post_prompt(query, num):
    return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."

def query_transformation(query):
    # query <class 'str'>
    import os
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = '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'   

    # os.environ['OPENAI_API_KEY'] = 'sk-MAc8dDBZU9mJxF4EEe205fB6A1F54dC194C3537643B538Eb'
    os.environ['OPENAI_API_KEY'] = 'sk-yoqywCsSrqGTGaVaA1FdF9AdBc7b444aB2Cc6478DfF1132e'

    # BASE_URL = "https://api.gpts.vin/v1"
    BASE_URL = "https://uiuiapi.com/v1"
    os.environ["OPENAI_API_BASE"] = BASE_URL

    from langchain.prompts import ChatPromptTemplate

    # Decomposition
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Ensure the output language matches the language of the input question. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)

    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

    # Chain
    generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

    # Run
    question = query
    questions = generate_queries_decomposition.invoke({"question":question})

    # 去掉序号，拼接字符串
    joined_string = ''.join([question.replace(str(i) + '. ', '') for i, question in enumerate(questions, start=1)])
    query = query + joined_string
    return query

# 关键部分
def create_permutation_instruction(item=None, rank_start=0, rank_end=100, model_name='gpt-3.5-turbo'):
    query = item['query']
    # query = query_transformation(query)
    print(item['query'])
    num = len(item['hits'][rank_start: rank_end])

    max_length = 300

    messages = get_prefix_prompt(query, num)
    rank = 0
    for hit in item['hits'][rank_start: rank_end]:
        rank += 1
        content = hit['content']
        content = content.replace('Title: Content: ', '')
        content = content.strip()
        # For Japanese should cut by character: content = content[:int(max_length)]
        content = ' '.join(content.split()[:int(max_length)])
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_prompt(query, num)})
    for itemm in messages:
        print(itemm)

    return messages


def run_llm(messages, api_key=None, model_name="gpt-3.5-turbo"):
    if 'gpt' in model_name:
        Client = OpenaiClient
    elif 'o1' in model_name:
        Client = OpenaiClient_o1
    elif 'claude' in model_name:
        Client = ClaudeClient
    elif 'gemini' in model_name:
        Client = GeminiClient
    else:
        Client = LitellmClient

    agent = Client(api_key)
    response = agent.chat(model=model_name, messages=messages, temperature=1, return_text=True) #temperature used to be 0
    return response


def clean_response(response: str):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    return item


def permutation_pipeline(item=None, rank_start=0, rank_end=100, model_name='gpt-3.5-turbo', api_key=None):
    messages = create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end,
                                              model_name=model_name)  # chan
    permutation = run_llm(messages, api_key=api_key, model_name=model_name)
    item = receive_permutation(item, permutation, rank_start=rank_start, rank_end=rank_end)
    return item


def sliding_windows(item=None, rank_start=0, rank_end=100, window_size=20, step=10, model_name='gpt-3.5-turbo',
                    api_key=None):
    item = copy.deepcopy(item)
    
    end_pos = rank_end
    start_pos = rank_end - window_size
    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)
        item = permutation_pipeline(item, start_pos, end_pos, model_name=model_name, api_key=api_key)
        end_pos = end_pos - step
        start_pos = start_pos - step
    return item


def write_eval_file(rank_results, file):
    with open(file, 'w') as f:
        for i in range(len(rank_results)):
            rank = 1
            hits = rank_results[i]['hits']
            for hit in hits:
                f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                rank += 1
    return True


def main():
    from pyserini.search import LuceneSearcher
    from pyserini.search import get_topics, get_qrels
    import tempfile

    api_key = None  # Your openai key

    searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
    topics = get_topics('dl19-passage')
    qrels = get_qrels('dl19-passage')

    rank_results = run_retriever(topics, searcher, qrels, k=100)

    new_results = []
    for item in tqdm(rank_results):
        new_item = permutation_pipeline(item, rank_start=0, rank_end=20, model_name='gpt-3.5-turbo',
                                        api_key=api_key)
        new_results.append(new_item)

    temp_file = tempfile.NamedTemporaryFile(delete=False).name
    from RankGPT.trec_evall import EvalFunction

    EvalFunction.write_file(new_results, temp_file)
    EvalFunction.main(THE_TOPICS[data], temp_file)


if __name__ == '__main__':
    main()
