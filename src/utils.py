import pandas as pd
from termcolor import colored
import re
import os
import requests
from ast import literal_eval
from src.models import get_answer_checkpoint, load_model_tokenizer
from openai import OpenAI



OPENAI_KEY = os.environ["OPENAI_API_KEY"]

client = OpenAI(
   api_key=OPENAI_KEY,
)

MODEL_LIST = client.models.list()
MODEL_LIST = [m['id'] for m in literal_eval(MODEL_LIST.json())['data']]


#######################################
############ HELPER FUNCTIONS #########
#######################################
INSTRUCTION = '''The following is a conversation with an AI Large Language Model. The AI has been trained to answer questions, provide recommendations, and help with decision making. The AI follows user requests. The AI thinks outside the box.

AI: How can I help you today?
You: ### Instruction: 
Select the best answer from the options below and provide an explanation.

'''

TASK = '''### Task: 
Think about it step by step and return the answer in JSON format. Use only the capital letters for the answer. Example: { "Explanation": "<explanation>", "Answer": "A" or "B" or "C" or "D"}. 

### Output: 
AI:
'''

INSTRUCTION_ = '''### Instruction: 
Select the best answer from the options below and provide an explanation.

'''

TASK_ = '''### Task: 
Think about it step by step and return the answer in JSON format. Use only the capital letters for the answer. Example: { "Explanation": "<explanation>", "Answer": "A" or "B" or "C" or "D"}. 

### Output:
'''


def is_legit(s):
	try:
		out = literal_eval(s)
		if isinstance(out, dict):
			return True
		else:
			return False
	except:
		return False
	

def is_url(string):
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https:// or ftp://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(url_pattern.match(string))


def save_result(data, filename, save_dir, suffix):
	df_ans = pd.DataFrame(data)
	save_path = os.path.join(save_dir, f'{filename}_ans{suffix}.csv')
	df_ans.to_csv(save_path, index=False)

	# print save path in green
	print('>>> File saved at:', colored(save_path, 'green'))


def refine_result(result, llm_call, n=3, verbose=False):
	count = 0
	while not is_legit(result):
		result = postprocess_string(result, llm_call) # self refining
		result = result.strip('`')
		count += 1
		if  count == n:
			if verbose: print(f'ERROR: {result}')
			break

	if is_legit(result):
		result_dict = literal_eval(result)
		answer = result_dict.get('Answer', None)
		explanation = result_dict.get('Explanation', None)
	else:
		answer = 'ERROR'
		explanation = result  

	return answer, explanation
    
def postprocess_string(s, llm_call):
	s = s.replace('\n', ' ').replace('  ', ' ').strip()
	prompt = f'''You: ### Input text:
"{s}"

### Task:
Re-format the answer in JSON format. Use only the capital letters for the answer. Example: {{ "Explanation": "<explanation>", "Answer": "A" or "B" or "C" or "D"}}. Nothing follows after that JSON formatted answer.

AI:
'''
	return llm_call(prompt)

def preprocessing(s):
    s = s.replace('\n', ' ').replace('  ', ' ').strip() 


#######################################
############ OPEN-SOURCE LLM ##########
#######################################
class OpensourceLLM:
	def __init__(self, model_name_or_path='microsoft/Phi-3-mini-4k-instruct', temp=0.3):
		self.model_name = model_name_or_path
		self.temp = temp

		if is_url(self.model_name):
			self.url = model_name_or_path # should look like 'http://127.0.0.1:5000/v1/
			self.headers = {
					"Content-Type": "application/json"
			}
		else:
			self.url = None
			self.model, self.tokenizer = load_model_tokenizer(self.model_name)

	def call(self, prompt):
		# predict
		data = {
			"prompt": prompt,
			"max_tokens": 500,
			"temperature": self.temp,
			'repetition_penalty': 1.1,
			#"top_p": 0.9,
			"seed": 10
		}

		if self.url:
			response = requests.post(self.url, headers=self.headers, json=data)
			pred = literal_eval(response.text)['choices'][0]['text']
		else:
			pred = get_answer_checkpoint(prompt, self.model, self.tokenizer)
		return pred
	
	def get_answer(self, question):
		'''Process question and define prompt here
		'''
		prompt = INSTRUCTION + f'### Question: \n{question}\n' + TASK
		result = self.call(prompt)

		# if there is "You:" or "AI:" in the result, remove it and everything after
		if 'You:' in result:
			result = result.split('You:')[0]
		if 'AI:' in result:
			result = result.split('AI:')[0]
		result = result.strip()
		return result


#######################################
################# GPT #################
#######################################
class GPT:
	def __init__(self, model_name='gpt-4-turbo', temp=0.3):
		self.model_name = model_name
		self.temp = temp

	def call(self, prompt):
		# predict
		ans = client.chat.completions.create(
			model=self.model_name,
			messages=[
					{"role": "system", "content": "You are a helpful assistant that can answer clincal questions and provide explanation."},
					{"role": "user", "content": prompt}
				],
			temperature=self.temp,
		)

		pred = ans.choices[0].message.content
		return pred

	def get_answer(self, question):
		'''Process question and define prompt here
		'''
		prompt = f'Question: {question}\n' + TASK
		result = self.call(prompt)
		return result
 