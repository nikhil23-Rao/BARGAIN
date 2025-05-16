from openai import OpenAI
import os
import json
from pydantic import BaseModel
import numpy as np

from BARGAIN.models.AbstractModels import Oracle, Proxy


class GeneralOracleAnswer(BaseModel):
    is_correct: bool
    correct_answer: str


def get_bool_val_prob(res, logprobs=None):
    if logprobs is None:
        output = False
        if 'true' in res.lower() and 'false' not in res.lower():
            output = True
        return output

    true_prob = 0
    false_prob = 0
    for toplogprob in logprobs[0].top_logprobs:
        if toplogprob.token == 'True':
            true_prob = np.exp(toplogprob.logprob)
        if toplogprob.token == 'False':
            false_prob = np.exp(toplogprob.logprob)
    if true_prob == 0 and false_prob == 0:
        return False, 0
    norm = true_prob+false_prob
    true_prob = true_prob/norm
    false_prob = false_prob/norm
    if true_prob > false_prob:
        return True, true_prob
    return False, false_prob


class OpenAIProxy(Proxy):
    def __init__(
        self,
        task: str,
        is_binary: bool = False,
        model: str = 'gpt-4o-mini',
        verbose: bool = True
    ) -> None:
        '''
        Args: 
            task: prompt to perform on data records. `task` must be a templatized string: `task.format(data_record)` is passed to `model` to process a `data_record`
            is_binary: Set to `True` if the task is a binary classifiction task. **WARNING** If `True`, `task` should have directions to ensure `model` outputs only True or False
            model: Name of OpenAI model
            verbose: provide progress updates

        '''
        super().__init__(verbose=verbose)
        self.task = task
        self.is_binary = is_binary
        self.model = model

        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )


# logprobs_data = {
#     "tokens": ["The", " capital", " of", " France", " is", " Paris", "."],
#     "token_logprobs": [-0.01, -0.02, -0.03, -0.01, -0.02, -0.05, -0.1],
#     "top_logprobs": [
#         {"The": -0.01, "A": -0.7},
#         {" capital": -0.02, " city": -0.6},
#         {" of": -0.03, " for": -1.2},
#         {" France": -0.01, " Germany": -1.5},
#         {" is": -0.02, " was": -1.0},
#         {" Paris": -0.05, " Lyon": -1.2},
#         {".": -0.1, "!": -2.0}
#     ]
# }


    def determine_multi_step_classifier(self, s):
        return len(s.strip().split()) > 1

    def classifiers_proxy_func(self, data_record, classifiers):
        # example classifiers array: ["Cat", "Dog", "Tiger", "Tiger Shark"]
        classifier_output = []
        for classifier in classifiers:
            classifier_output.append({classifier: {"confidence_score": 0}})
        task_with_data = self.task.format(data_record)
        prompt = [
            {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
            {"role": "user", "content": task_with_data}
        ]
        response = self.client.beta.chat.completions.parse(
            model=self.model, messages=prompt, logprobs=True, seed=0, temperature=0, max_tokens=2, top_logprobs=10)
        res = response.choices[0].message.content
        logprobs = response.choices[0].logprobs.content

        # 1.) loop through all tokens for each step and see where all each classifier is mentioned
        for element in logprobs:
            for possible_token in element.top_logprobs:
                matched_classifier = None
                for c in classifiers:
                    if (possible_token.token.startswith(c) and (not self.determine_multi_step_classifier(c))):
                        # tiger
                        matched_classifier = c
                        classifier_output[c]["confidence_score"] += possible_token.logprob
                    elif (self.determine_multi_step_classifier(c)):
                        steps = c.strip().split()
                        matched_classifier = steps[-1]
                        if (possible_token.token.startswith(matched_classifier)):
                            # tiger shark
                            classifier_output[c]["confidence_score"] += possible_token.logprob
                    else:
                        continue
        print(classifier_output)
        return classifier_output

    def proxy_func_general(self, data_record):
        task_with_data = self.task.format(data_record)
        prompt = [
            {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
            {"role": "user", "content": task_with_data}
        ]
        response = self.client.beta.chat.completions.parse(
            model=self.model, messages=prompt, logprobs=True, seed=0, temperature=0)
        if response.choices[0].logprobs is None:
            prob = 0
        else:
            logprobs = response.choices[0].logprobs.content
            all_logprobs = 0
            for t in logprobs:
                all_logprobs += t.logprob
            prob = np.exp(all_logprobs)

        return response.choices[0].message.content, prob

    def proxy_func_binary(self, data_record):
        task_with_data = self.task.format(data_record)
        prompt = [
            {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
            {"role": "user", "content": task_with_data}
        ]
        response = self.client.beta.chat.completions.parse(
            model=self.model, messages=prompt, logprobs=True, seed=0, temperature=0, max_tokens=2, top_logprobs=10)
        res = response.choices[0].message.content
        logprobs = response.choices[0].logprobs.content
        return get_bool_val_prob(res, logprobs)

    def proxy_func(self, data_record):
        if self.is_binary:
            return self.proxy_func_binary(data_record)
        else:
            return self.classifiers_proxy_func(data_record, [
                "lion", "tiger", "elephant", "giraffe", "zebra",
                "kangaroo", "panda", "koala", "dolphin", "whale",
                "eagle", "falcon", "bear", "wolf", "fox",
                "rabbit", "deer", "monkey", "hippopotamus", "rhinoceros"
            ])


class OpenAIOracle(Oracle):
    def __init__(
        self,
        task: str,
        is_binary: bool = False,
        model: str = 'gpt-4o',
        verbose: bool = True
    ):
        '''
        Args: 
            task: prompt to perform on data records. `task` must be a templatized string: `task.format(data_record)` is passed to `model` to process a `data_record`
            is_binary: Set to `True` if the task is a binary classifiction task. **WARNING** If `True`, `task` should have directions to ensure `model` outputs only True or False
            model: Name of OpenAI model
            verbose: provide progress updates

        '''
        super().__init__(verbose=verbose)
        self.task = task
        self.is_binary = is_binary
        self.model = model

        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def oracle_func_binary(self, data_record, proxy_output):
        task_with_data = self.task.format(data_record)
        prompt = [
            {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
            {"role": "user", "content": task_with_data}
        ]
        response = self.client.beta.chat.completions.parse(
            model=self.model, messages=prompt, logprobs=False, seed=0, temperature=0, max_tokens=2)
        res = response.choices[0].message.content
        oracle_output = get_bool_val_prob(res)
        return oracle_output == proxy_output, oracle_output

    def oracle_func_general(self, data_record, proxy_output):
        task_with_data = self.task.format(data_record)
        prompt = [
            {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
            {"role": "user", "content": f'''
                        Consider the following task and a given response:
                        
                        Task:
                        {task_with_data}

                        Response: {proxy_output}

                        Is the provided response correct? If the provided answer is incorrect, provide the correct answer.
                        '''}
        ]
        response = self.client.beta.chat.completions.parse(
            model=self.model, messages=prompt, response_format=GeneralOracleAnswer, logprobs=False, seed=0, temperature=0)
        res = json.loads(response.choices[0].message.content)
        correct_answer = res['correct_answer']
        if res['is_correct']:
            correct_answer = proxy_output
        return res['is_correct'], correct_answer

    def oracle_func(self, data_record, proxy_output):
        if self.is_binary:
            return self.oracle_func_binary(data_record, proxy_output)
        else:
            return self.oracle_func_general(data_record, proxy_output)
