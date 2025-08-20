
import numpy as np
import time
import pandas as pd
import random
from typing import List, Union, Tuple
import re

from BARGAIN.sampler.wor_sampler import WoR_Sampler
from BARGAIN.models.AbstractModels import Oracle, Proxy
from BARGAIN.bounds.betting_bounds import test_if_true_mean_is_above_m, test_if_true_mean_is_below_m

from openai import OpenAI
import os
import json
from pydantic import BaseModel

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


class TokenStep:
    def __init__(self, token_string, logprob, prob, topten):
        self.token_string = token_string
        self.logprob = logprob
        self.prob = prob
        self.topten = topten

    def __repr__(self):
        return (f"top_ten={[repr(t) for t in self.topten]},")

    def get_normalized_top_ten_probs(self):
        total = 0
        pij_matrix = []
        for t in self.topten:  # use np
            total += t.prob
        for token in self.topten:
            pij_matrix.append((token.prob / total))
        return pij_matrix


class Token:
    def __init__(self, token_string, logprob, prob):
        self.token_string = token_string
        self.logprob = logprob
        self.prob = prob

    def __repr__(self):
        return (f"str={self.token_string} + prob={self.prob:.4f},")


class OpenAIProxy(Proxy):
    total_proxy_cost = 0.0

    def calculate_openai_cost(self, usage_obj: dict, model: str) -> float:
        """
        Calculate the estimated cost of an OpenAI LLM request.

        Args:
            usage (dict): Dictionary with 'prompt_tokens' and 'completion_tokens'.
            model (str): Model name string (e.g., 'gpt-4o', 'gpt-3.5-turbo').

        Returns:
            float: Estimated cost in USD.
        """

        # Pricing per 1000 tokens (prompt and completion) for some example models
        MODEL_PRICING = {
            "gpt-4o": {"prompt": 2.5, "completion": 10},
            "gpt-4o-mini": {"prompt": 0.15, "completion": 0.6}
            # Add your other models and prices here
        }

        if model not in MODEL_PRICING:
            raise ValueError(
                f"Pricing for model '{model}' not found. Please add it to MODEL_PRICING.")

        prompt_tokens = getattr(usage_obj, "prompt_tokens", 0)
        completion_tokens = getattr(usage_obj, "completion_tokens", 0)

        prompt_rate = MODEL_PRICING[model]["prompt"]
        completion_rate = MODEL_PRICING[model]["completion"]

        cost = (prompt_tokens / 1000000) * prompt_rate + \
            (completion_tokens / 1000000) * completion_rate
        return cost

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

    def determine_multi_step_classifier(self, s):
        return len(s.strip().split()) > 1

    def is_valid_start(self, token, classes):
        return any(cls.startswith(token) for cls in classes)

    def retrieve_llm_response(self, data_record):
        # run LLM
        task_with_data = self.task.format(data_record)
        prompt = [
            {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
            {"role": "user", "content": task_with_data}
        ]
        response = self.client.beta.chat.completions.parse(
            model=self.model, messages=prompt, logprobs=True, seed=0, temperature=0, max_tokens=1002, top_logprobs=10, top_p=1,)
        # print(response.usage)
        return response, self.calculate_openai_cost(response.usage, self.model)

    def is_valid_prefix(self, prefix, classes, predicted_class):
        # eject predicted class
        classes_besides_predicted = [s for s in classes if s.lower().replace(
            " ", "") != predicted_class.lower().replace(" ", "")]

        # loop through this set of classes to check
        for cls in classes_besides_predicted:
            if (cls.replace(" ", "").lower().startswith(prefix.replace(" ", "").lower())):
                return True
        return False

    def class_proxy_func(self, data_record, classes):
        list_of_token_steps = []
        predicted_string = ""

        # Call LLM
        response = self.retrieve_llm_response(data_record=data_record)
        cost = response[1]
        response = response[0]
        logprobs = response.choices[0].logprobs.content

        # if answer is something not listed in the class -> return a 0.0 confidence early
        # if not any(c.lower().replace(" ", "") == response.choices[0].message.content.lower().replace(" ", "") for c in classes):
        #     print("RESPONSE" + response.choices[0].message.content)
        #     print(response.choices[0].message.content, 0.0)
        #     return response.choices[0].message.content, 0.0, 0.0

        for token_step in logprobs:
            # Build top-10 list as Token objects
            top_available_tokens = []
            top_available_tokens.append(Token(
                token_step.token,
                token_step.logprob,
                np.exp(token_step.logprob),
            ))
            for possible_token in token_step.top_logprobs[1:]:
                # is_valid_prefix:
                # 1.) first arg -> prefix being checked
                # 2.) second arg -> list of classes to check against
                if self.is_valid_prefix(predicted_string + possible_token.token, classes, response.choices[0].message.content):
                    # print("TOKEN + PROB", predicted_string + possible_token.token,
                    #       np.exp(possible_token.logprob))
                    top_available_tokens.append(Token(
                        possible_token.token,
                        possible_token.logprob,
                        np.exp(possible_token.logprob),
                    ))

            predicted_string += token_step.token

            # create step with updated data
            t = TokenStep(
                token_step.token,
                token_step.logprob,
                np.exp(token_step.logprob),
                top_available_tokens
            )

            # store all token_steps for future iteration
            list_of_token_steps.append(t)

        # Run Algorithm
        prob_output = 1  # numerator
        part_denom = 0
        for step in list_of_token_steps:
            sum_of_available_top_ten = np.sum(
                # sum of all not_removed probs where i>1
                step.get_normalized_top_ten_probs()[1:])
            # sum * (product of all p1s to k)
            part_denom += (sum_of_available_top_ten * prob_output)
            prob_output *= step.get_normalized_top_ten_probs()[0]  # p1

        confidence = prob_output / (prob_output + part_denom)  # formula
        print(response.choices[0].message.content, "  ", confidence)
        return response.choices[0].message.content, confidence, cost

    def proxy_func_general(self, data_record):
        task_with_data = self.task.format(data_record)
        prompt = [
            {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
            {"role": "user", "content": task_with_data}
        ]
        response = self.client.beta.chat.completions.parse(
            model=self.model, messages=prompt, logprobs=True, seed=0, temperature=0, max_tokens=1002, top_logprobs=10, top_p=1)
        cost = self.calculate_openai_cost(response.usage, self.model)
        if response.choices[0].logprobs is None:
            prob = 0
        else:
            logprobs = response.choices[0].logprobs.content
            all_logprobs = 0
            for t in logprobs:
                all_logprobs += t.logprob
            prob = np.exp(all_logprobs)

        print(response.choices[0].message.content, "  ", prob)
        return response.choices[0].message.content, prob, cost

    def proxy_func_binary(self, data_record):
        task_with_data = self.task.format(data_record)
        prompt = [
            {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
            {"role": "user", "content": task_with_data}
        ]
        response = self.client.beta.chat.completions.parse(
            model=self.model, messages=prompt, logprobs=True, seed=0, temperature=0, max_tokens=1000, top_logprobs=10)
        res = response.choices[0].message.content
        logprobs = response.choices[0].logprobs.content
        return get_bool_val_prob(res, logprobs)

    def proxy_func(self, data_record):
        print("new starting")
        if self.is_binary:
            return self.proxy_func_binary(data_record)
        else:
            # hard coded classes for now
            # return self.proxy_func_general(data_record)
            res = self.class_proxy_func(
                data_record, ['technology', 'entertainment', 'world', 'science', 'politics', 'sports', 'automobile'])
            # res = self.proxy_func_general(data_record)
            OpenAIProxy.total_proxy_cost += res[2]
            # print("TOTAL COST", OpenAIProxy.total_proxy_cost)
            time.sleep(0.5)
            return res[0:2]


class OpenAIOracle(Oracle):
    total_oracle_cost = 0.0

    def calculate_openai_cost(self, usage_obj: dict, model: str) -> float:
        """
        Calculate the estimated cost of an OpenAI LLM request.

        Args:
            usage (dict): Dictionary with 'prompt_tokens' and 'completion_tokens'.
            model (str): Model name string (e.g., 'gpt-4o', 'gpt-3.5-turbo').

        Returns:
            float: Estimated cost in USD.
        """

        # Pricing per 1000 tokens (prompt and completion) for some example models
        MODEL_PRICING = {
            "gpt-4o": {"prompt": 2.5, "completion": 10},
            "gpt-4o-mini": {"prompt": 0.15, "completion": 0.6}
            # Add your other models and prices here
        }

        if model not in MODEL_PRICING:
            raise ValueError(
                f"Pricing for model '{model}' not found. Please add it to MODEL_PRICING.")

        prompt_tokens = getattr(usage_obj, "prompt_tokens", 0)
        completion_tokens = getattr(usage_obj, "completion_tokens", 0)

        prompt_rate = MODEL_PRICING[model]["prompt"]
        completion_rate = MODEL_PRICING[model]["completion"]

        cost = (prompt_tokens / 1000000) * prompt_rate + \
            (completion_tokens / 1000000) * completion_rate
        return cost

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
        cost = self.calculate_openai_cost(response.usage, self.model)
        res = json.loads(response.choices[0].message.content)
        correct_answer = res['correct_answer']
        print(res)

        if res['is_correct']:
            correct_answer = proxy_output
        return res['is_correct'], correct_answer, cost

    def oracle_func(self, data_record, proxy_output):
        if self.is_binary:
            return self.oracle_func_binary(data_record, proxy_output)
        else:
            res = self.oracle_func_general(data_record, proxy_output)
            OpenAIOracle.total_oracle_cost += res[2]
            # print("TOTAL ORACLE COST:", OpenAIOracle.total_oracle_cost)
            time.sleep(3)
            return res[0:2]


class BARGAIN_A():
    '''
    Class to process a dataset using a cheap proxy or an expensive oracle while guaranteeing the output is validatd by the oracle with a desired accuracy target
    '''

    def __init__(
        self,
        proxy: Proxy,
        oracle: Oracle,
        target: float = 0.9,
        delta: float = 0.1,
        M: int = 20,
        verbose: bool = True,
        seed: int = 0
    ) -> None:
        '''
        Args:
            proxy: Proxy model to use
            oracle: Oracle model to use
            target: Desired precision target, float between 0 and 1
            delta: Probability of failure, float between 0 and 1
            M: Number of different thresholds to be considered by algorithm
            verbose: output progress details or not
            seed: Random seed

        '''
        self.delta = delta
        self.target = target

        self.proxy = proxy
        self.oracle = oracle

        self.M = M
        if seed is not None:
            np.random.seed(seed)
        self.verbose = verbose

    def __check_worth_trying(self, sample_indx, sample_is_correct, t, target):
        if len(sample_indx) < 50:
            return True
        mask_at_t = sample_indx <= t
        samples_at_thresh = sample_is_correct[mask_at_t]
        if np.mean(samples_at_thresh)-np.std(samples_at_thresh) < target:
            return False
        return True

    def __sample_till_confident_above_target(self, all_data_indexes, all_preds, confidence, target, total_sampled, curr_thresh, data_records):
        sample_step = 10
        sampled_is_correct = np.array([])
        sampled_preds = np.array([])
        sampled_index = np.array([]).astype(int)

        while self.__check_worth_trying(sampled_index, sampled_is_correct, curr_thresh, target):
            sampled_indexes, budget_used, sampled_all = self.sampler.sample(
                curr_thresh, sample_step)

            sampled_data_indexes = all_data_indexes[sampled_indexes]
            proxy_preds = all_preds[sampled_indexes]
            sampled_is_correct = np.concatenate([sampled_is_correct, self.oracle.is_answer_correct(
                sampled_data_indexes, data_records[sampled_indexes], proxy_preds)])
            sampled_index = np.concatenate([sampled_index, sampled_indexes])
            sampled_preds = np.concatenate([sampled_preds, proxy_preds])
            total_sampled += budget_used

            if sampled_all:
                return not np.mean(sampled_is_correct) < target, sampled_index, total_sampled

            samples_at_thresh = sampled_is_correct[sampled_index <= curr_thresh]
            N = curr_thresh+1
            if np.mean(samples_at_thresh) < target:
                conf_has_target = test_if_true_mean_is_below_m(
                    samples_at_thresh, target, alpha=confidence, without_replacement=True, N=N, fixed_sample_size=False)
            else:
                conf_has_target = test_if_true_mean_is_above_m(
                    samples_at_thresh, target, alpha=confidence, without_replacement=True, N=N, fixed_sample_size=False)

            if np.mean(samples_at_thresh) < target:
                is_below_target = True
            else:
                is_below_target = False
            if not conf_has_target:
                return not is_below_target, sampled_index, total_sampled

        return False, sampled_index, total_sampled

    def process(self, data_records: List[str], return_oracle_usage: bool = False) -> Union[List[str], Tuple[List[str], List[bool]]]:
        '''
        Returns the computed output for all data records. It guarantees the output matches what the `oracle` would've provided on at least `target` fraction of the records with probability 1-`delta` but minimizes number of `oracle` usags
        Args:
            data_records: String array containing data records to be processed.
            return_oracle_usage: If `True`, the function additionally outputs whether a record was processed by oracle or not

        Returns:
            Union[List[str], Tuple[List[str], List[bool]]]:
                - If `return_oracle_usage` is False, returns a list of processed output strings:
                    - List[str]: The computed outputs for the input `data_records` in the same order as `data_records`
                - If `return_oracle_usage` is True, returns a tuple:
                    - List[str]: The computed outputs for the input `data_records` in the same order as `data_records`
                    - List[bool]: A list of booleans where each element indicates whether the oracle was used for that record.

        '''
        self.proxy.reset()
        self.oracle.reset()

        data_records = np.array(data_records)
        data_idxs = np.arange(len(data_records))
        self.sampler = WoR_Sampler(len(data_idxs))
        thresh_step = max(len(data_idxs)//self.M, 1)

        if self.verbose:
            print("Getting Proxy output and Scores")
        proxy_preds, proxy_scores = self.proxy.get_preds_and_scores(
            data_idxs, data_records[data_idxs])

        sort_indx = np.argsort(proxy_scores)[::-1]
        proxy_preds = proxy_preds[sort_indx]
        proxy_scores = proxy_scores[sort_indx]
        data_idxs = data_idxs[sort_indx]
        data_records = data_records[sort_indx]

        sample_indexes = []
        total_sampled = 0

        best_thresh = 0
        if self.verbose:
            print("Determining Cascade Threshold")
        for curr_thresh in range(thresh_step-1, len(data_idxs), thresh_step):
            if curr_thresh == len(data_idxs)-1:
                new_target = self.target
            else:
                n_from_proxy = curr_thresh+1
                n_from_oracle = len(data_idxs)-n_from_proxy
                new_target = (self.target*(n_from_oracle +
                              n_from_proxy)-n_from_oracle)/n_from_proxy
                if new_target <= 0:
                    continue

            is_confident_above_target, sampled_index, total_sampled = self.__sample_till_confident_above_target(
                data_idxs, proxy_preds, self.delta, new_target, total_sampled,  curr_thresh, data_records)

            sample_indexes = np.concatenate([sample_indexes, sampled_index])

            if not is_confident_above_target:
                break
            best_thresh = curr_thresh
        proxy_indxs = np.setdiff1d(
            data_idxs[:best_thresh], data_idxs[np.array(sample_indexes).astype(int)])

        if self.verbose:
            print(
                f"Found Threshold, {len(proxy_indxs)*100/len(data_idxs):.1f}% of Data is Processed with Proxy")

        oracle_indexes = np.setdiff1d(data_idxs, proxy_indxs)
        if self.verbose:
            print(f"Processing with Oracle")
        oracle_outputs = self.oracle.get_pred(
            data_records[oracle_indexes], oracle_indexes)

        if self.verbose:
            print(f"Processing with Proxy")
        proxy_preds, proxy_scores = self.proxy.get_preds_and_scores(
            proxy_indxs, data_records[proxy_indxs])
        indexes_data_indx = np.concatenate([oracle_indexes, proxy_indxs])
        output = np.concatenate([oracle_outputs, proxy_preds])

        order = np.argsort(indexes_data_indx)
        output = output[order]

        if return_oracle_usage:
            used_oracle = np.array(
                [True]*len(oracle_indexes)+[False]*len(proxy_indxs))
            used_oracle = used_oracle[order]
            return output.tolist(), used_oracle.tolist()

        return output.tolist()


def generate_color_or_animal_data(n, animal_prop, hard_prop, misleading_text_length):
    colors = [
        "red", "blue", "green", "yellow", "orange",
        "purple", "pink", "brown", "black", "white",
        "cyan", "magenta", "lime", "teal", "indigo",
                "violet", "gold", "silver", "beige", "maroon"
    ]
    animals = [
        "lion", "tiger", "elephant", "zebra", "giraffe",
                "kangaroo", "panda", "koala", "dolphin", "whale",
                "eagle", "falcon", "bear", "wolf", "fox",
                "rabbit", "deer", "monkey", "hippopotamus", "rhinoceros"
    ]
    np.random.shuffle(animals)
    long_misleading_text = '''Color theory is a conceptual framework used in visual arts, design, and many areas of visual communication to understand how colors relate to each other and how they can be combined to create pleasing or effective compositions. Rooted in both science and aesthetics, color theory explores the nature of color, the psychological impact it has on viewers, and the ways in which different colors interact. It informs countless decisions in fields ranging from painting and graphic design to interior decoration, fashion, marketing, and branding.

                    At the heart of color theory lies the color wheel, a circular diagram of colors arranged according to their chromatic relationship. The first known color wheel was developed by Sir Isaac Newton in the 17th century, who demonstrated that white light could be split into a spectrum of colors and then recombined into white light. His color circle laid the groundwork for modern color theory.

                    The traditional color wheel consists of three primary colors: red, yellow, and blue. These are the building blocks of all other colors, as they cannot be made by mixing any other colors together. By combining two primary colors, you get secondary colors: green, orange, and purple. Mixing a primary color with a neighboring secondary color produces tertiary colors such as red-orange or blue-green. These twelve hues form the basis of the artist’s color wheel.

                    Understanding how colors relate to one another on the wheel allows artists and designers to create color harmonies. Color harmony refers to aesthetically pleasing combinations of colors that evoke a sense of balance and unity. Some common types of color harmonies include complementary, analogous, triadic, and split-complementary.

                    Complementary colors are those located directly opposite each other on the color wheel, such as blue and orange or red and green. These pairs produce high contrast and high visual energy when used together, often making elements stand out sharply. Analogous colors are found next to each other on the wheel, such as blue, blue-green, and green. They share a similar hue and tend to be harmonious and soothing, often found in natural environments.

                    Triadic color schemes involve three colors that are evenly spaced around the color wheel, forming a triangle. An example of this would be red, yellow, and blue. This approach offers strong visual contrast while retaining balance and richness. Split-complementary schemes use a base color and the two colors adjacent to its complementary color. This creates a vibrant yet less jarring contrast than a direct complementary scheme.

                    Beyond hue relationships, color theory also takes into account other dimensions of color, such as value, saturation, and temperature. Value refers to the lightness or darkness of a color. For example, adding white to a color creates a tint, while adding black produces a shade. Saturation, or chroma, describes the intensity or purity of a color. Highly saturated colors appear vivid and intense, while desaturated colors appear more muted or gray.

                    Color temperature refers to the psychological association of colors with warmth or coolness. Warm colors such as red, orange, and yellow tend to evoke energy, warmth, and excitement. Cool colors like blue, green, and violet convey calmness, tranquility, and sometimes sadness. These associations are not just aesthetic—they have psychological and emotional impacts on viewers, which makes color choice critical in communication and design.'''
    long_misleading_text = long_misleading_text[:misleading_text_length]
    np.random.seed(0)
    data = {'id': [], 'value': [], 'is_animal': [], 'animal_name': []}
    for i in range(n):
        data['id'].append(i)
        is_animal = np.random.rand() <= animal_prop
        is_hard = np.random.rand() <= hard_prop
        if is_animal:
            val = np.random.choice(animals)
            data['is_animal'].append(True)
            data['animal_name'].append(val)
        else:
            val = np.random.choice(colors)
            data['is_animal'].append(False)
            data['animal_name'].append("")
        if is_hard:
            val = long_misleading_text[:len(
                long_misleading_text)//2] + f" {val} " + long_misleading_text[len(long_misleading_text)//2:]
        data['value'].append(val)
    print(data)
    return pd.DataFrame.from_dict(data)


# Define Data and Task
# df = generate_color_or_animal_data(
#     n=100, animal_prop=1, hard_prop=1, misleading_text_length=300)

# task = '''
#         I will give you a text. Your task is to extract the name of the animal mentioned is the text.

#         Here is the text: {}

#         THERE IS EXACTLY ONE ANIMAL WITHIN THE TEXT.

#         You must respond with ONLY the name of the animal or "None" (don't resort to this, since there IS one animal in the document).
#         '''

# task = '''
#         I will give you a text. There will be multiple animals mentioned in the text. Your task is to extract the name of the most mentioned animal in the text.
#         You MUST respond with ONLY the name of the MOST mentioned animal. THERE IS ALWAYS A MOST MENTIONED ANIMAL.
#         here is the list of potential animals: [
#         "lion", "tiger", "elephant", "zebra", "giraffe",
#                 "kangaroo", "panda", "koala", "dolphin", "whale",
#                 "eagle", "falcon", "bear", "wolf", "fox",
#                 "rabbit", "deer", "monkey", "hippopotamus", "rhinoceros"
#         ]

#         Here is the text: {}


#         '''


# task = '''
#   I will give you a text. There will be multiple animals mentioned in the text. Your task is to extract the name of the MOST MENTIONED animal in the text.
#         You MUST respond with ONLY the name of the MOST animal. THERE IS ALWAYS A MOST MENTIONED animal.
#         here is the list of potential animals:
#         ["lion", "tiger", "elephant", "giraffe", "zebra",
#                                               "kangaroo", "panda", "koala", "dolphin", "whale",
#                                               "eagle", "falcon", "bear", "wolf", "fox",
#                                               "rabbit", "deer", "monkey", "hippopotamus", "rhinoceros"]

#         Here is the text: {}


#          '''


# task = '''
# I will give you a twitter Tweet. Your task is to identify the feeling of the tweet. You can choose from the following
# emotions: [happy, sad, surprise, fear, disgust, angry]

# You must ONLY return ONE of the emotions in the tweet

# Here is the tweet: {}
# '''


# task = '''
# you will be given an article title + description.
# Your task is to determine which category the article belongs to based on the text you see.
# Here are the potential categories: ['technology', 'entertainment', 'world', 'science', 'politics', 'sports', 'automobile']
# You must choose Only ONE Category out of the list, and return that category ONLY.
# -----
# here is the Article title + description: {}
# '''


# task = '''
# you will be given a random article with country names injected anywhere within the article.
# Your task is to determine which COUNTRY appears (is stated) the MOST in the document. There will be countries just mentioned once, but one COUNTRY will always be mentioned more than once (hence it is the most mentioned country in the article).
# Here are the potential categories:  ["United States", "Canada", "Mexico", "Brazil", "Argentina", "United Kingdom", "France", "Germany", "Italy", "Spain", "Portugal", "Netherlands", "Belgium", "Sweden", "Norway", "Russia", "Poland", "Ukraine", "Switzerland", "Greece", "India", "China", "Japan", "South Korea", "Indonesia", "Thailand", "Vietnam", "Philippines", "Pakistan", "Bangladesh", "Australia", "New Zealand", "South Africa", "Nigeria", "Egypt", "Kenya", "Ethiopia", "Turkey", "Saudi Arabia", "Iran"]
# You must choose Only ONE COUNTRY out of the list, and return THAT COUNTRY ONLY.
# -----
# here is the Article with randomly injected countries: {}
# '''


task = '''
you will be given a random article.
Your task is to determine which CATEGORY the article belongs to based on the contents you see. You may only select ONE category to classify the article as.
Here are the potential categories: ["Scientific Research and Technological Advancements in the Natural and Physical Sciences", "Entertainment Industry, Popular Culture, and Media Arts", "Innovations and Developments in Information Technology and Applied Sciences", "Global Affairs, International Relations, and World Events", "Political Systems, Governance, Public Policy, and Sociopolitical Analysis", "Athletics, Competitive Sports Events, and Physical Performance", "Automotive Engineering, Vehicle Technology, and Transportation Industry" ]
Return your answer as two full sentences:

- The first sentence should be a long, generic sentence stating the article’s category (from the list).

- The second sentence should be a brief, confident statement affirming that the classification is correct, without mentioning any details from the article.

Do NOT include any article-specific content in either sentence.

Do NOT use quotation marks.
-----
here is the Article: {}
'''

print(task)


# Define oracle and proxy
proxy = OpenAIProxy(task, model='gpt-4o-mini')
oracle = OpenAIOracle(task, model='gpt-4o')
df = pd.read_csv("class_proxy_results.csv").head(500)
# df = pd.read_csv("general_proxy_results.csv")


def extract_category_and_score(text):
    # Match full all-caps phrase with optional & and spaces (min 2 letters)
    category_match = re.search(
        r'\b(?:[A-Z]{2,}(?:\s*&\s*|\s+))*[A-Z]{2,}\b', text)
    # Match confidence score at the end
    score_match = re.search(r'(\d\.\d+)\s*$', text)

    if category_match and score_match:
        return f"{category_match.group(0).strip()}    {score_match.group(1)}"
    return ""


# Read the file


# Print just the extracted string(s)
# for text in df["ClassProxyResult"]:
#     result = extract_category_and_score(text)
#     if result:
#         print(result)
# print(df['ClassProxyResult'])
# extract_all_caps_phrases(df['ClassProxyResult'])

str_list = []
newstr_list = []


categories = ['science', 'entertainment', 'technology',
              'world', 'politics', 'sports', 'automobile']


# Map each category to a letter a-n
category_to_letter = {category: chr(97 + i)
                      for i, category in enumerate(categories)}


def get_letter_from_category(category):
    return category_to_letter.get(category, None)


def extract_cat_prob(text):
    # Extract the category name
    category_match = re.search(r'category of (.+?)\.', text)
    category = category_match.group(1) if category_match else None

    # Extract the probability (last floating number)
    prob_match = re.findall(r'(\d+\.\d+)', text)
    prob = prob_match[-1] if prob_match else None

    return category, prob


def extract_category(sentence: str) -> str:
    """
    Extracts the category and probability score from a sentence of the form:
    'The article belongs to the category of <CATEGORY>. This classification ... <NUMBER>'

    Args:
        sentence (str): The input sentence.

    Returns:
        (str, float): A tuple containing the category and probability, or None if not found.
    """
    # Extract category
    category_match = re.search(r"category of (.+?)\.", sentence)
    # Extract probability (last number in string)
    prob_match = re.search(r"([0-9]*\.[0-9]+)\s*$", sentence)

    if category_match and prob_match:
        category = category_match.group(1).strip()
        prob = float(prob_match.group(1))
        return category, prob

    return None


# # Apply extraction to each row in the 'ClassProxyResult' column
# df[['Category', 'Probability']] = df['GeneralProxyResult'].apply(
#     lambda x: pd.Series(extract_cat_prob(x)))

# # Print result

for text in df['ClassProxyResult']:
    category, prob = extract_category(text)
    print(f"{category}   {prob}")


# for index, row in df.iterrows():
#     # Get the article text from the current row
#     current_article = str(row['news_headline']) + \
#         " | " + str(row['news_article'])
#     # injecteda = row['news_category']
#     # print(get_letter_from_category(injecteda))
#     # print(row['news_category'])
#     category_map = {
#         "science": "Scientific Research and Technological Advancements in the Natural and Physical Sciences",
#         "entertainment": "Entertainment Industry, Popular Culture, and Media Arts",
#         "technology": "Innovations and Developments in Information Technology and Applied Sciences",
#         "world": "Global Affairs, International Relations, and World Events",
#         "politics": "Political Systems, Governance, Public Policy, and Sociopolitical Analysis",
#         "sports": "Athletics, Competitive Sports Events, and Physical Performance",
#         "automobile": "Automotive Engineering, Vehicle Technology, and Transportation Industry"
#     }

#     str_result = proxy.class_proxy_func(
#         current_article, [
#             "The article belongs to the category of Scientific Research and Technological Advancements in the Natural and Physical Sciences. This classification is correct.",
#             "The article belongs to the category of Scientific Research and Technological Advancements in the Natural and Physical Sciences. This classification is accurate and reflects the content of the article.",
#             "The article belongs to the category of Global Affairs, International Relations, and World Events. This classification is correct.",
#             "The article belongs to the category of Global Affairs, International Relations, and World Events. This classification is accurate and reflects the content of the article.",
#             "The article belongs to the category of Political Systems, Governance, Public Policy, and Sociopolitical Analysis. This classification is correct.",
#             "The article belongs to the category of Political Systems, Governance, Public Policy, and Sociopolitical Analysis. This classification is accurate and reflects the content of the article.",
#             "The article belongs to the category of Entertainment Industry, Popular Culture, and Media Arts. This classification is correct.",
#             "The article belongs to the category of Entertainment Industry, Popular Culture, and Media Arts. This classification is accurate and reflects the content of the article.",
#             "The article belongs to the category of Athletics, Competitive Sports Events, and Physical Performance. This classification is correct.",
#             "The article belongs to the category of Athletics, Competitive Sports Events, and Physical Performance. This classification is accurate and reflects the content of the article.",
#             "The article belongs to the category of Innovations and Developments in Information Technology and Applied Sciences. This classification is correct.",
#             "The article belongs to the category of Innovations and Developments in Information Technology and Applied Sciences. This classification is accurate and reflects the content of the article.",
#             "The article belongs to the category of Automotive Engineering, Vehicle Technology, and Transportation Industry. This classification is correct.",
#             "The article belongs to the category of Automotive Engineering, Vehicle Technology, and Transportation Industry. This classification is accurate and reflects the content of the article."
#         ])

#     str_list.append(str(str_result[0]) + "  " + str(str_result[1]))
#     newstr_result = proxy.proxy_func_general(current_article)
#     newstr_list.append(
#         str(newstr_result[0]) + "  " + str(newstr_result[1]))
#     time.sleep(2)


# # # # Option 1: Save as two CSV files
# pd.DataFrame({'ClassProxyResult': str_list}).to_csv(
#     'class_proxy_results.csv', index=False)
# pd.DataFrame({'GeneralProxyResult': newstr_list}).to_csv(
#     'general_proxy_results.csv', index=False)


# # Call BARGAIN to process
# print("starting process")

# bargain = BARGAIN_A(proxy, oracle, target=0.9,  delta=0.1, seed=0)
# df['output'] = bargain.process(df['value'].to_numpy())

# # Evaluate output
# df['is_correct'] = df['animal_name'] == df['output']


# df.to_csv("testcase1.csv", index=True)

# # Display the first 5 rows of the new DataFrame

# def print_most_frequent_element(arr):
#     if not arr:
#         print("Array is empty.")
#         return

#     freq = {}
#     for item in arr:
#         if item in freq:
#             freq[item] += 1
#         else:
#             freq[item] = 1

#     max_item = None
#     max_count = 0

#     for item, count in freq.items():
#         if count > max_count:
#             max_item = item
#             max_count = count

#     print(max_item)

def parse_animal_string(animal_string: str) -> list[str]:
    """
    Converts a string that looks like a Python list into an actual list of strings.
    Example input: "['koala', 'falcon', 'falcon']"
    """
    animal_string = animal_string.strip("[]").replace("'", "").replace('"', '')
    return [a.strip() for a in animal_string.split(",") if a.strip()]


def most_mentioned_animal_from_string(animal_string: str) -> str:
    """
    Given a stringified list of animals, returns the most mentioned animal.
    """
    animals = parse_animal_string(animal_string)

    if not animals:
        return None

    freq = {}
    for animal in animals:
        if animal in freq:
            freq[animal] += 1
        else:
            freq[animal] = 1

    # Find animal with highest frequency
    most_common = None
    max_count = 0
    for animal, count in freq.items():
        if count > max_count:
            most_common = animal
            max_count = count

    return most_common


# shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True).head(1000)
# shuffled.to_csv("kaggle1.csv", index=False)


# print(df)
str_list = []
newstr_list = []

proxy_cost = 0
oracle_cost = 0


# bargain = BARGAIN_A(proxy, oracle, target=0.8,  delta=0.2, seed=0)


# df['output'] = bargain.process(
#     (df['news_headline'] + " | " + df['news_article']).to_numpy())
# print("PROXY COST, ", OpenAIProxy.total_proxy_cost)
# print("ORACLE COST, ", OpenAIOracle.total_oracle_cost)
# print("TOTAL COST, ", OpenAIOracle.total_oracle_cost +
#       OpenAIProxy.total_proxy_cost)


# label_map = {
#     1: 'World',
#     2: 'Sports',
#     3: 'Business',
#     4: 'Sci/Tech'
# }


# df['is_correct'] = df['news_category'] == df['output']


# print(
#     f"Accuracy: {df['is_correct'].mean()}, Used Proxy: {1-oracle.get_number_preds()/len(df):.2f}")

# for index, row in df.iterrows():
#     # Get the article text from the current row
#     current_article = str(row['title']) + " | " + str(row['body'])
#     # injecteda = row['category']
#     # print(injecteda)
#     # print(most_mentioned_animal_from_string(injecteda))

#     str_result = proxy.class_proxy_func(
#         current_article, ['TECH', 'MEDIA', 'ENVIRONMENT', 'SPORTS', 'CRIME', 'BUSINESS', 'SCIENCE', 'ARTS & CULTURE', 'ENTERTAINMENT', 'RELIGION', 'POLITICS', 'COMEDY', 'EDUCATION', 'WOMEN'])

#     proxy_cost += str_result[2]

#     str_list.append(str(str_result[0]) + "  " + str(str_result[1]))
#     newstr_result = proxy.proxy_func_general(current_article)
#     newstr_list.append(
#         str(newstr_result[0]) + "  " + str(newstr_result[1]))
#     oracle_cost += str_result[2]
#     time.sleep(0.5)

# # Option 1: Save as two CSV files
# # print("PROXY COST", proxy_cost)
# # print("ORCALE COST", oracle_cost)
# pd.DataFrame({'ClassProxyResult': str_list}).to_csv(
#     'class_proxy_results.csv', index=False)
# pd.DataFrame({'GeneralProxyResult': newstr_list}).to_csv(
#     'general_proxy_results.csv', index=False)


model1_confidences = []
model1_correct = []
model2_confidences = []
model2_correct = []

data = """
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.05221519803009322	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.591514784913161
Entertainment Industry, Popular Culture, and Media Arts	Global Affairs, International Relations, and World Events   0.29385123007720326	Global Affairs, International Relations, and World Events   0.4945005389297827
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.4505902063359455	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.5621665758106109
Innovations and Developments in Information Technology and Applied Sciences	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.025009046743346742	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5492572734884418
Innovations and Developments in Information Technology and Applied Sciences	Entertainment Industry, Popular Culture, and Media Arts   0.3778653983658322	Global Affairs, International Relations, and World Events   0.36444325579214676
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.06496280729887731	Global Affairs, International Relations, and World Events   0.49999568042091175
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.11155860299792987	Global Affairs, International Relations, and World Events   0.6513439557146621
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.04775112604180494	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5620034618225254
Innovations and Developments in Information Technology and Applied Sciences	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.04405137975554162	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.4129060345836188
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.07641229232343226	Athletics, Competitive Sports Events, and Physical Performance   0.4999736209159835
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.07165089041281424	Global Affairs, International Relations, and World Events   0.6513249335645351
Global Affairs, International Relations, and World Events	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.08521946889259834	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.6041631386565991
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.08039495292966076	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.6223280093972635
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.09160728177109258	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.4087332022232199
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.44656723090974926	Global Affairs, International Relations, and World Events   0.562175266593423
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.16699060451583633	Entertainment Industry, Popular Culture, and Media Arts   0.7057837605014964
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.3470778932945691	Global Affairs, International Relations, and World Events   0.469946853737531
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.047068542262047895	Innovations and Developments in Information Technology and Applied Sciences   0.49999543477072433
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.06739633056832264	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5311881556795395
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.05930896766396924	Global Affairs, International Relations, and World Events   0.6221901794538198
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.0805378254905144	Athletics, Competitive Sports Events, and Physical Performance   0.5311687270005518
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.32545089170773533	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5621217028998932
Entertainment Industry, Popular Culture, and Media Arts	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.36452877185891497	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.530394858991738
Global Affairs, International Relations, and World Events	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.36602091648509183	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.5195227274689829
Entertainment Industry, Popular Culture, and Media Arts	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.04815232569384935	Entertainment Industry, Popular Culture, and Media Arts   0.31583942686193484
Entertainment Industry, Popular Culture, and Media Arts	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.03943200128473909	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.6220079257095584
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.4208639361840619	Athletics, Competitive Sports Events, and Physical Performance   0.5307234204768053
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.263973929586422	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.3497801818678671
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.0955942971794037	Global Affairs, International Relations, and World Events   0.5926617235216238
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.09060228329703914	Innovations and Developments in Information Technology and Applied Sciences   0.6224527210309968
Automotive Engineering, Vehicle Technology, and Transportation Industry	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.3909585851147521	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.7057832140729348
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.1339400615838357	Entertainment Industry, Popular Culture, and Media Arts   0.6224583520584267
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.3870393310672428	Entertainment Industry, Popular Culture, and Media Arts   0.5621708133063558
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.044989244376731616	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.3460037951342488
Innovations and Developments in Information Technology and Applied Sciences	Global Affairs, International Relations, and World Events   0.045618964658493556	Global Affairs, International Relations, and World Events   0.545696833146877
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.04127304924132415	Innovations and Developments in Information Technology and Applied Sciences   0.49994303399197354
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.28052004083818416	Innovations and Developments in Information Technology and Applied Sciences   0.5621729177137059
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.0649438019255331	Athletics, Competitive Sports Events, and Physical Performance   0.6791669698163079
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.3965731834077457	Innovations and Developments in Information Technology and Applied Sciences   0.4904339399720903
Innovations and Developments in Information Technology and Applied Sciences	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.04422669749859244	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.4990336778851019
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.1123224554121119	Athletics, Competitive Sports Events, and Physical Performance   0.7548696257983365
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.08053243931556503	Athletics, Competitive Sports Events, and Physical Performance   0.6511154344191873
Entertainment Industry, Popular Culture, and Media Arts	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.30744770497046725	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5237363878937757
Innovations and Developments in Information Technology and Applied Sciences	Global Affairs, International Relations, and World Events   0.04219335162568897	Global Affairs, International Relations, and World Events   0.592376220505458
Innovations and Developments in Information Technology and Applied Sciences	Global Affairs, International Relations, and World Events   0.3358166637335972	Global Affairs, International Relations, and World Events   0.4087708761648555
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.0780455079837504	Entertainment Industry, Popular Culture, and Media Arts   0.6224565973038113
Automotive Engineering, Vehicle Technology, and Transportation Industry	Global Affairs, International Relations, and World Events   0.051091769896390284	Global Affairs, International Relations, and World Events   0.6471994846220381
Global Affairs, International Relations, and World Events	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.3069935990172976	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.29858756525979735
Entertainment Industry, Popular Culture, and Media Arts	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.030940561131085318	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5645463294879379
Athletics, Competitive Sports Events, and Physical Performance	Entertainment Industry, Popular Culture, and Media Arts   0.250351606877487	Entertainment Industry, Popular Culture, and Media Arts   0.5563406039122181
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.11516822209398736	Entertainment Industry, Popular Culture, and Media Arts   0.5621727322232553
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.33044319170294134	Entertainment Industry, Popular Culture, and Media Arts   0.5621735097596541
Innovations and Developments in Information Technology and Applied Sciences	Entertainment Industry, Popular Culture, and Media Arts   0.06531109874403306	Entertainment Industry, Popular Culture, and Media Arts   0.5252292068516575
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.3144639808299746	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.46784736894827733
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.39168939164406635	Entertainment Industry, Popular Culture, and Media Arts   0.5926637069577501
Global Affairs, International Relations, and World Events	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.033726693705347935	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.6171109153120844
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.10549810290920382	Athletics, Competitive Sports Events, and Physical Performance   0.6513300405210906
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.06741718769791884	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.6221804090528115
Global Affairs, International Relations, and World Events	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.054553539145996	Global Affairs, International Relations, and World Events   0.6171906547897277
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Global Affairs, International Relations, and World Events   0.379715078740004	Global Affairs, International Relations, and World Events   0.40541670746929104
Athletics, Competitive Sports Events, and Physical Performance	Entertainment Industry, Popular Culture, and Media Arts   0.07371030922801546	Entertainment Industry, Popular Culture, and Media Arts   0.6791768783747532
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.3708438320964369	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5216310420453667
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.36076561851629113	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.5312059612559051
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.050317477324106026	Entertainment Industry, Popular Culture, and Media Arts   0.4999951372932279
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.10733632694191547	Entertainment Industry, Popular Culture, and Media Arts   0.6791765708779369
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.09719782137907304	Global Affairs, International Relations, and World Events   0.592658565419622
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.08457327829520798	Entertainment Industry, Popular Culture, and Media Arts   0.5411460978137176
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.09867367648268335	Athletics, Competitive Sports Events, and Physical Performance   0.5621214462337824
Innovations and Developments in Information Technology and Applied Sciences	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.07280069303561024	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.5925433626891768
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.37801731217218015	Athletics, Competitive Sports Events, and Physical Performance   0.5621379888731278
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.0817091244219405	Entertainment Industry, Popular Culture, and Media Arts   0.6224533841884199
Automotive Engineering, Vehicle Technology, and Transportation Industry	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.07388233048083401	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.7310574595023805
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Global Affairs, International Relations, and World Events   0.37118760772173537	Global Affairs, International Relations, and World Events   0.3256644388179853
Global Affairs, International Relations, and World Events	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.3613625607075249	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.6513273303546169
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.07725875811811814	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.5621735192403315
Automotive Engineering, Vehicle Technology, and Transportation Industry	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.07056388013462887	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.622411841223615
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.37009548932821723	Global Affairs, International Relations, and World Events   0.5621689066274679
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.4035729055715333	Entertainment Industry, Popular Culture, and Media Arts   0.531207062794229
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.4128909789115336	Global Affairs, International Relations, and World Events   0.4885100962304479
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.07469447653172263	Athletics, Competitive Sports Events, and Physical Performance   0.5924787974630221
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.3763042575529647	Global Affairs, International Relations, and World Events   0.6695177236173714
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Global Affairs, International Relations, and World Events   0.03538934775174768	Global Affairs, International Relations, and World Events   0.6224406166669592
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.07467797252945649	Innovations and Developments in Information Technology and Applied Sciences   0.5621601749984417
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.20955045957096982	Athletics, Competitive Sports Events, and Physical Performance   0.5429302858529179
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Global Affairs, International Relations, and World Events   0.04187064113560984	Global Affairs, International Relations, and World Events   0.5921591183206873
Entertainment Industry, Popular Culture, and Media Arts	Athletics, Competitive Sports Events, and Physical Performance   0.05518308245972472	Athletics, Competitive Sports Events, and Physical Performance   0.5925720251952087
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.20301069121309281	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.49982226514574374
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.0718428965147484	Global Affairs, International Relations, and World Events   0.499998186013175
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.12625997551011012	Global Affairs, International Relations, and World Events   0.5456916432811786
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.05329511517788741	Global Affairs, International Relations, and World Events   0.5621680499180294
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.07701797706915711	Innovations and Developments in Information Technology and Applied Sciences   0.59265780174594
Athletics, Competitive Sports Events, and Physical Performance	Entertainment Industry, Popular Culture, and Media Arts   0.048220437477311466	Athletics, Competitive Sports Events, and Physical Performance   0.5479166519365184
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.05489960237152562	Entertainment Industry, Popular Culture, and Media Arts   0.6511955261437414
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.11281946607186458	Entertainment Industry, Popular Culture, and Media Arts   0.6513530437656669
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.08874213719113809	Entertainment Industry, Popular Culture, and Media Arts   0.6224546607362146
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.050180826991236584	Entertainment Industry, Popular Culture, and Media Arts   0.5312045821987815
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.06447393750214879	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5621502540332125
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Global Affairs, International Relations, and World Events   0.06971998890067999	Global Affairs, International Relations, and World Events   0.638358026904302
Innovations and Developments in Information Technology and Applied Sciences	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.270814050066183	Global Affairs, International Relations, and World Events   0.4369763574920751
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.35099451561670403	Entertainment Industry, Popular Culture, and Media Arts   0.4999975506613146
Innovations and Developments in Information Technology and Applied Sciences	Global Affairs, International Relations, and World Events   0.03075881588614651	Global Affairs, International Relations, and World Events   0.5549135810610119
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.09608059775296707	Entertainment Industry, Popular Culture, and Media Arts   0.5926653094408961
Entertainment Industry, Popular Culture, and Media Arts	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.049488129097683055	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.6212577641508924
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.383997131402251	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.5926589122219503
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.11823763389442198	Athletics, Competitive Sports Events, and Physical Performance   0.7057666274801427
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.07564079677310481	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5619333144632437
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.10060609870809441	Athletics, Competitive Sports Events, and Physical Performance   0.592648689711239
Automotive Engineering, Vehicle Technology, and Transportation Industry	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.07823421492344271	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.5926659463473657
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.5213330379157277	Innovations and Developments in Information Technology and Applied Sciences   0.6799727857241534
Athletics, Competitive Sports Events, and Physical Performance	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.0032175140498187766	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.305599274624387
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.08911872565774771	Entertainment Industry, Popular Culture, and Media Arts   0.5559740321546088
Global Affairs, International Relations, and World Events	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.07394646419280046	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.6222739858653793
Innovations and Developments in Information Technology and Applied Sciences	Global Affairs, International Relations, and World Events   0.40033958738955866	Global Affairs, International Relations, and World Events   0.592631609129963
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.3989055656676671	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.4999885626860767
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.06661357450343625	Entertainment Industry, Popular Culture, and Media Arts   0.5621678427211
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.1310289130881406	Athletics, Competitive Sports Events, and Physical Performance   0.6512622213526087
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.10125554497568516	Athletics, Competitive Sports Events, and Physical Performance   0.592651485464515
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.0844664595700287	Entertainment Industry, Popular Culture, and Media Arts   0.5621577081267259
Innovations and Developments in Information Technology and Applied Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.07468461464640584	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.528358426569804
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.10167975538787782	Athletics, Competitive Sports Events, and Physical Performance   0.5621661123407413
Innovations and Developments in Information Technology and Applied Sciences	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.0335262783500821	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.42597442955445813
Athletics, Competitive Sports Events, and Physical Performance	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.3006825562435181	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5305729580698992
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.5459443469452717	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.6791644353747701
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.39851494385070807	Entertainment Industry, Popular Culture, and Media Arts   0.49999728071355515
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.08058383996354346	Global Affairs, International Relations, and World Events   0.6513514766952798
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.2531234588458712	Athletics, Competitive Sports Events, and Physical Performance   0.49998360559633864
Innovations and Developments in Information Technology and Applied Sciences	Global Affairs, International Relations, and World Events   0.01756005810161457	Global Affairs, International Relations, and World Events   0.45271264547481205
Entertainment Industry, Popular Culture, and Media Arts	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.28274788940253737	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.46788499525302857
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.06980914029070295	Entertainment Industry, Popular Culture, and Media Arts   0.592660236480583
Innovations and Developments in Information Technology and Applied Sciences	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.3461715291279895	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5113765074507196
Innovations and Developments in Information Technology and Applied Sciences	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.4093111185039807	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.49940739686390695
Automotive Engineering, Vehicle Technology, and Transportation Industry	Global Affairs, International Relations, and World Events   0.23985002900568952	Global Affairs, International Relations, and World Events   0.5190020427702214
Entertainment Industry, Popular Culture, and Media Arts	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.34183292845269775	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5311970472345641
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.32516621227605236	Innovations and Developments in Information Technology and Applied Sciences   0.5259438863693677
Automotive Engineering, Vehicle Technology, and Transportation Industry	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.32204836336773884	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.5926581143948711
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.06380037146171999	Global Affairs, International Relations, and World Events   0.4999932990726499
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.07951351497172436	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.592583086549385
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.3697784121352748	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5310862305377031
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.3200169747160134	Global Affairs, International Relations, and World Events   0.5275245721614642
Global Affairs, International Relations, and World Events	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.3545422726014445	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.4992661005064904
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.2893937026365036	Global Affairs, International Relations, and World Events   0.5312070075996375
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.4431281584363265	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.5926559799376343
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.5000176758256965	Global Affairs, International Relations, and World Events   0.5312001312032587
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.04151707755461384	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.6222923459795865
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.06832549930374288	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5592088597185317
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.13637294712118195	Entertainment Industry, Popular Culture, and Media Arts   0.5312064606219407
Global Affairs, International Relations, and World Events	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.431110476143062	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5572662539663411
Athletics, Competitive Sports Events, and Physical Performance	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.3525469756466038	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.561932075084965
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.06084479318566674	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5311412979538455
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.29941169977551735	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.6224478982465137
Innovations and Developments in Information Technology and Applied Sciences	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.03580666992681798	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5621684783658284
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.017720623115840006	Innovations and Developments in Information Technology and Applied Sciences   0.4992466358155559
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.11442604539383355	Global Affairs, International Relations, and World Events   0.7057789972137021
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.03949492223618121	Global Affairs, International Relations, and World Events   0.5159650411067518
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.07135816582593021	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5926054502664877
Global Affairs, International Relations, and World Events	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.03964044274363485	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5492570029751557
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.0806016356559971	Entertainment Industry, Popular Culture, and Media Arts   0.5926653438924516
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.11413277811275427	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5924510174669673
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.06342287557659335	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5311691088288077
Global Affairs, International Relations, and World Events	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.4366359802394385	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5282984427506372
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.12746050447347257	Athletics, Competitive Sports Events, and Physical Performance   0.6791326907479238
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.08922409353048415	Athletics, Competitive Sports Events, and Physical Performance   0.5311103355199897
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.06723240366039676	Entertainment Industry, Popular Culture, and Media Arts   0.5926655805391632
Automotive Engineering, Vehicle Technology, and Transportation Industry	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.04834180442260902	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.5923373725432879
Automotive Engineering, Vehicle Technology, and Transportation Industry	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.058115352569638616	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.4928666357638974
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.06500175715360733	Global Affairs, International Relations, and World Events   0.6222569023093708
Innovations and Developments in Information Technology and Applied Sciences	Global Affairs, International Relations, and World Events   0.04856872774655734	Global Affairs, International Relations, and World Events   0.4618250990067912
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.10050395865928437	Global Affairs, International Relations, and World Events   0.5926420777038841
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.062135345906763235	Athletics, Competitive Sports Events, and Physical Performance   0.6791325382559853
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.0680906399895982	Entertainment Industry, Popular Culture, and Media Arts   0.5926641577846001
Global Affairs, International Relations, and World Events	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.07144482784786675	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.62217318914655
Innovations and Developments in Information Technology and Applied Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.08883922962549115	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.6221013430322152
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.09093416168904295	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.6512958426174771
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.06533727046570305	Global Affairs, International Relations, and World Events   0.6222479208812969
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.19357435567939202	Global Affairs, International Relations, and World Events   0.5621754689844785
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.09045207241831342	Entertainment Industry, Popular Culture, and Media Arts   0.622456365649922
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.07823535447617003	Entertainment Industry, Popular Culture, and Media Arts   0.4909557960243656
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.06902659374240697	Athletics, Competitive Sports Events, and Physical Performance   0.6224267426735602
Entertainment Industry, Popular Culture, and Media Arts	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.3869859821861383	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5919544817681781
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.05381937566646348	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5311846349032594
Innovations and Developments in Information Technology and Applied Sciences	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.49469663983478923	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.57054273032925
Global Affairs, International Relations, and World Events	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.37481361460817725	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.6182865671593377
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.10837796106686663	Athletics, Competitive Sports Events, and Physical Performance   0.5621559845995234
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.06666079773874935	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.28108053530453375
Automotive Engineering, Vehicle Technology, and Transportation Industry	Global Affairs, International Relations, and World Events   0.07117347547688316	Global Affairs, International Relations, and World Events   0.5926552967749126
Entertainment Industry, Popular Culture, and Media Arts	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.05061222465141916	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5620599495375124
Athletics, Competitive Sports Events, and Physical Performance	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.021754937146007715	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.4403884939451816
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.03897099122628737	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5456894038322303
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Global Affairs, International Relations, and World Events   0.056340799689535745	Global Affairs, International Relations, and World Events   0.4991042413868942
Innovations and Developments in Information Technology and Applied Sciences	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.07100128060084826	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.49997840895461526
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.1374843049928618	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.4403956775711017
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.12495916807211381	Athletics, Competitive Sports Events, and Physical Performance   0.6492379485932563
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.03888501465118259	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.4087785006657419
Innovations and Developments in Information Technology and Applied Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.046878794736388646	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.4999970188178116
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.04938016750319764	Global Affairs, International Relations, and World Events   0.5584101921264435
Innovations and Developments in Information Technology and Applied Sciences	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.44379281289860845	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.564552766046698
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.047317763735311144	Global Affairs, International Relations, and World Events   0.5602102472238562
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.1158570309394859	Athletics, Competitive Sports Events, and Physical Performance   0.6512977241838295
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.41644628771302034	Global Affairs, International Relations, and World Events   0.562170774604143
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.08645076025912969	Global Affairs, International Relations, and World Events   0.6507592031468972
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.06574649613974477	Innovations and Developments in Information Technology and Applied Sciences   0.6224554343383921
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.08921121995349737	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.49999283015079105
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.3797543632241686	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.5312080070303878
Innovations and Developments in Information Technology and Applied Sciences	Global Affairs, International Relations, and World Events   0.03813228645041427	Global Affairs, International Relations, and World Events   0.5312068249816647
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.09171677409689093	Entertainment Industry, Popular Culture, and Media Arts   0.6513520814208619
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Global Affairs, International Relations, and World Events   0.054073809960386174	Global Affairs, International Relations, and World Events   0.5270780037490183
Entertainment Industry, Popular Culture, and Media Arts	Global Affairs, International Relations, and World Events   0.3739480307270936	Entertainment Industry, Popular Culture, and Media Arts   0.5113781836093289
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.41328174322294403	Innovations and Developments in Information Technology and Applied Sciences   0.5312069678891849
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.3355969709403802	Innovations and Developments in Information Technology and Applied Sciences   0.45954530767862023
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.047843567457531135	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.531181732800795
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.12504627879638625	Entertainment Industry, Popular Culture, and Media Arts   0.6224471745632992
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.04563835810628453	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.5262729973607885
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.5572618579528501	Innovations and Developments in Information Technology and Applied Sciences   0.5621489687521173
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.2796310089341835	Global Affairs, International Relations, and World Events   0.4999946623122346
Global Affairs, International Relations, and World Events	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.04447344953613737	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.2998972376201632
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.06923975398922201	Global Affairs, International Relations, and World Events   0.5929379294503058
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.07739410797733222	Entertainment Industry, Popular Culture, and Media Arts   0.5926645419794566
Entertainment Industry, Popular Culture, and Media Arts	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.3766037170907903	Entertainment Industry, Popular Culture, and Media Arts   0.5924311257799981
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.4799989160618461	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.5492175623836746
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.38357485930194013	Athletics, Competitive Sports Events, and Physical Performance   0.5215666054023886
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.07758708690948243	Innovations and Developments in Information Technology and Applied Sciences   0.4087473563994634
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.0449887580458616	Global Affairs, International Relations, and World Events   0.6224539967216413
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.07522512978675222	Innovations and Developments in Information Technology and Applied Sciences   0.4999988082526254
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.055977213779418285	Global Affairs, International Relations, and World Events   0.5607731900985613
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.07220890332489265	Athletics, Competitive Sports Events, and Physical Performance   0.6790206747337044
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.30948444405317305	Entertainment Industry, Popular Culture, and Media Arts   0.6224585089116245
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.09877386070284634	Athletics, Competitive Sports Events, and Physical Performance   0.6513400058492844
Innovations and Developments in Information Technology and Applied Sciences	Global Affairs, International Relations, and World Events   0.3309954662166535	Global Affairs, International Relations, and World Events   0.679177995612762
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.0638576800676434	Entertainment Industry, Popular Culture, and Media Arts   0.6513538139812073
Entertainment Industry, Popular Culture, and Media Arts	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.18335989965111985	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.3306547431304566
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.07223076908595065	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.5926610601248941
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.058072670417273146	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.6224576646214043
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.34406687210526365	Global Affairs, International Relations, and World Events   0.5307713566482108
Automotive Engineering, Vehicle Technology, and Transportation Industry	Global Affairs, International Relations, and World Events   0.3536888359973994	Global Affairs, International Relations, and World Events   0.4781304976193817
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.11871886808767002	Athletics, Competitive Sports Events, and Physical Performance   0.7057645366985804
Innovations and Developments in Information Technology and Applied Sciences	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.0397141230450121	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5195150568167098
Global Affairs, International Relations, and World Events	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.05897079936422794	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5619113361636725
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.0838822623279735	Global Affairs, International Relations, and World Events   0.5926492710313882
Innovations and Developments in Information Technology and Applied Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.023033216263637267	Global Affairs, International Relations, and World Events   0.33293060211382
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.11262840505828106	Athletics, Competitive Sports Events, and Physical Performance   0.6511185402519858
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.4378169867985836	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5924904013176889
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.07701670764238581	Athletics, Competitive Sports Events, and Physical Performance   0.5620180166357458
Global Affairs, International Relations, and World Events	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.31936333967077946	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.4342900998503442
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.383392012327891	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.531199872351591
Entertainment Industry, Popular Culture, and Media Arts	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.1447355595466178	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.29861543633350396
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.06432092827128583	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.4838173640301162
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.09161368317446927	Athletics, Competitive Sports Events, and Physical Performance   0.4999838451683008
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.21765461850349152	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.28127790844447204
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.42743948803860593	Entertainment Industry, Popular Culture, and Media Arts   0.5621745217974751
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.08859449386891047	Entertainment Industry, Popular Culture, and Media Arts   0.38181709773054257
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.027815217181111573	Global Affairs, International Relations, and World Events   0.4109640227801544
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.05071520937615393	Entertainment Industry, Popular Culture, and Media Arts   0.2586987365283213
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.07150497022494613	Athletics, Competitive Sports Events, and Physical Performance   0.5311958185863243
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.031905367070834016	Global Affairs, International Relations, and World Events   0.36890069732565056
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.05789593850203071	Athletics, Competitive Sports Events, and Physical Performance   0.7056574093958004
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.09572665051079429	Athletics, Competitive Sports Events, and Physical Performance   0.6512489977702477
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.0867406011370676	Global Affairs, International Relations, and World Events   0.6224558487359915
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.08770224880060051	Entertainment Industry, Popular Culture, and Media Arts   0.7057802538431245
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.11923738787443729	Athletics, Competitive Sports Events, and Physical Performance   0.5621567249195689
Automotive Engineering, Vehicle Technology, and Transportation Industry	Global Affairs, International Relations, and World Events   0.055001826339617456	Global Affairs, International Relations, and World Events   0.5511610860613505
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.04564225787276827	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5886497244191067
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.07351926231002899	Athletics, Competitive Sports Events, and Physical Performance   0.7310140628367954
Innovations and Developments in Information Technology and Applied Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.03339824944013368	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.4999358700814006
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.08597565498514786	Entertainment Industry, Popular Culture, and Media Arts   0.6216144848593166
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.062196106375042005	Innovations and Developments in Information Technology and Applied Sciences   0.5411827156829553
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.06750994524681957	Athletics, Competitive Sports Events, and Physical Performance   0.6513048336228556
Entertainment Industry, Popular Culture, and Media Arts	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.04206631283919604	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.6202348568281727
Global Affairs, International Relations, and World Events	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.16844793168445132	Global Affairs, International Relations, and World Events   0.2810723826907438
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.09765803913891323	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.6513450164330772
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.42734326052375377	Athletics, Competitive Sports Events, and Physical Performance   0.49998826064915297
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.07367438199856222	Global Affairs, International Relations, and World Events   0.4999951973164082
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.06611473167435154	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.6224504261561181
Global Affairs, International Relations, and World Events	Entertainment Industry, Popular Culture, and Media Arts   0.06390920355358336	Entertainment Industry, Popular Culture, and Media Arts   0.5506573227722784
Global Affairs, International Relations, and World Events	Entertainment Industry, Popular Culture, and Media Arts   0.06685708326096343	Entertainment Industry, Popular Culture, and Media Arts   0.6787863786948435
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.10289485616540289	Entertainment Industry, Popular Culture, and Media Arts   0.49999733989289313
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.3744999034521991	Global Affairs, International Relations, and World Events   0.5311929194519596
Athletics, Competitive Sports Events, and Physical Performance	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.020833886871453804	Global Affairs, International Relations, and World Events   0.3499311857273516
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.08145736789720265	Global Affairs, International Relations, and World Events   0.5312035883305926
Innovations and Developments in Information Technology and Applied Sciences	Global Affairs, International Relations, and World Events   0.004144840082635896	Global Affairs, International Relations, and World Events   0.3015250187665215
Innovations and Developments in Information Technology and Applied Sciences	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.3710689612632151	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5311132991310232
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.12639505021103067	Athletics, Competitive Sports Events, and Physical Performance   0.5311803827307704
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.12470921169880102	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.4999656401386174
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.05230699004399835	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5926313996639929
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Global Affairs, International Relations, and World Events   0.42487039836913765	Global Affairs, International Relations, and World Events   0.5312001740104375
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.10462667472490106	Athletics, Competitive Sports Events, and Physical Performance   0.6513335483397313
Innovations and Developments in Information Technology and Applied Sciences	Global Affairs, International Relations, and World Events   0.046186146133031444	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.2986319507385784
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.05805013287352925	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.6198934493758783
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.3720730932308135	Athletics, Competitive Sports Events, and Physical Performance   0.6224474683832122
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.08480867584103874	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.47892619247270823
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.08626907964463222	Global Affairs, International Relations, and World Events   0.5621730359613958
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.36717979050489813	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.6112045583945965
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.411156172257214	Athletics, Competitive Sports Events, and Physical Performance   0.5621632463516815
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.05617588766517281	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5310353741848914
Global Affairs, International Relations, and World Events	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.29923132614444997	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.46786574627839744
Global Affairs, International Relations, and World Events	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.027117139103538055	Global Affairs, International Relations, and World Events   0.4547819410076179
Entertainment Industry, Popular Culture, and Media Arts	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.04256881035210843	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.3459878826933983
Automotive Engineering, Vehicle Technology, and Transportation Industry	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.08987968693615189	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.5289705405721012
Athletics, Competitive Sports Events, and Physical Performance	Entertainment Industry, Popular Culture, and Media Arts   0.0851394438047047	Entertainment Industry, Popular Culture, and Media Arts   0.47576836583534093
Automotive Engineering, Vehicle Technology, and Transportation Industry	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.38282018664631917	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.5593306289141692
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.06751227371452581	Athletics, Competitive Sports Events, and Physical Performance   0.622405403741283
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.33515451207235486	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.499976714796205
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.05567145830476967	Global Affairs, International Relations, and World Events   0.3654370080454554
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.3474950791534614	Innovations and Developments in Information Technology and Applied Sciences   0.471657422371302
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.12796877226592948	Athletics, Competitive Sports Events, and Physical Performance   0.6791647616579903
Entertainment Industry, Popular Culture, and Media Arts	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.010456666686424752	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5895693805666907
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.0867003054282737	Global Affairs, International Relations, and World Events   0.5907818127621939
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.43672581363609736	Innovations and Developments in Information Technology and Applied Sciences   0.5621748495362979
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.1320369249000365	Athletics, Competitive Sports Events, and Physical Performance   0.5311810648781793
Innovations and Developments in Information Technology and Applied Sciences	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.05963665116010092	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.49972107296376694
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.19913800995189873	Athletics, Competitive Sports Events, and Physical Performance   0.6223313921874053
Global Affairs, International Relations, and World Events	Entertainment Industry, Popular Culture, and Media Arts   0.06620601340760704	Entertainment Industry, Popular Culture, and Media Arts   0.6223976464577564
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.41583706026965755	Global Affairs, International Relations, and World Events   0.49997713444398545
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.41014808413169845	Global Affairs, International Relations, and World Events   0.5312083637241244
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.08117896919003975	Global Affairs, International Relations, and World Events   0.4992773023917217
Innovations and Developments in Information Technology and Applied Sciences	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.049877303851073244	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5911992995487855
Automotive Engineering, Vehicle Technology, and Transportation Industry	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.06026888542977889	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.6467104702352833
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.04741795831713529	Athletics, Competitive Sports Events, and Physical Performance   0.5592296645321091
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.0818334461484425	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.6224296177208894
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.11948877391129402	Entertainment Industry, Popular Culture, and Media Arts   0.6513529127526522
Global Affairs, International Relations, and World Events	Entertainment Industry, Popular Culture, and Media Arts   0.08153750116722251	Entertainment Industry, Popular Culture, and Media Arts   0.5621716451634173
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.11408602899485334	Athletics, Competitive Sports Events, and Physical Performance   0.5926214805507591
Innovations and Developments in Information Technology and Applied Sciences	Global Affairs, International Relations, and World Events   0.07866289849960613	Global Affairs, International Relations, and World Events   0.5312059344647245
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.42591754543349397	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5925692252148204
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.08744052762258661	Entertainment Industry, Popular Culture, and Media Arts   0.7057819868219384
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.11989028516959271	Entertainment Industry, Popular Culture, and Media Arts   0.6224545351971286
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.3394506790665136	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5620892553709623
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.0593303502072822	Global Affairs, International Relations, and World Events   0.49998862453530246
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.07502264610348754	Athletics, Competitive Sports Events, and Physical Performance   0.5620797043308847
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.42079598869617096	Entertainment Industry, Popular Culture, and Media Arts   0.49999811264712396
Innovations and Developments in Information Technology and Applied Sciences	Entertainment Industry, Popular Culture, and Media Arts   0.08398132355263528	Entertainment Industry, Popular Culture, and Media Arts   0.5615100647038652
Innovations and Developments in Information Technology and Applied Sciences	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.045795679839830156	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5266492293743341
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.045017485034533594	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.34992017504175904
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.13298874411022368	Entertainment Industry, Popular Culture, and Media Arts   0.7310560643413258
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.08913590210567551	Entertainment Industry, Popular Culture, and Media Arts   0.5621754010973207
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.3073043318354708	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5621664857137963
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.06646621662956284	Global Affairs, International Relations, and World Events   0.4999332199803714
Innovations and Developments in Information Technology and Applied Sciences	Global Affairs, International Relations, and World Events   0.4497017028218562	Global Affairs, International Relations, and World Events   0.5311569321993934
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.08983013859004277	Athletics, Competitive Sports Events, and Physical Performance   0.5926479112171702
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.10235897383871116	Athletics, Competitive Sports Events, and Physical Performance   0.6220832616632461
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.2989510090189538	Athletics, Competitive Sports Events, and Physical Performance   0.4999830421780541
Global Affairs, International Relations, and World Events	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.032425580242048085	Global Affairs, International Relations, and World Events   0.34992615407158323
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.3373587819450536	Entertainment Industry, Popular Culture, and Media Arts   0.5312081069677339
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.050010496666535655	Innovations and Developments in Information Technology and Applied Sciences   0.5619582236802735
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.415266605237223	Global Affairs, International Relations, and World Events   0.5621730750908035
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.09898471153953918	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.52837863469161
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.312862470102865	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5253336282740494
Entertainment Industry, Popular Culture, and Media Arts	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.28399209971890943	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5283755887854705
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.05469304916970871	Global Affairs, International Relations, and World Events   0.6224419047265698
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.08225142521505356	Global Affairs, International Relations, and World Events   0.6513499835173103
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.4697043474493268	Entertainment Industry, Popular Culture, and Media Arts   0.5621738221697535
Athletics, Competitive Sports Events, and Physical Performance	Entertainment Industry, Popular Culture, and Media Arts   0.3603374112882074	Entertainment Industry, Popular Culture, and Media Arts   0.5312052443758117
Global Affairs, International Relations, and World Events	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.051324294541008564	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5573467473487941
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.10650463006821052	Entertainment Industry, Popular Culture, and Media Arts   0.6224569710485319
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.10424145903488027	Entertainment Industry, Popular Culture, and Media Arts   0.562173203019164
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.089296945709219	Entertainment Industry, Popular Culture, and Media Arts   0.6224571170743068
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Global Affairs, International Relations, and World Events   0.04511661175229307	Global Affairs, International Relations, and World Events   0.5311901082674989
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.5185958000807607	Entertainment Industry, Popular Culture, and Media Arts   0.5621746656050679
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.11390138398785016	Entertainment Industry, Popular Culture, and Media Arts   0.592659394765068
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.000965358358697448	Global Affairs, International Relations, and World Events   0.3318257715768387
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.399136844417402	Entertainment Industry, Popular Culture, and Media Arts   0.5621739999273414
Automotive Engineering, Vehicle Technology, and Transportation Industry	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.32596790873657105	Global Affairs, International Relations, and World Events   0.31536445403499974
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.410423946336732	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.5311914059892483
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.038663296864095684	Innovations and Developments in Information Technology and Applied Sciences   0.4678685935829886
Innovations and Developments in Information Technology and Applied Sciences	Entertainment Industry, Popular Culture, and Media Arts   0.08742629632350266	Entertainment Industry, Popular Culture, and Media Arts   0.7549107257432786
Automotive Engineering, Vehicle Technology, and Transportation Industry	Global Affairs, International Relations, and World Events   0.05425957054262491	Global Affairs, International Relations, and World Events   0.5613224353169686
Automotive Engineering, Vehicle Technology, and Transportation Industry	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.06756212703727266	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.5205373286612338
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.04014789942943178	Global Affairs, International Relations, and World Events   0.5312007214465914
Athletics, Competitive Sports Events, and Physical Performance	Entertainment Industry, Popular Culture, and Media Arts   0.05498857762837963	Athletics, Competitive Sports Events, and Physical Performance   0.3867938324675773
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.41803090146535515	Entertainment Industry, Popular Culture, and Media Arts   0.5621745470525131
Global Affairs, International Relations, and World Events	Entertainment Industry, Popular Culture, and Media Arts   0.08157176550823075	Entertainment Industry, Popular Culture, and Media Arts   0.5907726807084741
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.050256918667048525	Global Affairs, International Relations, and World Events   0.6218905646913966
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.1248227439533763	Athletics, Competitive Sports Events, and Physical Performance   0.731038295183394
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.10967178889053758	Athletics, Competitive Sports Events, and Physical Performance   0.5311920000712667
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.06640257110101039	Athletics, Competitive Sports Events, and Physical Performance   0.7056709007961939
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.11832755034783535	Entertainment Industry, Popular Culture, and Media Arts   0.7981813951916038
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.30133148325308107	Innovations and Developments in Information Technology and Applied Sciences   0.5621687122650674
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.06997857608072242	Athletics, Competitive Sports Events, and Physical Performance   0.622314228919871
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.08531868003623827	Athletics, Competitive Sports Events, and Physical Performance   0.5926210056043851
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.3201597653355305	Global Affairs, International Relations, and World Events   0.5621759618891193
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.43066846702050166	Entertainment Industry, Popular Culture, and Media Arts   0.6224566021472111
Innovations and Developments in Information Technology and Applied Sciences	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.045099411749814175	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.4999752482904856
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.07989742880564317	Athletics, Competitive Sports Events, and Physical Performance   0.7771824971425194
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.060671508632213544	Global Affairs, International Relations, and World Events   0.6513477822139548
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.10247821531456919	Athletics, Competitive Sports Events, and Physical Performance   0.5621042930837796
Athletics, Competitive Sports Events, and Physical Performance	Global Affairs, International Relations, and World Events   0.1579275898636469	Athletics, Competitive Sports Events, and Physical Performance   0.47217095799846354
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.0771586008455389	Athletics, Competitive Sports Events, and Physical Performance   0.5297178269229414
Athletics, Competitive Sports Events, and Physical Performance	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.3537680158624678	Athletics, Competitive Sports Events, and Physical Performance   0.38033498692161455
Innovations and Developments in Information Technology and Applied Sciences	Global Affairs, International Relations, and World Events   0.40176212939903383	Global Affairs, International Relations, and World Events   0.5311412922759919
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.08628763380047993	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.651346051213164
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.09453575024638725	Global Affairs, International Relations, and World Events   0.562171757317886
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.12501743348210176	Athletics, Competitive Sports Events, and Physical Performance   0.5311802432700583
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.06898414668824017	Entertainment Industry, Popular Culture, and Media Arts   0.5312056547073283
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.0785354968481949	Innovations and Developments in Information Technology and Applied Sciences   0.5926632905361062
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.09447144530820004	Athletics, Competitive Sports Events, and Physical Performance   0.7979341841426775
Innovations and Developments in Information Technology and Applied Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.07188017711876321	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.5290309970824836
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.0732108890355129	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.6513476475126843
Athletics, Competitive Sports Events, and Physical Performance	Entertainment Industry, Popular Culture, and Media Arts   0.035460630998622344	Entertainment Industry, Popular Culture, and Media Arts   0.5621691461941223
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.07881640945305002	Entertainment Industry, Popular Culture, and Media Arts   0.622456029578347
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.4354521199806209	Global Affairs, International Relations, and World Events   0.5312079603623385
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.0528076056785896	Innovations and Developments in Information Technology and Applied Sciences   0.5902522193287265
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.05260835578715144	Global Affairs, International Relations, and World Events   0.5926629328176423
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.3528472101192678	Global Affairs, International Relations, and World Events   0.6513264950584897
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.07151229868410712	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.5819965504523731
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.17246850102748906	Athletics, Competitive Sports Events, and Physical Performance   0.49999053585096453
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.056303276938484144	Global Affairs, International Relations, and World Events   0.5312073242040065
Innovations and Developments in Information Technology and Applied Sciences	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.0545455454865071	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.6212372274283604
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.057337693720531414	Global Affairs, International Relations, and World Events   0.6224578580488641
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.07101598839468393	Entertainment Industry, Popular Culture, and Media Arts   0.5621709447794416
Global Affairs, International Relations, and World Events	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.04967957686502749	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.5492450523480268
Innovations and Developments in Information Technology and Applied Sciences	Global Affairs, International Relations, and World Events   0.3447880218769404	Global Affairs, International Relations, and World Events   0.5926355253965211
Innovations and Developments in Information Technology and Applied Sciences	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.04353604573147241	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5311987332443753
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.06543500931093493	Global Affairs, International Relations, and World Events   0.562119718820967
Innovations and Developments in Information Technology and Applied Sciences	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.36673313635097193	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.4929669270284408
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.055913336115086025	Global Affairs, International Relations, and World Events   0.5312077939206925
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.08234777891774732	Athletics, Competitive Sports Events, and Physical Performance   0.5305188298587908
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.4004996679311369	Entertainment Industry, Popular Culture, and Media Arts   0.5621722674390287
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.06469166043839084	Global Affairs, International Relations, and World Events   0.4678854391651691
Automotive Engineering, Vehicle Technology, and Transportation Industry	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.3055887888260067	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.5018408784861731
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.09381346091060921	Athletics, Competitive Sports Events, and Physical Performance   0.4999912897965589
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.4218730458544524	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.49449566992507504
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.49370815930123313	Innovations and Developments in Information Technology and Applied Sciences   0.5156331082555005
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.04788776979369166	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.4332160124725976
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.07326964702446304	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5621176219473861
Innovations and Developments in Information Technology and Applied Sciences	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.4023838460867879	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5621080586305442
Athletics, Competitive Sports Events, and Physical Performance	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.032996065250079136	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.3499292384203581
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.11157575144439993	Athletics, Competitive Sports Events, and Physical Performance   0.5926387225984149
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.12098680089775075	Athletics, Competitive Sports Events, and Physical Performance   0.5311869468029053
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.36416589547313	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.49977786692746573
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.04948536533455103	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.4999860424558789
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.1259972602513398	Entertainment Industry, Popular Culture, and Media Arts   0.5926630070776542
Automotive Engineering, Vehicle Technology, and Transportation Industry	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.07854563420976828	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.5573517586069423
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.06680108571791377	Global Affairs, International Relations, and World Events   0.49998457496338683
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.07627011209504654	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5156296466756014
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.07880494783484393	Global Affairs, International Relations, and World Events   0.6224566455183306
Automotive Engineering, Vehicle Technology, and Transportation Industry	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.06602200695851454	Automotive Engineering, Vehicle Technology, and Transportation Industry   0.6513534438900696
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.46943809859649915	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.4995190664271955
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.10251266107611505	Athletics, Competitive Sports Events, and Physical Performance   0.5621497391600252
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.10200772102931185	Entertainment Industry, Popular Culture, and Media Arts   0.6224572053848608
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.028584274858679155	Global Affairs, International Relations, and World Events   0.4109756998939296
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.31485583696603725	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.44821667784533525
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.36751422173433984	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.4678741810937022
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.04207469784300452	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.5311526778201046
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.09147051098464781	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.6791500791845563
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.03242499966760078	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.3306523194239059
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.08310939681265803	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.6224431037475365
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.06179270671398338	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.5621730536565912
Innovations and Developments in Information Technology and Applied Sciences	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.06754040848868524	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5311905819463135
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.055889296096535895	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.5895663364619108
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.34798049510861223	Global Affairs, International Relations, and World Events   0.622456411629589
Entertainment Industry, Popular Culture, and Media Arts	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.3733883933482097	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5906467700935285
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.12729379728979098	Athletics, Competitive Sports Events, and Physical Performance   0.6791263488373397
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.08503801943823364	Global Affairs, International Relations, and World Events   0.6791695982451729
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.07854100963633373	Entertainment Industry, Popular Culture, and Media Arts   0.6513523650956449
Global Affairs, International Relations, and World Events	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.09696280736562199	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.49994908035189584
Automotive Engineering, Vehicle Technology, and Transportation Industry	Global Affairs, International Relations, and World Events   0.07704347571397367	Global Affairs, International Relations, and World Events   0.6203009233972229
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.0795097800959811	Athletics, Competitive Sports Events, and Physical Performance   0.6513108457000983
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.114266863816805	Entertainment Industry, Popular Culture, and Media Arts   0.5312029295208307
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.07102830509799392	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5620644889986386
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Global Affairs, International Relations, and World Events   0.41204757986393103	Global Affairs, International Relations, and World Events   0.5621725628224833
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.08251056251841146	Athletics, Competitive Sports Events, and Physical Performance   0.7310242224466703
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.06811415619820167	Global Affairs, International Relations, and World Events   0.5049169440160142
Innovations and Developments in Information Technology and Applied Sciences	Global Affairs, International Relations, and World Events   0.39568703202702843	Global Affairs, International Relations, and World Events   0.42597135368115013
Athletics, Competitive Sports Events, and Physical Performance	Entertainment Industry, Popular Culture, and Media Arts   0.46431360713314596	Entertainment Industry, Popular Culture, and Media Arts   0.49999780902488566
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.1513562330930066	Athletics, Competitive Sports Events, and Physical Performance   0.7057617585814159
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.0492247636715727	Global Affairs, International Relations, and World Events   0.49855967368073956
Global Affairs, International Relations, and World Events	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.30205307962425765	Global Affairs, International Relations, and World Events   0.30373618655734813
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.08905291792793923	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.5621723710923799
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.41887272678357795	Innovations and Developments in Information Technology and Applied Sciences   0.4999922787360564
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.13547941768402016	Entertainment Industry, Popular Culture, and Media Arts   0.7310546342102965
Innovations and Developments in Information Technology and Applied Sciences	Global Affairs, International Relations, and World Events   0.06448717529934757	Global Affairs, International Relations, and World Events   0.4999994470651736
Innovations and Developments in Information Technology and Applied Sciences	Global Affairs, International Relations, and World Events   0.04704097808985563	Global Affairs, International Relations, and World Events   0.5253719358774698
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.10933677922523753	Athletics, Competitive Sports Events, and Physical Performance   0.5621523864462213
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.39279306820615745	Athletics, Competitive Sports Events, and Physical Performance   0.5309995993877947
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Global Affairs, International Relations, and World Events   0.33954888033398883	Global Affairs, International Relations, and World Events   0.5621429199179758
Innovations and Developments in Information Technology and Applied Sciences	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.32114793844682515	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.387453976814497
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.0844283364751445	Global Affairs, International Relations, and World Events   0.5926656825095669
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.1113213890525178	Entertainment Industry, Popular Culture, and Media Arts   0.5312082423691581
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Athletics, Competitive Sports Events, and Physical Performance   0.03688334765040113	Athletics, Competitive Sports Events, and Physical Performance   0.48484075044168906
Political Systems, Governance, Public Policy, and Sociopolitical Analysis	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.10141196563324202	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.6508911386107595
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.41844566335057665	Innovations and Developments in Information Technology and Applied Sciences   0.531206677483103
Global Affairs, International Relations, and World Events	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.07055776121235612	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.4789203350888162
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.07833097887888427	Athletics, Competitive Sports Events, and Physical Performance   0.6788807955055141
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.43017042497754415	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.5926613093810408
Innovations and Developments in Information Technology and Applied Sciences	Global Affairs, International Relations, and World Events   0.04660041942456693	Global Affairs, International Relations, and World Events   0.4522396302756696
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.3657375193201384	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.48452646232658747
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.09628499889092149	Entertainment Industry, Popular Culture, and Media Arts   0.5312071622534612
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.3149389120859413	Innovations and Developments in Information Technology and Applied Sciences   0.44038345517319233
Innovations and Developments in Information Technology and Applied Sciences	Global Affairs, International Relations, and World Events   0.27637164145632925	Global Affairs, International Relations, and World Events   0.3112146055868893
Entertainment Industry, Popular Culture, and Media Arts	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.24865752881223782	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.5621710306885119
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Global Affairs, International Relations, and World Events   0.0730604504633279	Global Affairs, International Relations, and World Events   0.5312011213702597
Global Affairs, International Relations, and World Events	Global Affairs, International Relations, and World Events   0.10962785589907403	Global Affairs, International Relations, and World Events   0.6156172387975329
Entertainment Industry, Popular Culture, and Media Arts	Athletics, Competitive Sports Events, and Physical Performance   0.051848398108795754	Athletics, Competitive Sports Events, and Physical Performance   0.6223316968713476
Entertainment Industry, Popular Culture, and Media Arts	Entertainment Industry, Popular Culture, and Media Arts   0.10436902158778999	Entertainment Industry, Popular Culture, and Media Arts   0.5621752069468828
Scientific Research and Technological Advancements in the Natural and Physical Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.36129830660981865	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.5926538906522115
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.31992258229335024	Innovations and Developments in Information Technology and Applied Sciences   0.7057359853434735
Athletics, Competitive Sports Events, and Physical Performance	Athletics, Competitive Sports Events, and Physical Performance   0.09765295330712405	Athletics, Competitive Sports Events, and Physical Performance   0.5621604086125868
Innovations and Developments in Information Technology and Applied Sciences	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.0365429642252281	Scientific Research and Technological Advancements in the Natural and Physical Sciences   0.6197114513670301
Innovations and Developments in Information Technology and Applied Sciences	Global Affairs, International Relations, and World Events   0.3530910770239336	Global Affairs, International Relations, and World Events   0.6081555464104201
Innovations and Developments in Information Technology and Applied Sciences	Innovations and Developments in Information Technology and Applied Sciences   0.44218956275681126	Innovations and Developments in Information Technology and Applied Sciences   0.531207173759492
Entertainment Industry, Popular Culture, and Media Arts	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.34889437667709333	Political Systems, Governance, Public Policy, and Sociopolitical Analysis   0.4998674277564509
"""


def compute_ece(confidences, correctness, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)  # fraction of samples in bin

        if prop_in_bin > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(correctness[in_bin])
            ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin

    return ece


def parse_and_compute_ece(data_str):
    model1_confidences = []
    model1_correct = []
    model2_confidences = []
    model2_correct = []

    for line in data_str.strip().split("\n"):
        parts = line.strip().split("\t")
        if len(parts) != 3:
            print(f"Skipping malformed line: {line}")
            continue

        true_label = parts[0].strip()

        # Parse model 1 prediction and confidence
        m1_parts = parts[1].strip().split()
        m1_conf = float(m1_parts[-1])
        m1_pred = " ".join(m1_parts[:-1])

        m1_corr = 1 if m1_pred == true_label else 0

        # Parse model 2 prediction and confidence
        m2_parts = parts[2].strip().split()
        m2_conf = float(m2_parts[-1])
        m2_pred = " ".join(m2_parts[:-1])

        m2_corr = 1 if m2_pred == true_label else 0

        model1_confidences.append(m1_conf)
        model1_correct.append(m1_corr)
        model2_confidences.append(m2_conf)
        model2_correct.append(m2_corr)

    # Convert lists to numpy arrays

    print(model1_confidences)
    print(model1_correct)
    print(model2_confidences)
    print(model2_correct)
    model1_confidences = np.array(model1_confidences)
    model1_correct = np.array(model1_correct)
    model2_confidences = np.array(model2_confidences)
    model2_correct = np.array(model2_correct)

    # Compute ECE for all samples (no mask)
    ece_model1 = compute_ece(model1_confidences, model1_correct)
    ece_model2 = compute_ece(model2_confidences, model2_correct)

    print(f"ECE for Model 1: {ece_model1:.4f}")
    print(f"ECE for Model 2: {ece_model2:.4f}")
    print(
        f"Model 1 correct predictions: {np.sum(model1_correct)} / {len(model1_correct)}")
    print(
        f"Model 2 correct predictions: {np.sum(model2_correct)} / {len(model2_correct)}")

    return {
        "model1": {"ece": ece_model1, "accuracy": np.mean(model1_correct)},
        "model2": {"ece": ece_model2, "accuracy": np.mean(model2_correct)},
    }


print(parse_and_compute_ece(data))
