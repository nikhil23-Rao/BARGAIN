
import numpy as np
import time
import pandas as pd
import random
from typing import List, Union, Tuple

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
        return response

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
        logprobs = response.choices[0].logprobs.content

        # if answer is something not listed in the class -> return a 0.0 confidence early
        if not any(c.lower().replace(" ", "") == response.choices[0].message.content.lower().replace(" ", "") for c in classes):
            print(response.choices[0].message.content, 0.0)
            return response.choices[0].message.content, 0.0

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
        return response.choices[0].message.content, confidence

    def proxy_func_general(self, data_record):
        task_with_data = self.task.format(data_record)
        prompt = [
            {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
            {"role": "user", "content": task_with_data}
        ]
        response = self.client.beta.chat.completions.parse(
            model=self.model, messages=prompt, logprobs=True, seed=0, temperature=0, max_tokens=1002, top_logprobs=10, top_p=1)
        if response.choices[0].logprobs is None:
            prob = 0
        else:
            logprobs = response.choices[0].logprobs.content
            all_logprobs = 0
            for t in logprobs:
                all_logprobs += t.logprob
            prob = np.exp(all_logprobs)

        print(response.choices[0].message.content, "  ", prob)
        return response.choices[0].message.content, prob

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
            return self.class_proxy_func(data_record,  ["lion", "tiger", "elephant", "giraffe", "zebra",
                                         "kangaroo", "panda", "koala", "dolphin", "whale",
                                                        "eagle", "falcon", "bear", "wolf", "fox",
                                                        "rabbit", "deer", "monkey", "hippopotamus", "rhinoceros"])


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
# Here are the potential categories: ["Business", "Sports", "World", "Sci/Tech"]
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
Here are the potential categories: ['TECH', 'MEDIA', 'ENVIRONMENT', 'SPORTS', 'CRIME', 'BUSINESS', 'SCIENCE', 'ARTS & CULTURE', 'ENTERTAINMENT', 'RELIGION', 'POLITICS', 'COMEDY', 'EDUCATION', 'WOMEN']
You must choose Only ONE CATEGORY out of the list, and return THAT CATEGORY ONLY. 
-----
here is the Article: {}
'''

print(task)


# Define oracle and proxy
proxy = OpenAIProxy(task, model='gpt-4o')
oracle = OpenAIOracle(task, model='gpt-4o')


# # Call BARGAIN to process
# print("starting process")

# bargain = BARGAIN_A(proxy, oracle, target=0.9,  delta=0.1, seed=0)
# df['output'] = bargain.process(df['value'].to_numpy())

# # Evaluate output
# df['is_correct'] = df['animal_name'] == df['output']
# print(
#     f"Accuracy: {df['is_correct'].mean()}, Used Proxy: {1-oracle.get_number_preds()/len(df):.2f}")


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


df = pd.read_csv("BARGAIN/examples/newtests/kaggle1.csv")

# shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True).head(1000)
# shuffled.to_csv("kaggle1.csv", index=False)


# print(df)

str_list = []
newstr_list = []

# for index, row in df.iterrows():
#     # Get the article text from the current row
#     current_article = str(row['title']) + " | " + str(row['body'])
#     # injecteda = row['category']
#     # print(injecteda)
#     # print(most_mentioned_animal_from_string(injecteda))

#     str_result = proxy.class_proxy_func(
#         current_article, ['TECH', 'MEDIA', 'ENVIRONMENT', 'SPORTS', 'CRIME', 'BUSINESS', 'SCIENCE', 'ARTS & CULTURE', 'ENTERTAINMENT', 'RELIGION', 'POLITICS', 'COMEDY', 'EDUCATION', 'WOMEN'])

#     str_list.append(str(str_result[0]) + "  " + str(str_result[1]))
#     # newstr_result = proxy.proxy_func_general(current_article)
#     # newstr_list.append(
#     #     str(newstr_result[0]) + "  " + str(newstr_result[1]))
#     time.sleep(1)

# # Option 1: Save as two CSV files
# pd.DataFrame({'ClassProxyResult': str_list}).to_csv(
#     'class_proxy_results.csv', index=False)
# pd.DataFrame({'GeneralProxyResult': newstr_list}).to_csv(
#     'general_proxy_results.csv', index=False)


model1_confidences = []
model1_correct = []
model2_confidences = []
model2_correct = []

data = """
science	science  0.9998620238065092	science  0.9998204720858084
entertainment	entertainment  0.9996861556004201	entertainment  0.9932234683865432
science	science  0.9999825220092511	science  0.9999999943972032
technology	technology  0.9913824727073883	technology  0.9968271328520341
technology	politics  0.6217998113676496	politics  0.622432067011902
world	world  0.9979752713271665	world  0.9980728215356096
world	politics  0.999999448776502	politics  0.9999999970010379
politics	politics  0.9999982567411922	politics  0.9999999950555533
technology	politics  0.9999522452651637	politics  0.9999546021280219
sports	sports  0.9999987335551229	sports  1.0
world	world  0.9045317249219155	world  0.9706541690163114
world	politics  0.999998137537802	politics  0.9999989322975598
politics	politics  0.9999989719621736	politics  1.0
politics	politics  0.999999448776502	politics  0.9999999985833904
world	world  0.9996196850411753	world  0.9996200155341546
entertainment	entertainment  0.9999998063873693	entertainment  1.0
politics	world  0.999580854842463	world  0.9999522737820985
technology	technology  0.9999995679801056	technology  0.9999999943972019
politics	politics  0.9999998063873693	politics  1.0
world	politics  0.9999540332370438	politics  0.9999485578070847
sports	sports  0.9999989719621736	sports  1.0
politics	politics  0.9999992103693378	politics  0.9999999985833901
entertainment	politics  0.9999732243287754	politics  0.9999756916058542
world	science  0.9999840716313717	science  0.999991440996135
entertainment	entertainment  0.9999987335551229	entertainment  0.9999957771667455
entertainment	politics  0.9999874092856329	politics  0.9999919004958266
sports	sports  0.999997779927489	sports  0.9999999997210531
world	science  0.999696996139406	science  0.9997387105378425
world	world  0.9999971839107363	world  0.9999970976891832
technology	technology  0.9999995679801056	technology  1.0
automobile	automobile  0.9999980631288848	automobile  0.9999987901339381
entertainment	entertainment  0.9999996871837232	entertainment  1.0
entertainment	entertainment  0.9999992103693378	entertainment  1.0
science	science  0.9999919389781311	science  0.9999997814401964
technology	politics  0.994745235899965	politics  0.9990880968685629
technology	technology  0.9999953958623826	technology  1.0
technology	technology  0.9999976607241555	technology  0.999999997646591
sports	sports  0.9999982567411922	sports  0.999999999868235
technology	technology  0.9999418750971238	technology  0.9999442360771842
technology	politics  0.9999963494877431	politics  0.9999999092153509
sports	sports  0.9999995679801056	sports  0.9999999998682348
sports	sports  0.999999091165777	sports  0.9999999999294713
entertainment	politics  0.9988270098222798	politics  0.9995116379647999
technology	technology  0.9999521260688496	technology  0.9999582653802959
technology	technology  0.9999572515923018	technology  0.9999595893381978
entertainment	entertainment  0.999999493571153	entertainment  1.0
automobile	automobile  0.9999972287051722	automobile  0.9999989322973052
world	politics  0.9997056954112391	politics  0.9998924210415887
entertainment	politics  0.9953818084611212	politics  0.9978162465456809
sports	sports  0.9999962302845802	sports  0.9999999750211083
entertainment	entertainment  0.9999996871837232	entertainment  1.0
entertainment	entertainment  0.999999448776502	entertainment  1.0
technology	entertainment  0.9996088453696297	entertainment  0.9997387774795389
politics	politics  0.999997779927489	politics  0.9999999386901113
entertainment	entertainment  0.9999996871837232	entertainment  1.0
world	science  0.9999692907208221	science  0.9999976121564919
sports	sports  0.9999995679801056	sports  1.0
politics	politics  0.9999978991308366	politics  0.9999999881388777
world	politics  0.9999925349917576	politics  0.9999756997632746
politics	politics  0.999999091165777	politics  0.9999999973534239
sports	entertainment  0.998498172958146	entertainment  0.9984987647970694
politics	politics  0.9999996871837232	politics  1.0
science	science  0.9999870516787017	science  0.9999994216749389
entertainment	entertainment  0.9999996871837232	entertainment  1.0
entertainment	entertainment  1.0	entertainment  1.0
world	science  0.9046259450543088	science  0.9046499537527232
entertainment	entertainment  0.9999996871837232	entertainment  1.0
sports	sports  0.9999988527586979	sports  0.9999999998308103
technology	automobile  0.9994385907282414	automobile  0.9999852196257912
sports	sports  0.9999992103693378	sports  1.0
entertainment	entertainment  0.9999998063873693	entertainment  1.0
automobile	automobile  0.9999993743675585	automobile  1.0
science	science  0.9993754421066019	science  0.9996200154981933
world	politics  0.999999091165777	politics  1.0
science	science  0.9999720323248966	science  1.0
automobile	automobile  0.9999985399427375	automobile  1.0
world	world  0.9999908661546805	world  0.9999919982072509
entertainment	entertainment  0.9999998063873693	entertainment  1.0
world	politics  0.9999919389781311	politics  0.9999952149034396
sports	sports  0.9999897933314713	sports  0.9999962200111138
world	science  0.9931323744001624	science  0.9933052885027995
science	world  0.9999213734689336	world  0.9999236004745512
technology	technology  0.9999975415208362	technology  1.0
sports	sports  0.999999091165777	sports  0.9999999988967437
science	technology  0.8515961538789503	technology  0.677036607116563
entertainment	entertainment  0.9997012861790182	entertainment  0.9980732650122859
politics	politics  0.9999995679801056	politics  1.0
world	world  0.9999920581810099	world  0.9999933785023049
world	politics  0.9999949190498161	politics  0.999998005269655
world	politics  0.9999996871837232	politics  0.9999999847700157
technology	technology  0.999999448776502	technology  1.0
sports	entertainment  0.731054678550306	sports  0.562176152078593
entertainment	entertainment  0.999997064707474	entertainment  0.9999996533674358
entertainment	entertainment  0.9999995679801056	entertainment  1.0
entertainment	entertainment  0.9999996871837232	entertainment  1.0
entertainment	entertainment  0.9999995679801056	entertainment  1.0
politics	politics  0.9999982567411922	politics  0.9999999847700229
politics	politics  0.9992744879943766	politics  0.9980720600731209
technology	politics  0.9996975919763319	politics  0.9999970976891822
entertainment	entertainment  1.0	entertainment  1.0
technology	politics  0.9999944422379315	politics  0.999997093325744
entertainment	entertainment  0.9999996871837232	entertainment  1.0
entertainment	politics  0.9999577283873626	politics  0.9999599062432757
science	science  0.9999355576902692	science  0.9999995820579832
sports	sports  0.9999989719621736	sports  1.0
politics	politics  0.999999448776502	politics  0.9999999985833904
sports	sports  0.9999987335551229	sports  1.0
automobile	automobile  0.9999992551639639	automobile  0.9999999936511972
technology	technology  0.9999984951481292	technology  0.9999998362623211
sports	politics  0.9990856170417217	politics  0.9988300493187479
entertainment	entertainment  0.9999984951481292	entertainment  1.0
world	politics  0.9999998063873693	politics  0.9999999997538304
technology	politics  0.9155447788898323	politics  0.9967709244309365
politics	politics  0.9999998063873693	politics  1.0
entertainment	entertainment  0.9999998063873693	entertainment  1.0
sports	sports  0.9999986143516758	sports  0.9999999998308103
sports	sports  0.9999992103693378	sports  1.0
entertainment	entertainment  0.999994725438169	entertainment  0.9999952149062914
technology	technology  0.8175443033428308	technology  0.8175742769524458
sports	sports  0.9999995679801056	sports  0.9999999999294712
technology	politics  0.9998554685272633	politics  0.9999992750609523
sports	sports  0.999997303114013	sports  0.999999761447041
science	science  0.9998776374486098	science  0.9984988082623636
entertainment	entertainment  0.9999998063873693	entertainment  1.0
world	world  0.9999993295729128	world  0.9999999778405064
sports	sports  0.9999993295729128	sports  1.0
technology	technology  0.9998487941142732	technology  0.9998742185067865
entertainment	entertainment  0.999668403658134	entertainment  0.9996727921091261
entertainment	entertainment  0.9999998063873693	entertainment  1.0
technology	politics  0.9999955150656735	politics  0.9999994956525544
technology	politics  0.9999956342685238	politics  0.9999998144608844
automobile	politics  0.8296140154024607	politics  0.9991901691077303
entertainment	entertainment  0.48082572508643784	entertainment  0.4808554009900919
technology	technology  0.999287940934704	technology  0.9995694380444314
automobile	automobile  0.9999787523592636	automobile  0.9999852437648803
world	world  0.9991945066903416	world  0.9986749774455297
politics	politics  0.9999993295729128	politics  0.9999999981810374
politics	politics  0.9999996871837232	politics  0.9999999936511964
world	world  0.7216956987357867	science  0.546724154279772
world	politics  0.9999995679801056	politics  0.9999999397642392
world	politics  0.9999975415208362	politics  0.9999986290423645
science	science  0.9999826412110073	science  0.9999999715466472
world	politics  0.9994465245032937	politics  0.9995694429845969
politics	politics  0.9999987335551229	politics  1.0
politics	politics  0.9999992103693378	politics  0.999999998394771
entertainment	entertainment  0.9999992103693378	entertainment  1.0
world	politics  0.9999996871837232	politics  0.9999999992417438
sports	politics  0.9395778517897807	sports  0.6782668013460019
politics	politics  0.9999992103693378	politics  0.9999999992417434
science	science  0.9999843100334488	science  0.9999989322966695
technology	politics  0.9999810915911035	politics  0.9999997279829378
technology	technology  0.9999887205103225	technology  0.9999952132117575
world	world  0.7310567919047004	world  0.6791780667747438
world	politics  0.9999931310057394	politics  0.9999989322971785
politics	politics  0.9999993295729128	politics  1.0
world	politics  0.9999944422379315	politics  0.9999965667540363
entertainment	entertainment  0.999999612774776	entertainment  1.0
politics	politics  0.9999993295729128	politics  0.9999999970010371
politics	politics  0.999999448776502	politics  0.999999999330841
world	politics  0.9999996871837232	politics  0.9999999976644063
sports	sports  0.9999986143516758	sports  1.0
sports	sports  0.999999448776502	sports  0.9999999979388478
entertainment	entertainment  0.9999996871837232	entertainment  1.0
automobile	technology  0.9999821644040676	technology  0.999972464310156
automobile	automobile  0.9980694636210206	automobile  0.9999545698782462
world	entertainment  0.998496030415804	entertainment  0.977022627413796
technology	world  0.9044043613743277	world  0.8172247309645242
world	politics  0.9999995679801056	politics  0.999999836262243
sports	sports  0.9999993295729128	sports  1.0
entertainment	entertainment  0.999999448776502	entertainment  1.0
world	science  0.9997365532199805	science  0.9999850972901355
technology	science  0.923950721189477	science  0.9046496920372468
politics	politics  0.9999998063873693	politics  1.0
world	world  0.8517453098248156	world  0.5621653706559238
world	world  0.9999993295729128	world  0.9999999804443108
entertainment	entertainment  1.0	entertainment  1.0
entertainment	entertainment  0.999997303114013	entertainment  0.9999980052706059
sports	sports  0.9999980183344259	sports  1.0
entertainment	politics  0.9999976607241555	politics  0.9999984465015744
politics	politics  0.999988958915327	politics  0.9999990577551116
technology	politics  0.9972967445544979	politics  0.9984931072825006
world	politics  0.9999971839107363	politics  0.9999999715466337
sports	sports  0.9999987335551229	sports  0.9999999997210534
politics	politics  0.9999986143516758	politics  0.9999999510272283
automobile	automobile  0.9999447806125014	automobile  0.9999468888037542
entertainment	entertainment  0.9999995679801056	entertainment  0.9999999895325951
sports	sports  0.9999962302845802	sports  0.9999999715466302
politics	politics  0.9999971839107363	politics  0.9999999305268614
science	science  0.9989267994313638	science  0.9958988015753472
technology	politics  0.9999983759447105	politics  0.9999996533674771
politics	politics  0.9999951574562983	politics  0.9999994956527347
sports	sports  0.9999937270200764	sports  0.9999994825563299
politics	politics  0.9999986143516758	politics  0.9999999647591831
technology	science  0.9999771579522025	science  0.9999999530883418
world	world  0.9997654978328642	world  0.9998910258070302
technology	technology  0.999999091165777	technology  0.9999998632435072
world	entertainment  0.49992108872563473	entertainment  0.499961758234936
sports	sports  0.9999992103693378	sports  1.0
world	world  0.985359588110206	world  0.98568856265167
world	politics  0.999997064707474	politics  0.99999521490458
technology	technology  0.9999998063873693	technology  0.999999999782756
science	science  0.9999752507380847	science  0.9999999909989362
science	science  0.9999846676393996	science  0.9999989195468275
technology	politics  0.9997016436710272	politics  0.999915188921987
entertainment	entertainment  0.9999995679801056	entertainment  1.0
science	world  0.9958623955385857	world  0.9912840330833236
entertainment	entertainment  0.9999998063873693	entertainment  1.0
technology	technology  0.9999995679801056	technology  0.9999999936511972
technology	technology  0.9999473581624593	technology  0.9999953843827902
science	science  0.9999590395731335	science  0.9999998362623214
entertainment	entertainment  0.9999996871837232	entertainment  1.0
science	technology  0.8519465016903714	science  0.9046501883114898
technology	technology  0.9999996871837232	technology  0.9999999973534258
world	world  0.9999996871837232	world  1.0
world	politics  0.9999967070975446	politics  0.9999979638713631
world	politics  0.9413387038763679	politics  0.9195845309308718
entertainment	entertainment  0.9999996871837232	entertainment  1.0
entertainment	entertainment  0.9999974223173038	entertainment  0.9999980052706059
science	science  0.9996630449319085	science  0.9997040427785489
sports	sports  0.9999982567411922	sports  0.9999999966842691
technology	automobile  0.9996177041132345	automobile  0.9996199030845251
world	world  0.9998107819607386	world  0.9999917179540285
technology	technology  0.9999405639338598	technology  0.9999417087162682
world	world  0.994521042263202	world  0.9924960558504508
sports	sports  0.9999993295729128	sports  1.0
entertainment	entertainment  0.9999996871837232	entertainment  1.0
sports	sports  0.9999993295729128	sports  1.0
technology	technology  0.9986144137004188	technology  0.9985598088386379
entertainment	entertainment  0.9999996871837232	entertainment  1.0
entertainment	entertainment  0.5401493324822229	entertainment  0.47777091751554707
science	science  0.9999132682857789	science  0.9999998362623023
science	science  0.9999853828507753	science  0.9999919802406577
world	world  0.8114465210193982	world  0.678936644277554
automobile	automobile  0.999962898734859	automobile  0.9999644435052599
sports	sports  0.9999992103693378	sports  1.0
technology	politics  0.9999984951481292	politics  0.9999999992417443
world	politics  0.999999448776502	politics  0.9999999995941345
world	politics  0.9999677411216067	politics  0.9999687980937707
technology	science  0.9816985463908783	science  0.9819639747859215
sports	sports  0.9999978991308366	sports  0.9999992661788775
politics	politics  0.9999996871837232	politics  1.0
sports	sports  0.9999987335551229	sports  0.999999977840525
world	science  0.9942176352970272	science  0.9947798741081557
science	science  0.9999925349917576	science  0.9999999976465915
entertainment	politics  0.6224237320344193	politics  0.7310350623567452
world	politics  0.9999765619468316	politics  0.9999785550602326
sports	sports  0.9999993295729128	sports  0.9999999992417433
politics	politics  0.999997303114013	politics  0.9999999677581389
entertainment	entertainment  0.9999998063873693	entertainment  1.0
entertainment	entertainment  1.0	entertainment  1.0
world	politics  0.9889270457518086	politics  0.9990828156374282
sports	sports  0.9999930118027327	sports  0.9999990505609969
sports	sports  0.9999983759447105	sports  1.0
world	politics  0.999998137537802	politics  0.9999982396570756
sports	sports  0.9999993295729128	sports  0.9999999999705992
sports	sports  0.9999984951481292	sports  0.9999999987498477
world	world  0.9999944422379315	world  0.9999961554495218
entertainment	entertainment  0.9999995679801056	entertainment  1.0
sports	sports  0.9999975415208362	sports  0.999999999641825
automobile	automobile  0.8805307305474357	automobile  0.8807709398015549
politics	politics  0.9999568939979742	politics  0.9999687980789217
sports	sports  0.9999988527586979	sports  1.0
technology	science  0.9999787075660107	science  0.9999947413725023
entertainment	entertainment  0.9999987335551229	entertainment  1.0
technology	technology  0.7772524021405924	science  0.8175719728677742
sports	sports  0.9999987335551229	sports  1.0
entertainment	entertainment  0.9968269734921458	entertainment  0.9241412725577554
world	politics  0.9999964686909203	politics  0.9999974387185151
politics	politics  0.9999996871837232	politics  1.0
sports	sports  0.9999995679801056	sports  1.0
world	politics  0.999991104559287	politics  0.9999921107349491
science	science  0.9999591587702716	science  1.0
world	entertainment  0.9953865628044604	entertainment  0.9953900484759729
world	sports  0.7772930790601773	sports  0.7772956224494284
entertainment	entertainment  0.9999993743675585	entertainment  1.0
world	politics  0.9999479541486054	politics  0.9998911030564758
sports	sports  0.9999978991308366	sports  0.999999772977441
world	politics  0.9999846676393996	politics  0.9999251537582656
technology	politics  0.8929335921410886	politics  0.8918459212728491
technology	politics  0.9295228814196062	politics  0.9958716329070708
sports	sports  0.9999983759447105	sports  1.0
politics	politics  0.999999091165777	politics  1.0
politics	politics  0.9999992103693378	politics  1.0
science	science  0.9706036918229505	science  0.9626731153331316
sports	sports  0.9999976607241555	sports  1.0
technology	politics  0.9999944422379315	politics  0.9998750328112115
politics	politics  0.9999993295729128	politics  1.0
sports	sports  0.999998137537802	sports  1.0
politics	politics  0.999997779927489	politics  0.999999951406477
world	entertainment  0.9999996871837232	entertainment  0.9999998144608401
science	science  0.9996630449319085	science  0.9997040427785489
sports	sports  0.9999995679801056	sports  1.0
politics	politics  0.9999995679801056	politics  0.999999999140783
world	politics  0.9999860980626626	politics  0.9999869928767764
world	technology  0.9890091360243637	technology  0.9890129180328536
entertainment	politics  0.9999964686909203	politics  0.9999982223982685
automobile	automobile  0.9999981823323232	automobile  0.9999996072137237
sports	sports  0.9819956128282563	sports  0.9525741077402083
automobile	automobile  0.9999978247221077	automobile  0.9999991536440606
sports	sports  0.9999978991308366	sports  0.9999999161956873
politics	politics  0.999999448776502	politics  1.0
world	politics  0.9999996871837232	politics  0.9999999992417438
technology	technology  0.9997948118226208	technology  0.998073251823987
sports	sports  0.999999448776502	sports  1.0
entertainment	entertainment  0.999999448776502	entertainment  0.9999999317439333
world	politics  0.9999996871837232	politics  0.999999993651198
technology	technology  0.9999995679801056	technology  1.0
sports	sports  0.9999983759447105	sports  1.0
technology	automobile  0.9998396653162127	automobile  0.9991958150375266
sports	sports  0.9999978991308366	sports  1.0
world	entertainment  0.9999974223173038	entertainment  0.9999992398702936
world	world  0.9999998063873693	world  1.0
world	world  0.9999149369997226	world  0.9996200155341546
world	science  0.9624829065317214	science  0.9241292338766482
technology	politics  0.9993353219870835	politics  0.9993588460120367
automobile	automobile  0.9999990167568458	automobile  0.9999998874647759
sports	sports  0.9999764427476189	sports  0.9999920010859148
politics	politics  0.9999992103693378	politics  0.999999996149255
entertainment	entertainment  0.9999996871837232	entertainment  1.0
world	entertainment  0.9999995679801056	entertainment  1.0
sports	sports  0.999999091165777	sports  1.0
technology	politics  0.9999938462231682	politics  0.9999945777786889
politics	politics  0.999999448776502	politics  0.9999999819839064
entertainment	entertainment  0.9999996871837232	entertainment  1.0
entertainment	entertainment  0.9999996871837232	entertainment  1.0
politics	politics  0.9999995679801056	politics  0.9999999979388453
world	politics  0.9999716747234644	politics  0.9999756997516839
sports	sports  0.9999987335551229	sports  0.9999999999094393
entertainment	entertainment  0.9999996871837232	entertainment  1.0
technology	entertainment  0.9999963494877431	entertainment  0.9999957240093473
technology	technology  0.9994397373135924	technology  0.9994469232067343
world	politics  0.9875543128441154	politics  0.9875661430483508
entertainment	entertainment  1.0	entertainment  1.0
entertainment	entertainment  0.9999998063873693	entertainment  1.0
politics	politics  0.9999992103693378	politics  1.0
world	science  0.9930867552932566	science  0.9933067084331592
technology	world  0.7846384283267412	world  0.7585455083142912
sports	sports  0.9999988527586979	sports  0.999999999868235
sports	sports  0.9999897933314713	sports  0.9999999586005869
sports	sports  0.9999988527586979	sports  1.0
world	world  0.9706613001476925	world  0.9706673385367722
entertainment	entertainment  0.9999996871837232	entertainment  1.0
technology	technology  0.9999951574562983	technology  0.999999540832263
world	politics  0.9980695414770505	politics  0.9990888716121434
politics	politics  0.9999993295729128	politics  0.9999999907625442
politics	politics  0.9999992103693378	politics  0.9999999918479856
entertainment	entertainment  0.9999984951481292	entertainment  0.9999999006881067
world	world  0.9999993295729128	world  0.999999984770014
world	politics  0.998829867377086	politics  0.9988304895217126
entertainment	entertainment  0.9999996871837232	entertainment  1.0
sports	sports  0.9999984951481292	sports  0.9999599362953181
world	politics  0.9998461757073605	politics  0.9999485577703126
entertainment	entertainment  0.9999998063873693	entertainment  1.0
entertainment	entertainment  0.9999996871837232	entertainment  1.0
entertainment	entertainment  0.9999996871837232	entertainment  1.0
science	science  0.9996790093284479	science  0.9997039296623333
entertainment	entertainment  0.999999448776502	entertainment  1.0
entertainment	entertainment  0.9999996871837232	entertainment  1.0
world	world  0.9726980315050993	world  0.966644568435901
entertainment	entertainment  0.999999493571153	entertainment  1.0
automobile	automobile  0.99999484464138	automobile  0.9999966429947703
science	science  0.9998309200947905	science  0.9999646241798499
technology	technology  0.9996573251213864	technology  0.9996645372876998
technology	entertainment  0.9999996871837232	entertainment  1.0
automobile	technology  0.4402417407888257	technology  0.573205097089311
automobile	automobile  0.9932977610075414	automobile  0.9933022799549505
world	world  0.9999968263007644	world  0.999998958443429
sports	sports  0.9999984951481292	sports  0.9999999457389163
entertainment	entertainment  0.999999612774776	entertainment  1.0
world	entertainment  0.999997064707474	entertainment  0.9999996072138173
world	politics  0.9859335430652338	politics  0.9859363574499695
sports	sports  0.999999091165777	sports  1.0
sports	sports  0.9999988527586979	sports  1.0
sports	sports  0.9999989719621736	sports  0.9999999999094393
entertainment	entertainment  0.9999998063873693	entertainment  1.0
technology	technology  0.9999995679801056	technology  1.0
sports	sports  0.9999989719621736	sports  0.9999999998973813
sports	sports  0.9999993295729128	sports  1.0
world	world  0.9701986769551807	world  0.9944906008274109
entertainment	entertainment  0.9999995679801056	entertainment  1.0
technology	politics  0.9999967070975446	politics  0.999999324828145
sports	sports  0.9998896682925232	sports  0.9999646437340446
world	world  0.9959269504097475	world  0.9959298618065433
sports	sports  0.9999986143516758	sports  0.9999999997210534
sports	entertainment  0.9923834746852109	sports  0.9975215532539417
sports	entertainment  0.9999992551639781	entertainment  1.0
sports	sports  0.9999945614411088	sports  0.9999999918479807
technology	politics  0.6789277531926196	politics  0.8932774392698529
politics	politics  0.9999996871837232	politics  1.0
world	science  0.7310420639124862	science  0.7310575817833787
sports	sports  0.999999448776502	sports  0.9999999999200804
entertainment	entertainment  0.9999989719621736	entertainment  0.9999952149074319
technology	technology  0.9999996871837232	technology  0.9999999998682348
sports	sports  0.999997064707474	sports  0.9999995544495182
technology	science  0.9913962973279161	science  0.9914171165203384
science	science  0.9999684563217857	science  0.9999999891267414
sports	entertainment  0.9999539140405167	entertainment  0.9999679666024968
entertainment	entertainment  0.9999998063873693	entertainment  1.0
world	science  0.9238874183762867	science  0.9706837890245315
technology	technology  0.9999806147849026	technology  0.9999852610186963
world	politics  0.9998756112425662	politics  0.9999417087301645
world	politics  0.999971913124405	politics  0.9999724643200028
politics	science  0.9982149122965097	science  0.9999837045652957
sports	sports  0.9999993295729128	sports  0.9999999999666845
world	world  0.9999735819307618	world  0.9999846134466938
technology	politics  0.999997303114013	politics  0.9999998724809325
world	politics  0.999989435724597	politics  0.999959936319194
entertainment	entertainment  0.8519502340262392	entertainment  0.9947797904391463
world	science  0.9936085653587158	science  0.9968273175212737
technology	politics  0.9999051631755985	politics  0.9998576850052522
technology	politics  0.9999992103693378	politics  1.0
world	politics  0.9999996871837232	politics  0.999999982742167
technology	politics  0.9998364025480796	politics  0.999999197923329
world	world  0.999992892600195	world  0.9999997646466653
sports	sports  0.9999995679801056	sports  0.9999999999450722
entertainment	entertainment  0.9999888397128177	entertainment  0.9999969983766986
world	politics  0.9770181781323494	politics  0.9706877684006775
automobile	automobile  0.9999981823323232	automobile  0.9999982847029738
sports	sports  0.9999995679801056	sports  1.0
politics	politics  0.9999974223173038	politics  0.9999999764239087
science	science  0.9913954774291953	science  0.9933071162702313
world	politics  0.9999988527586979	politics  0.9999996940977138
politics	politics  0.999999448776502	politics  1.0
technology	politics  0.9999988527586979	politics  0.9999999677581581
sports	entertainment  0.9999583243796896	entertainment  0.9999599363191958
sports	sports  0.999999091165777	sports  0.9999999998082827
sports	sports  0.9999988527586979	sports  0.9999999998973813
politics	politics  0.9999993295729128	politics  1.0
politics	politics  0.9999949190498161	politics  0.9999990577551116
entertainment	entertainment  0.9999998063873693	entertainment  1.0
automobile	automobile  0.9997361213082742	automobile  0.9998600035506494
world	politics  0.9999845484374018	politics  0.9999910603030341
politics	politics  0.9999975415208362	politics  0.9999999305268614
world	world  0.9990879989921388	world  0.9990889485983874
automobile	automobile  0.9999984207392991	automobile  1.0
science	science  0.9998958662125743	science  0.9999723005848149
sports	sports  0.999996587894339	sports  1.0
entertainment	entertainment  0.9999998063873693	entertainment  1.0
world	politics  0.9999984951481292	politics  0.9999977396760732
science	science  0.9397205762938434	science  0.851950532956997
science	science  0.9999007530708943	science  0.9998204720002077
science	science  0.9999831180163555	science  0.9999971209572263
politics	politics  0.999999448776502	politics  0.999999999026379
world	politics  0.9996075346394109	politics  0.9998765689348423
politics	politics  0.9999996871837232	politics  1.0
science	science  0.9999589203760095	science  1.0
technology	politics  0.999999091165777	politics  1.0
science	science  0.9998690558742386	science  0.999989870010696
world	world  0.999999448776502	world  1.0
entertainment	entertainment  0.9999967070975446	entertainment  0.9999977396747259
sports	sports  0.9999993295729128	sports  0.9999999999200804
world	world  0.999999448776502	world  1.0
entertainment	entertainment  0.9999996871837232	entertainment  1.0
world	politics  0.9999968263007644	politics  0.9999999956365405
automobile	automobile  0.9999990167568458	automobile  0.999999974302605
sports	sports  0.9999993295729128	sports  0.9999999999450722
entertainment	entertainment  0.9999996871837232	entertainment  1.0
politics	politics  0.999999448776502	politics  0.999999999330841
science	science  0.9694000385513554	science  0.9902840484184401
sports	sports  0.9999988527586979	sports  1.0
world	politics  0.995966917519599	politics  0.9989488829640133
technology	technology  0.9997833743490786	technology  0.9999881746052579
sports	entertainment  0.9999993743675585	entertainment  0.9999999634651591
sports	sports  0.9999993295729128	sports  1.0
world	technology  0.9999944422379315	technology  0.9999756472556653
world	science  0.999257710898426	science  0.9998015395734016
science	science  0.9999846676393996	science  0.9999999829510576
technology	technology  0.9999998063873693	technology  0.9999999996418252
entertainment	entertainment  0.9999996871837232	entertainment  1.0
technology	technology  0.6146522992893749	entertainment  0.5616644875961069
technology	politics  0.9984780835495043	politics  0.9990880968685629
sports	sports  0.999999448776502	sports  0.9999999998837176
sports	sports  0.9999989719621736	sports  0.9999999998308103
science	politics  0.9980508884145404	politics  0.9704946194495481
technology	entertainment  0.9778704450034684	entertainment  0.992438340126868
world	politics  0.9999959918780695	politics  0.9999957771657387
entertainment	entertainment  0.9999996871837232	entertainment  1.0
science	world  0.8812807534235297	world  0.848994842322946
politics	politics  0.999999448776502	politics  1.0
technology	technology  0.999999448776502	technology  1.0
world	politics  0.9999995679801056	politics  0.9999999992417437
sports	sports  0.9999961110814314	sports  0.9999999397642608
science	science  0.9999849060434378	science  0.9999999939372983
technology	world  0.9044043613743277	world  0.9045296602297053
science	science  0.9999261412514249	science  0.9999351921454726
entertainment	entertainment  1.0	entertainment  1.0
technology	technology  0.9999978991308366	technology  1.0
technology	politics  0.9999963494877431	politics  0.999997093325744
entertainment	entertainment  0.6224562133588518	entertainment  0.6224573538612461
science	science  0.5620961553337784	science  0.5621763202673481
world	politics  0.9999585627774474	politics  0.9998204720208086
entertainment	entertainment  0.9989604654836198	entertainment  0.9980732650122859
entertainment	entertainment  0.9999996871837232	entertainment  1.0
science	science  0.9999741779325377	science  0.9999967105815402
technology	technology  0.9999974223173038	technology  0.9999999434520023
sports	sports  0.9999986143516758	sports  1.0
technology	science  0.9914134027119484	science  0.9046501883114898
technology	politics  0.660194377554224	politics  0.9975729562813339
technology	technology  0.9999996871837232	technology  0.9999999804443132
entertainment	entertainment  0.9999995679801056	entertainment  0.9999999397642392
science	science  0.9999756083407958	science  0.9999998724810085
world	politics  0.999999448776502	politics  0.9999999970010375
entertainment	entertainment  0.9999996871837232	entertainment  1.0
world	politics  0.8519508433887499	politics  0.8807969993938691
technology	technology  0.9994397373135924	technology  0.9994469232067343
entertainment	entertainment  0.999999448776502	entertainment  0.9999996940978233
world	science  0.9239765639716914	science  0.9706832636909101
world	politics  0.9999992103693378	politics  0.9999996072138642
entertainment	entertainment  0.9999996871837232	entertainment  1.0
world	politics  0.9999996871837232	politics  0.9999999996418251
sports	entertainment  0.9241372955929888	entertainment  0.9241392905637926
entertainment	entertainment  0.9999996871837232	entertainment  1.0
entertainment	entertainment  0.9999996871837232	entertainment  1.0
sports	sports  0.9999995679801056	sports  1.0
world	world  0.975717403737737	world  0.9910470612901257
world	politics  0.9999995679801056	politics  0.9999997617631291
entertainment	entertainment  0.9999996871837232	entertainment  1.0
entertainment	entertainment  0.9999996871837232	entertainment  1.0
world	science  0.9998703669473757	science  0.999948409932427
technology	technology  0.998109210080282	technology  0.9983870101220393
world	politics  0.9999891973194788	politics  0.9999973394061682
sports	sports  0.9999976607241555	sports  0.9999999998082828
entertainment	entertainment  0.9999998063873693	entertainment  1.0
entertainment	entertainment  0.9999998063873693	entertainment  1.0
world	politics  0.99999158136958	politics  0.9999921107387107
world	world  0.9914205538951522	world  0.9914222067568571
sports	sports  0.9999987335551229	sports  0.999999999808283
entertainment	entertainment  0.9999995679801056	entertainment  1.0
entertainment	entertainment  0.9999995679801056	entertainment  1.0
entertainment	entertainment  0.9999996871837232	entertainment  1.0
world	entertainment  0.9914192603073838	entertainment  0.9890124259301699
sports	sports  0.9999993295729128	sports  1.0
sports	sports  0.999999448776502	sports  1.0
politics	politics  0.9999996871837232	politics  1.0
automobile	automobile  0.9999993743675585	automobile  1.0
entertainment	entertainment  1.0	entertainment  1.0
technology	technology  0.9999995679801056	technology  0.9999999995941347
sports	sports  0.9999921773839029	sports  0.9999990691601192
world	world  0.9525504290235206	world  0.9706673385367722
sports	sports  0.9999986143516758	sports  1.0
politics	politics  0.9999998063873693	politics  1.0
technology	politics  0.997523846969743	politics  0.9975269335932745
automobile	automobile  0.9999398935442898	automobile  0.9999414388247825
entertainment	entertainment  0.9999655955277769	entertainment  0.9999995080031399
science	science  0.9940543392904475	science  0.9947796326146473
science	science  0.9999595163690469	science  0.9999999973534247
world	politics  0.9999969455039983	politics  0.9999974387175992
entertainment	entertainment  0.9999996871837232	entertainment  1.0
automobile	automobile  0.9932236601767735	automobile  0.9706736133284704
entertainment	entertainment  0.9999995679801056	entertainment  0.999999992805863
sports	sports  0.9999934886143899	sports  0.9999999993308415
entertainment	entertainment  0.9999998063873693	entertainment  1.0
entertainment	entertainment  0.9999996871837232	entertainment  1.0
world	politics  0.9999996871837232	politics  0.9999999995400943
entertainment	entertainment  0.9999971839107363	entertainment  0.9999991684717123
sports	sports  0.9999995679801056	sports  1.0
technology	technology  0.999997779927489	technology  0.9999999121883073
sports	sports  0.9999983759447105	sports  0.9999999998682348
automobile	automobile  0.9999990167568458	automobile  1.0
technology	automobile  0.999962898734859	automobile  0.9999640401386644
technology	entertainment  0.9869009043467659	entertainment  0.992438340126868
sports	sports  0.9999995679801056	sports  1.0
science	science  0.9999796611750019	science  1.0
entertainment	entertainment  0.9999998063873693	entertainment  1.0
sports	sports  0.999998137537802	sports  1.0
science	science  0.999978111559716	science  0.999999997923125
sports	entertainment  0.9975264593674983	entertainment  0.9975272534946519
entertainment	entertainment  1.0	entertainment  1.0
technology	politics  0.9999932502087602	politics  0.9999977989092962
sports	sports  0.9999986143516758	sports  0.9999999987498472
world	politics  0.9999989719621736	politics  0.9999999970010418
politics	politics  0.9999978991308366	politics  0.9999999895326
technology	technology  0.999999448776502	technology  0.9999999842881295
science	science  0.6221435095782175	science  0.6224592567455316
politics	politics  0.9999996871837232	politics  1.0
technology	politics  0.48160999925283415	politics  0.6656445266856921
technology	technology  0.9999998063873693	technology  0.9999999996418252
sports	sports  0.9999982567411922	sports  0.9999999586006313
politics	entertainment  0.9999995679801056	entertainment  0.9999998874647893
politics	politics  0.9999996871837232	politics  1.0
world	politics  0.999999448776502	politics  1.0
sports	sports  0.9999987335551229	sports  1.0
politics	politics  0.999999448776502	politics  0.9999999928058622
technology	politics  0.9999971839107363	politics  0.9999999778404906
entertainment	entertainment  0.9999995679801056	entertainment  1.0
technology	technology  0.9999939654258196	technology  0.9999966421936308
sports	sports  0.999998137537802	sports  0.9999999973534243
technology	politics  0.999998137537802	politics  0.999999836262243
science	science  0.9996531544383603	science  0.9995689025185006
technology	politics  0.999967383523528	politics  0.9999863760106301
world	politics  0.9999396103603339	politics  0.9999485577703126
technology	politics  0.4976954674941458	technology  0.850673507910895
world	politics  0.9999825220092511	politics  0.9999869928659224
entertainment	entertainment  0.9999995679801056	entertainment  1.0
world	world  0.9975264593674983	world  0.9975273487458829
technology	technology  0.9998410507423425	technology  0.999737326896907
sports	sports  0.9999993295729128	sports  0.9999999999515263
technology	technology  0.999982402807509	technology  0.9999957771677594
science	science  0.999936153669382	science  0.9999599363096863
automobile	automobile  0.99999818233238	automobile  0.9999997300421704
politics	politics  0.999999091165777	politics  0.9999999928058716
world	politics  0.9999986143516758	politics  0.9999984465028707
science	science  0.9984817695842013	science  0.9984988070361024
technology	technology  0.9999967070975446	technology  0.9999997063102423
sports	sports  0.9999995679801056	sports  1.0
technology	politics  0.9999992103693378	politics  0.9999998362622625
science	science  0.999982402807509	science  0.9999996072138178
technology	technology  0.9999995679801056	technology  0.9999999943972006
technology	technology  0.9999976607241555	technology  1.0
world	world  0.9999988527586979	world  0.9999998362623016
technology	technology  0.9999995679801056	technology  0.999999997664406
automobile	politics  0.5614695196076619	politics  0.9992096550431036
technology	politics  0.9999939654258196	politics  0.9999993524049995
sports	sports  0.9999978991308366	sports  0.9999999995400947
science	science  0.9997759853506765	science  0.9997965106669684
entertainment	politics  0.6791703167277879	politics  0.6791782260620198
world	politics  0.9999996871837232	politics  0.9999999992417437
entertainment	entertainment  0.9999996871837232	entertainment  1.0
technology	politics  0.9999968263007644	politics  0.9999999947827708
politics	politics  0.9999995679801056	politics  0.99999985550202
world	politics  0.9999527220578375	politics  0.9999724643134396
science	science  0.9999964686909203	science  0.9999999918479846
politics	politics  0.9999996871837232	politics  1.0
technology	politics  0.9999986143516758	politics  1.0
entertainment	entertainment  0.999999493571153	entertainment  1.0
technology	entertainment  0.9327761985514025	entertainment  0.9895123117488538
technology	technology  0.9329342694264289	technology  0.9953843395175074
sports	sports  0.9999967070975446	sports  0.9999999999450722
sports	sports  0.999999448776502	sports  1.0
sports	sports  0.9999996871837232	sports  1.0
world	world  0.9999945614411088	world  0.9999997532217818
entertainment	entertainment  0.811950215603555	politics  0.7528612583157152
entertainment	entertainment  0.9999998063873693	entertainment  1.0
sports	sports  0.999992415788822	sports  0.9999986290425279
sports	sports  0.9999975415208362	sports  1.0
world	world  0.9914206683883693	world  0.9940887874624009
politics	politics  0.9999995679801056	politics  1.0
sports	sports  0.9998725123368768	sports  0.9998765012299801
technology	science  0.9999838332275324	science  0.9999907439423583
world	politics  0.9999995679801056	politics  0.999999997664406
entertainment	entertainment  0.9875593171165674	entertainment  0.9998766054606149
world	politics  0.9999989719621736	politics  0.9999993524051537
world	politics  0.9947783891664611	politics  0.9947798688712451
world	entertainment  0.9999991359604119	entertainment  1.0
entertainment	entertainment  0.9999996871837232	entertainment  1.0
sports	sports  0.999999091165777	sports  0.9999999999094391
technology	politics  0.9902347658345843	politics  0.9731851780146622
sports	sports  0.9999992103693378	sports  1.0
entertainment	entertainment  0.9999998063873693	entertainment  1.0
world	politics  0.9988296292375166	politics  0.9988304898002195
entertainment	entertainment  0.999999091165777	entertainment  1.0
politics	politics  0.999999448776502	politics  1.0
world	entertainment  0.891524615052265	entertainment  0.48090729839592306
entertainment	entertainment  0.9999996871837232	entertainment  1.0
technology	technology  0.9999831180163555	technology  0.999989783504421
technology	politics  0.9999988527586979	politics  0.999999939764311
entertainment	entertainment  1.0	entertainment  1.0
entertainment	entertainment  0.9999998063873693	entertainment  1.0
world	world  0.9999663107264212	world  0.9999403379398387
sports	sports  0.9999989719621736	sports  0.9999999997827562
science	science  0.9998011283329947	science  0.9999151889523634
technology	technology  0.9999311474442338	technology  0.9999601515218967
world	world  0.9997329780644225	world  0.9998766054164874
world	politics  0.9980725121262062	politics  0.9980732652122012
entertainment	entertainment  0.9999996871837232	entertainment  1.0
politics	politics  0.7753507732561008	politics  0.775379836576707
science	science  0.9913962973279161	science  0.9982931539402691
sports	sports  0.9999992103693378	sports  1.0
world	politics  0.9990875225899702	politics  0.999290321518604
science	science  0.9999907469519438	science  0.9999999990263795
science	science  0.9046363920968283	science  0.9046501883114898
world	world  0.9999908661546805	world  0.9999919982072509
entertainment	entertainment  0.9999992103693378	entertainment  1.0
world	politics  0.999999091165777	politics  0.9999999979388445
technology	technology  0.9999996871837232	technology  0.9999999983947708
world	politics  0.9999998063873693	politics  0.9999999987498469
technology	politics  0.9999984951481292	politics  1.0
sports	sports  0.999999448776502	sports  1.0
science	science  0.9995662063822909	science  0.9996200154981933
entertainment	entertainment  0.9398380752046047	entertainment  0.9046478178980824
sports	entertainment  0.7772949554386956	entertainment  0.77729611718647
entertainment	politics  0.7526778932666842	politics  0.7528612583157152
world	entertainment  0.9999921773839029	entertainment  0.9999930377424044
entertainment	entertainment  0.9731556091469605	entertainment  0.9740343688364768
automobile	technology  0.9999902701411391	technology  0.9999910603030341
sports	sports  0.9999993295729128	sports  0.9999999999821675
sports	entertainment  0.9999880052965596	entertainment  0.9999869928705739
world	science  0.9957130995039434	science  0.9998415638034699
world	technology  0.994691087616606	technology  0.9874121453728198
world	world  0.6789069128799904	world  0.9959289269237142
technology	automobile  0.99997064671561	automobile  0.9999756632157306
technology	technology  0.9929927041196318	technology  0.9957893758470664
world	science  0.9999396103603339	science  0.9999038976015351
science	technology  0.9999988527586979	technology  0.9999995760231606
politics	politics  0.999999091165777	politics  1.0
world	politics  0.9999989719621736	politics  0.9999999976644091
world	politics  0.9999124339371267	politics  0.999933947850445
technology	technology  0.9999988527586979	technology  0.9999999530883864
sports	sports  0.9999989719621736	sports  1.0
entertainment	entertainment  0.9999996871837232	entertainment  1.0
politics	politics  0.9999996871837232	politics  1.0
world	science  0.9999844292354182	science  0.9999966747157166
world	politics  0.9999015874097997	politics  0.9999962733604395
entertainment	entertainment  0.9999998063873693	entertainment  1.0
politics	politics  0.9999996871837232	politics  1.0
world	world  0.7772969708132605	world  0.8519514453906682
entertainment	entertainment  0.9999996871837232	entertainment  1.0
politics	politics  0.999999448776502	politics  1.0
entertainment	politics  0.9999996871837232	politics  0.9999999994788586
science	science  0.9999535564437448	science  0.9999986290426938
technology	politics  0.9999634499385507	politics  0.9999829926909877
science	science  0.999840693185932	science  0.9998893431239464
entertainment	entertainment  0.999999448776502	entertainment  0.9999998555019338
sports	sports  0.9999992103693378	sports  0.9999999998082827
science	science  0.999953794840366	science  0.9999885212337828
entertainment	entertainment  0.9999996871837232	entertainment  1.0
technology	technology  0.999978111559716	technology  0.999981074845851
world	politics  0.9999797803764172	politics  0.9999970976881454
world	world  0.7978329645517502	world  0.8175560771370836
science	science  0.9999914621676675	science  0.9999989322966695
world	entertainment  0.49992108872563473	entertainment  0.5621260429226058
entertainment	entertainment  0.9999998063873693	entertainment  1.0
sports	sports  0.9999992103693378	sports  0.9999999979388448
sports	sports  0.9999992103693378	sports  1.0
automobile	automobile  0.9999991359603977	automobile  1.0
entertainment	entertainment  0.9999996871837232	entertainment  1.0
automobile	technology  0.4402439187517769	technology  0.5732076641698263
entertainment	entertainment  1.0	entertainment  1.0
entertainment	entertainment  0.9999998063873693	entertainment  1.0
world	politics  0.9999995679801056	politics  0.9999999990263794
entertainment	entertainment  0.9999996871837232	entertainment  1.0
sports	sports  0.9999995679801056	sports  1.0
sports	sports  0.9999993295729128	sports  1.0
science	science  0.9988111963838762	science  0.9997694909632698
science	science  0.9999446166242145	science  0.9999832985829125
entertainment	entertainment  0.9999996871837232	entertainment  1.0
automobile	automobile  0.9999993743675585	automobile  1.0
world	world  0.7770725319019323	world  0.7310017059600508
world	world  0.8519510972899247	world  0.7772995191043374
technology	technology  0.9999996871837232	technology  1.0
sports	sports  0.9999984951481292	sports  0.9999999979388448
world	politics  0.9999477157533767	politics  0.9997388097044556
entertainment	entertainment  1.0	entertainment  1.0
technology	politics  0.9999992103693378	politics  1.0
politics	politics  0.9999170824848561	politics  0.9997965408073016
world	politics  0.9999408023273834	politics  0.9999417087301645
automobile	automobile  0.9984944124777736	automobile  0.9933022799549505
politics	politics  0.9999995679801056	politics  1.0
science	science  0.9998683407415155	science  0.9999986263969428
entertainment	entertainment  1.0	entertainment  1.0
sports	sports  0.9999897933314713	sports  0.9999952128388959
world	politics  0.9999995679801056	politics  0.9999999970010383
world	world  0.9979752713271665	world  0.9980728215356096
entertainment	entertainment  0.999999493571153	entertainment  1.0
world	politics  0.999999448776502	politics  1.0
science	science  0.999990508546513	science  0.9999999863199189
sports	sports  0.9999987335551229	sports  0.9999999997210534
politics	politics  0.9999998063873693	politics  1.0
technology	politics  0.9997018820185011	politics  0.9999930377407446
world	world  0.9240721417338125	world  0.9240843421208231
world	politics  0.9999995679801056	politics  1.0
technology	technology  0.9999988527586979	technology  0.9999999976644054
politics	politics  0.9997946926377986	politics  0.9997965611069888
world	technology  0.9999926541947074	technology  0.9999963372726014
politics	politics  0.9999995679801056	politics  0.9999999985833904
technology	automobile  0.7770768279117916	technology  0.7772989722315782
world	politics  0.9999980183344259	politics  0.999999991847972
world	politics  0.999999448776502	politics  0.9999997897566935
science	science  0.9999806147849026	science  0.9999999961492569
sports	sports  0.9999989719621736	sports  1.0
world	world  0.7481055950565669	world  0.7721055088335552
politics	politics  0.9999996871837232	politics  1.0
entertainment	entertainment  0.9999996871837232	entertainment  1.0
sports	entertainment  0.999999493571153	entertainment  1.0
world	world  0.674422877145556	world  0.6745064430523752
world	politics  0.9980698948126878	politics  0.9980732656711332
sports	sports  0.9999984951481292	sports  0.9999999999094393
world	politics  0.9999969455039983	politics  0.9999990577554484
entertainment	entertainment  0.9999998063873693	entertainment  1.0
politics	politics  0.999781586685983	politics  0.996826945956771
technology	politics  0.9999634499385507	politics  0.9999646437382591
world	politics  0.6791631312630775	politics  0.9241386404183937
sports	sports  0.9999989719621736	sports  1.0
politics	politics  0.9999992103693378	politics  0.9999999991407826
politics	politics  0.9999993295729128	politics  0.9999999994788584
world	world  0.999992415788822	world  0.9999933785023049
automobile	automobile  0.9995673235284416	automobile  0.9996199030406355
entertainment	entertainment  0.9999996871837232	entertainment  1.0
sports	sports  0.9999982567411922	sports  0.9999999997210531
sports	entertainment  0.9890113503188723	entertainment  0.9890128548072498
entertainment	entertainment  0.9999975415208362	entertainment  0.9999999586005869
politics	politics  0.9999895549271773	politics  0.9999930377440652
science	science  0.99999158136958	science  0.999999994031305
entertainment	entertainment  1.0	entertainment  1.0
world	politics  0.9997895679512191	politics  0.9998115356017395
sports	sports  0.9999982567411922	sports  1.0
technology	entertainment  0.9999996871837232	entertainment  1.0
entertainment	entertainment  0.9999996871837232	entertainment  1.0
technology	technology  0.9940790812090692	technology  0.9940875764229145
technology	technology  0.999999448776502	technology  0.9999999714916952
sports	sports  0.9999984951481292	sports  1.0
world	politics  0.9999436630504616	politics  0.999948557782567
sports	sports  0.9999986143516758	sports  0.9999999998973812
entertainment	entertainment  0.9999993743675585	entertainment  1.0
entertainment	entertainment  0.999999612774776	entertainment  1.0
entertainment	entertainment  0.9999998063873693	entertainment  1.0
sports	politics  0.9230784730299122	politics  0.9230926124726059
sports	sports  0.999999448776502	sports  1.0
world	world  0.9999808531879747	world  0.9999842984942192
sports	entertainment  0.7772933802070418	entertainment  0.9988303278749469
entertainment	entertainment  0.9999996871837232	entertainment  1.0
politics	politics  0.999999448776502	politics  1.0
entertainment	entertainment  0.999999612774776	entertainment  1.0
entertainment	entertainment  0.9999993295729128	entertainment  0.9999999530883417
politics	politics  0.9999992103693378	politics  1.0
sports	sports  0.999999448776502	sports  1.0
politics	politics  0.9999982567411922	politics  0.9999966978099873
politics	politics  0.9999995679801056	politics  1.0
world	politics  0.9999995679801056	politics  1.0
science	technology  0.9933043030751828	technology  0.9933071487104175
politics	politics  0.9999995679801056	politics  1.0
world	politics  0.9999992103693378	politics  0.9999999979388448
sports	sports  0.9981381860991223	sports  0.9977068093295676
science	science  0.9999735819307618	science  0.9999999217322216
science	science  0.9999803763818875	science  0.999999992805863
technology	politics  0.9999932502087602	politics  0.9999977989092962
world	science  0.9994652219169623	science  0.9995121015621561
sports	sports  0.9999995679801056	sports  1.0
politics	politics  0.9999996871837232	politics  1.0
entertainment	entertainment  0.9999874092856329	entertainment  0.9999885077895663
technology	world  0.963540687534505	world  0.931434721019891
science	science  0.9999340081433281	science  0.9998676679737588
world	politics  0.9770039525609374	politics  0.977019074204214
sports	sports  0.9999978991308366	sports  0.999999999641825
science	science  0.9999806147849026	science  0.9999999881388763
sports	sports  0.9999988527586979	sports  0.9999999632190375
technology	politics  0.7772803383381834	politics  0.9840921866164437
politics	politics  0.9999931310057394	politics  0.999999554914862
politics	politics  0.9999993295729128	politics  1.0
world	politics  0.9999988527586979	politics  0.9999998144608402
politics	politics  0.9999996871837232	politics  1.0
sports	sports  0.999989435724597	sports  0.9999977210562134
technology	politics  0.9975486406373586	politics  0.9975731838677507
politics	politics  0.9999989719621736	politics  0.9999999970010397
world	politics  0.9999993295729128	politics  0.9999999397642176
sports	sports  0.9984972207165376	sports  0.9968273118250306
entertainment	entertainment  0.999999448776502	entertainment  1.0
technology	technology  0.9999992103693378	technology  0.9999999881388706
sports	sports  0.9999986143516758	sports  0.9999999748900221
science	science  0.9977941319192581	science  0.9959298367193495
politics	politics  0.9999995679801056	politics  1.0
politics	politics  0.9999987335551229	politics  0.9999999907625574
science	science  0.9888852005440555	science  0.939576908377743
entertainment	entertainment  0.9999996871837232	entertainment  1.0
science	science  0.9999400871469832	science  0.9999251537851237
world	world  0.984606217950396	world  0.9919505927235672
world	technology  0.9797285340611587	technology  0.9969639688076675
entertainment	entertainment  0.998040459280678	entertainment  0.9966493860391
world	politics  0.999978945968628	politics  0.9999832320658417
entertainment	entertainment  0.9999996871837232	entertainment  1.0
science	science  0.9999404447353004	science  0.9999885212269459
world	politics  0.9999988527586979	politics  0.999999607213911
entertainment	entertainment  0.9999996871837232	entertainment  1.0
world	world  0.9999931310057394	world  0.999994822122063
world	politics  0.9999998063873693	politics  0.9999999988967437
world	world  0.9999976607241555	world  0.9999989000548574
world	science  0.9999440206400576	science  0.9999983339675129
world	politics  0.817572920206005	politics  0.8807970415836783
sports	sports  0.9999961110814314	sports  0.9999991355796634
entertainment	entertainment  0.9999996871837232	entertainment  1.0
world	politics  0.9999996871837232	politics  0.9999999995400943
politics	politics  0.9999958726751765	politics  0.9999546021442743
science	science  0.9999138642588821	science  0.9999998555019357
sports	sports  0.999996587894339	sports  0.9999999998308103
world	politics  0.9999957534718431	politics  0.9999989322974326
politics	politics  0.9999992103693378	politics  0.9999999983947707
technology	technology  0.9999982567411922	technology  0.999999992805869
world	politics  0.9999930118027327	politics  0.9999921107349491
world	world  0.9998549917882363	world  0.9998592359347562
science	science  0.9999754891398779	science  0.9999999881388834
entertainment	entertainment  0.9999995679801056	entertainment  1.0
technology	politics  0.9972133977758844	politics  0.998100131312596
automobile	technology  0.9999993295729128	technology  0.9999999317439252
world	world  0.999999448776502	world  0.9999999123574772
entertainment	entertainment  0.9999980183344259	entertainment  0.9999992466234013
technology	technology  0.999999448776502	technology  0.9999999873642758
technology	technology  0.5024657640002651	technology  0.8352822370376431
world	politics  0.9999983759447105	politics  0.9999957771662421
technology	politics  0.7310477502766825	politics  0.7310575604247395
world	world  0.9914205538951522	world  0.9914222067568571
science	science  0.9947560830494715	science  0.9933071162702313
entertainment	entertainment  0.9999120763551886	entertainment  0.9998911030564861
politics	politics  0.999999448776502	politics  1.0
science	science  0.9999943230347685	science  0.9999999989495642
politics	entertainment  0.9982769696030038	entertainment  0.9982804166209275
world	politics  0.9999989719621736	politics  0.9999998362623601
science	technology  0.9524814197727844	technology  0.952520260925225
science	science  0.9999806147849026	science  0.9999999918479768
sports	sports  0.999999448776502	sports  0.9999999999622486
politics	politics  0.9999998063873693	politics  1.0
technology	politics  0.9999580859819887	politics  0.9999370028825838
entertainment	entertainment  0.9999996871837232	entertainment  1.0
entertainment	entertainment  0.9999996871837232	entertainment  1.0
technology	technology  0.9893665397928633	technology  0.9984985800159609
technology	technology  0.9999996871837232	technology  1.0
automobile	automobile  0.9999981823323942	automobile  0.999999958600651
politics	politics  0.9999993295729128	politics  1.0
entertainment	entertainment  0.9999996871837232	entertainment  1.0
politics	politics  0.9999995679801056	politics  0.9999999994094694
politics	politics  0.9999996871837232	politics  0.9999999997210531
politics	world  0.9986666033530434	world  0.9953716770855341
politics	politics  0.9999993295729128	politics  0.99999999660173
automobile	automobile  0.9999992551639781	automobile  1.0
entertainment	entertainment  0.999999612774776	entertainment  1.0
technology	science  0.9996452934000105	science  0.9984988070361024
technology	politics  0.999992415788822	politics  0.9999998555019339
entertainment	entertainment  0.9999998063873693	entertainment  1.0
sports	sports  0.999999091165777	sports  0.9999999999294712
entertainment	entertainment  0.9999998063873693	entertainment  1.0
world	world  0.9909674139493195	world  0.9910470612901257
politics	politics  0.9999989719621736	politics  1.0
entertainment	entertainment  0.9999998063873693	entertainment  1.0
technology	technology  0.9999983759447105	technology  0.9999999746984898
entertainment	entertainment  0.9999996871837232	entertainment  1.0
politics	politics  0.9999995679801056	politics  1.0
sports	sports  0.9999989719621736	sports  0.9999999999515263
world	politics  0.9999995679801056	politics  1.0
automobile	automobile  0.9999575347832431	automobile  0.999968425504057
automobile	automobile  0.9999991359603977	automobile  0.9999999906920811
world	world  0.9999998063873693	world  1.0
technology	politics  0.9999983759447105	politics  0.9999986290423645
sports	sports  0.9999992103693378	sports  0.9999999999294711
politics	politics  0.9999944422379315	politics  0.99999743871821
technology	politics  0.9240931064128612	politics  0.9046434048470297
technology	politics  0.9999975415208362	politics  0.9999997015889837
sports	sports  0.999999091165777	sports  1.0
world	politics  0.9999998063873693	politics  0.9999999988967437
science	world  0.9044950504681863	world  0.9045354955132482
entertainment	entertainment  1.0	entertainment  1.0
politics	politics  0.9999996871837232	politics  1.0
entertainment	entertainment  0.9999996871837232	entertainment  1.0
world	politics  0.9999982567411922	politics  0.9999999943972052
sports	sports  0.999998137537802	sports  0.9999999916440622
technology	technology  0.9999992103693378	technology  0.9999999990263791
sports	sports  0.9975161363950352	sports  0.997527371256569
sports	sports  0.9999988527586979	sports  1.0
world	politics  0.8732839135771824	politics  0.873993025376078
technology	technology  0.9999996871837232	technology  1.0
sports	sports  0.999999448776502	sports  1.0
technology	technology  0.9999849060434378	technology  0.999999529111165
sports	sports  0.999999448776502	sports  1.0
sports	sports  0.9999992103693378	sports  0.9999999998308101
politics	politics  0.9998491516735807	politics  0.9998601675014803
world	politics  0.9999974223173038	politics  0.9999998440276427
sports	sports  0.9999992103693378	sports  1.0
entertainment	entertainment  0.9999998063873693	entertainment  1.0
world	politics  0.9994441417562824	politics  0.9998766054017579
entertainment	politics  0.5621706051333789	politics  0.5621759580543182
sports	sports  0.9999984951481292	sports  1.0
entertainment	entertainment  0.9999998063873693	entertainment  1.0
automobile	automobile  0.9999247557040788	automobile  0.9999269349098178
world	world  0.9602845734551312	world  0.9625836404219579
world	politics  0.999999448776502	politics  1.0
world	politics  0.8338261058855976	politics  0.8278109882813123
entertainment	entertainment  0.9999996871837232	entertainment  1.0
sports	sports  0.9999971839107363	sports  0.999999999409469
automobile	automobile  0.9986719977057643	automobile  0.9986747059910934
world	politics  0.9968260263256912	politics  0.9975273764706583
world	science  0.9995881232385259	science  0.9997377425424463
entertainment	entertainment  0.9999993295729128	entertainment  1.0
technology	technology  0.9984963875058751	technology  0.9975273244666542
sports	politics  0.8776269860058222	politics  0.8776368378582262
technology	politics  0.9999956342685238	politics  0.9999997879426384
world	politics  0.9999993295729128	politics  0.9999999988967432
automobile	automobile  0.9999983015357756	automobile  0.9999995549148089
politics	politics  0.9999971839107363	politics  0.9999999865596021
sports	sports  0.9999996871837232	sports  0.9999999999666845
world	politics  0.9998739425942514	politics  0.9998911030824448
entertainment	entertainment  1.0	entertainment  1.0
world	politics  0.999997064707474	politics  0.999999887464937
world	science  0.9996985453045141	science  0.9998908620546263
world	politics  0.9994454522140778	politics  0.9994472214174559
world	politics  0.999989435724597	politics  0.9999980052701303
technology	technology  0.9999779923584998	technology  0.999982592260889
technology	politics  0.9999989719621736	politics  0.9999999907625431
technology	politics  0.9999759659436347	politics  0.9999785550474514
sports	entertainment  0.9999539140405167	entertainment  0.9999544165948815
sports	sports  0.9999983759447105	sports  1.0
entertainment	entertainment  0.9999998063873693	entertainment  1.0
science	science  0.9995996776545112	science  0.9996200105980053
entertainment	entertainment  0.9999992103693378	entertainment  0.9999987901342265
politics	politics  0.9999993295729128	politics  1.0
world	science  0.5618401795710882	world  0.8175590147740845
automobile	automobile  0.9999984207393133	automobile  0.99999926617879
world	world  0.9958032869027551	world  0.9958408236572717
science	world  0.8644410444510484	world  0.8833529291357423
technology	politics  0.999998137537802	politics  1.0
world	science  0.9768497939190265	science  0.9770216131888372
world	world  0.99999408462894	world  0.999995777165739
politics	politics  0.999999448776502	politics  1.0
"""


def compute_ece(confidences, correctness, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).

    Parameters:
    - confidences: np.array of confidence scores (floats between 0 and 1)
    - correctness: np.array of correctness indicators (1 if correct, 0 if incorrect)
    - n_bins: number of bins to partition confidence scores into

    Returns:
    - ece: float, the expected calibration error
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        # Find indices of samples whose confidence falls into current bin (exclusive lower, inclusive upper)
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)  # fraction of samples in this bin

        if prop_in_bin > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(correctness[in_bin])
            ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin

    return ece


for line in data.strip().split("\n"):
    parts = line.strip().split("\t")
    if len(parts) != 3:
        print(f"Skipping malformed line: {line}")
        continue

    true_label = parts[0].strip()

    m1_pred_conf = parts[1].strip().split()
    m1_pred = m1_pred_conf[0]
    # Handle multi-word labels in prediction (like "South Africa") by joining all except last element as label
    if len(m1_pred_conf) > 2:
        m1_pred = " ".join(m1_pred_conf[:-1])
    m1_conf = float(m1_pred_conf[-1])
    m1_corr = 1 if m1_pred == true_label else 0

    m2_pred_conf = parts[2].strip().split()
    m2_pred = m2_pred_conf[0]
    if len(m2_pred_conf) > 2:
        m2_pred = " ".join(m2_pred_conf[:-1])
    m2_conf = float(m2_pred_conf[-1])
    m2_corr = 1 if m2_pred == true_label else 0

    model1_confidences.append(m1_conf)
    model1_correct.append(m1_corr)
    model2_confidences.append(m2_conf)
    model2_correct.append(m2_corr)


print(model1_confidences, model1_correct)
print(model2_confidences, model2_correct)

model1_confidences = np.array(model1_confidences)
model1_correct = np.array(model1_correct)
model2_confidences = np.array(model2_confidences)
model2_correct = np.array(model2_correct)

mask = model1_confidences >= 0.9
mask1 = model2_confidences >= 0.9

ece_model1 = compute_ece(np.array(model1_confidences[mask]),
                         np.array(model1_correct[mask]))
ece_model2 = compute_ece(np.array(model2_confidences[mask1]),
                         np.array(model2_correct[mask1]))

print(f"ECE for Model 1: {ece_model1:.4f}")
print(f"ECE for Model 2: {ece_model2:.4f}")
