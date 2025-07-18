
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
            model=self.model, messages=prompt, logprobs=True, seed=0, temperature=0, max_tokens=1002, top_logprobs=10)
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
            model=self.model, messages=prompt, logprobs=True, seed=0, temperature=0, max_tokens=1002, top_logprobs=10)
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
#         I will give you a text. There will be multiple world countries mentioned in the text. Your task is to extract the name of the MOST MENTIONED world country in the text.
#         You MUST respond with ONLY the name of the MOST mentioned country. THERE IS ALWAYS A MOST MENTIONED country.
#         here is the list of potential countries:[
#     "United States", "Canada", "Mexico", "Brazil", "Argentina",
#     "United Kingdom", "France", "Germany", "Italy", "Spain",
#     "Portugal", "Netherlands", "Belgium", "Sweden", "Norway",
#     "Russia", "Poland", "Ukraine", "Switzerland", "Greece",
#     "India", "China", "Japan", "South Korea", "Indonesia",
#     "Thailand", "Vietnam", "Philippines", "Pakistan", "Bangladesh",
#     "Australia", "New Zealand", "South Africa", "Nigeria", "Egypt",
#     "Kenya", "Ethiopia", "Turkey", "Saudi Arabia", "Iran"
# ]

#         Here is the text: {}


#         '''


task = '''
I will give you a Supreme Court opinion.
Your task is to determine if this opinion reverses a lower court's ruling.
Note that the opinion may not be an appeal, but rather a new ruling.

- True if the Supreme Court reverses the lower court ruling
- False otherwise

Here is the opinion: {}

You must respond with ONLY True or False:
'''


# Define oracle and proxy
proxy = OpenAIProxy(task, model='gpt-4o-mini')
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


df = pd.read_csv("BARGAIN/examples/newtests/court_opinion.csv")

print(df)

# for index, row in df.iterrows():
#     # Get the article text from the current row
#     current_article = row['news_headline'] + " " + row['news_article']
#     # print(row['news_category'])
#     # injected_animal = row['injected_animals']
#     # arr = injected_animal.strip("[]").split(",")
#     # cleaned_arr = [item.strip(" '") for item in arr]
#     # print_most_frequent_element(cleaned_arr)

#     # Call the processing function with the current article.
#     # Note: We pass a single article string, not a numpy array.
#     # print(injected_animal)

#     # proxy.class_proxy_func(current_article, [
#     #     "ARTS & CULTURE",
#     #     "BUSINESS",
#     #     "COMEDY",
#     #     "CRIME",
#     #     "EDUCATION",
#     #     "ENTERTAINMENT",
#     #     "ENVIRONMENT",
#     #     "MEDIA",
#     #     "POLITICS",
#     #     "RELIGION",
#     #     "SCIENCE",
#     #     "SPORTS",
#     #     "TECH",
#     #     "WOMEN"
#     # ])

#     # proxy.class_proxy_func(current_article, [
#     #     "technology", "sports", "politics", "entertainment", "world", "automobile", "science", "business"
#     # ])
#     # proxy.proxy_func_general(current_article)
#     # time.sleep(0.5)
#     # print(f"{res[0]}, {res[1]}", file=f)


# model1_confidences = []
# model1_correct = []
# model2_confidences = []
# model2_correct = []

# data = """
# technology	business    0.5446693799825417	business    0.5765920746400071
# technology	automobile    0.9999911493539052	automobile    0.9999998874647357
# entertainment	entertainment    0.999796122783857	entertainment    1.0
# world	science    0.9699987668851393	science    0.9890087012784027
# science	science    0.9999888397128177	science    0.9999999998640658
# science	science    0.9622385837980258	science    0.9947795462171379
# world	science    0.9973203401951286	science    0.9999435572458588
# world	health    0.37094581496272444	health 0.0
# science	science    0.9999676219234456	science    0.9999998578045122
# politics	politics    0.9975225428637153	politics    1.0
# politics	politics    0.9975209967512624	politics    1.0
# sports	sports    0.9999536756438672	sports    0.9999999999377586
# science	technology    0.9998845431118482	technology    0.9999915343020258
# science	science    0.9872198364347871	science    0.9916333291363241
# entertainment	entertainment    0.999924830105154	entertainment    1.0
# technology	automobile    0.9946989142672498	automobile    0.9968234052615741
# politics	politics    0.9990840798466758	politics    1.0
# automobile	automobile    0.9999437078393169	automobile    0.9999993524053098
# science	science    0.9718378854683991	science    0.9947788181762558
# entertainment	entertainment    0.9999491461219632	entertainment    0.9999957514706149
# sports	sports    0.9999888397128177	sports    0.9999999999861121
# world	politics    0.917034440003687	politics    0.9464530079314734
# world	world    0.5784226305117621	world    0.764863960240922
# world	world    0.999841169932691	world    0.9999918839094775
# sports	sports    0.9997346464629393	sports    0.9999997579122563
# science	science    0.9718378854683991	science    0.9947788181762558
# politics	politics    0.9980450881895108	politics    0.999995522129431
# technology	politics    0.9992639012441108	politics    0.9999927501601974
# world	politics    0.998560920962901	politics    0.999858840938727
# politics	politics    0.9959129967574379	politics    1.0
# sports	sports    0.9999410407209638	sports    1.0
# entertainment	entertainment    0.9997383408025537	entertainment    1.0
# entertainment	entertainment    0.9997960036134275	entertainment    1.0
# world	politics    0.9984906852390655	politics    0.9999998362623447
# automobile	automobile    0.9994166807742675	automobile    0.9648518522139028
# world	politics    0.8729976366957376	politics    0.9889440195590139
# automobile	politics    0.9981260758405742	politics    0.9994642929858181
# automobile	automobile    0.9994323992590558	automobile    0.9997318510349229
# automobile	automobile    0.9999217758404069	automobile    0.9999999895325951
# sports	sports    0.9999827604127778	sports    0.9999999999821675
# sports	sports    0.9999768003489373	sports    0.9999999999622486
# technology	politics    0.9980603952283409	politics    0.9999952149036351
# technology	technology    0.9999027793278867	technology    0.99999999902638
# technology	technology    0.999691398943974	technology    0.9998155901543461
# sports	sports    0.9999866940728078	sports    1.0
# entertainment	entertainment    0.9999241597350703	entertainment    1.0
# world	politics    0.9979384667693594	politics    0.9988300457263218
# automobile	automobile    0.9999855468464867	automobile    0.9999999316121916
# entertainment	entertainment    0.9999249493054897	entertainment    1.0
# technology	technology    0.9999816875973553	technology    0.9999977396752664
# world	science    0.985156973421456	science    0.9994948558327851
# technology	business    0.957184547409924	business    0.9577654895623304
# entertainment	entertainment    0.9999036136757582	entertainment    1.0
# entertainment	entertainment    0.9998533231743326	entertainment    0.999995699823762
# politics	politics    0.9984957923558275	politics    1.0
# world	world    0.8174193949720834	politics    0.62240496981547
# world	politics    0.9988163050237228	politics    0.9999967112511237
# sports	sports    0.9999829988145424	sports    1.0
# world	politics    0.9990612285898378	politics    0.9999999950555453
# sports	sports    0.9932656074513888	sports    0.9995692792394817
# world	world    0.9982550660526994	world    0.9989774842256525
# automobile	automobile    0.9999508596836361	automobile    1.0
# sports	sports    0.9999639267365671	sports    0.9999999999675372
# technology	politics    0.9980055484257857	politics    0.9999191615116363
# politics	politics    0.9980652695068855	politics    0.99999999700104
# entertainment	entertainment    0.9999032560969745	entertainment    1.0
# technology	technology    0.9984142730541677	technology    0.9957674669958074
# technology	technology    0.9990156687750523	technology    0.9859334338362197
# world	world    0.989630670427599	world    0.9957691801931982
# world	entertainment    0.9999399679484807	entertainment    0.9999995515198705
# sports	entertainment    0.9999355576902692	entertainment    0.9994238449301469
# politics	politics    0.9871141207341216	politics    0.9959203633414181
# science	science    0.9999026601374692	science    0.9999999224385198
# science	science    0.9991420291940069	science    0.9980728524516409
# sports	sports    0.9999640459342878	sports    1.0
# automobile	automobile    0.999384775332067	automobile    0.9502456955431127
# entertainment	entertainment    0.9999782307609466	entertainment    1.0
# politics	politics    0.9998703669473757	politics    0.9999999458942399
# world	politics    0.9994116360765555	politics    0.9997386459848627
# automobile	technology    0.8515642523907887	technology    0.9240770337427974
# politics	politics    0.9947629317718498	politics    0.9999999677581369
# technology	technology    0.9769020047511321	technology    0.8519235962608017
# technology	politics    0.999431997095076	politics    0.9999991002157234
# sports	sports    0.9999517684727173	sports    0.9999999994374754
# world	politics    0.9995556056032558	politics    0.9999994956525253
# science	science    0.8514597497600772	science    0.9706662000628616
# world	politics    0.9990519504353282	politics    0.9999938558266853
# politics	politics    0.9994433114204889	politics    1.0
# sports	sports    0.9999969455039983	sports    0.9999999999969011
# sports	sports    0.9999864556683433	sports    1.0
# sports	sports    0.9999231613819755	sports    0.9999999999622486
# world	world    0.7761040425752184	world    0.954106913004162
# politics	politics    0.9988242749494053	politics    1.0
# world	politics    0.9974903957375728	politics    0.9999912945761689
# world	crime    0.7496150797240516	crime 0.0
# science	science    0.9999815683975317	science    0.9999998723491522
# technology	entertainment    0.9571297213883775	entertainment    0.9794517872331445
# sports	sports    0.99999503825305	sports    0.9999999999964886
# world	world    0.5086186574817935	world    0.548173138607509
# world	politics    0.9994069898349828	politics    0.9999334440225286
# world	politics    0.9947457069890675	politics    0.9959288030585767
# entertainment	entertainment    0.9999025409470658	entertainment    0.999999975548035
# world	science    0.9989618945090505	science    0.9998150517269109
# sports	sports    0.9999864556683433	sports    1.0
# world	politics    0.9987843144503546	politics    0.9999980052701831
# sports	sports    0.9999967070975446	sports    0.9999999999891841
# science	science    0.9957178219436122	science    0.9988272891622259
# sports	sports    0.9999862172648452	sports    0.9999999999797932
# world	technology    0.9955399783190387	technology    0.9979629360564487
# world	politics    0.9998387862448186	politics    0.9999996072137779
# world	politics    0.9948852712839863	politics    0.999446280101592
# technology	technology    0.9999609467545135	technology    0.9999983932421829
# technology	technology    0.9992403311617253	technology    0.9998415016936489
# world	world    0.9999229229962954	world    0.9999997084634831
# sports	sports    0.9999770387510999	sports    0.9999999998755172
# entertainment	entertainment    0.9998760879914247	entertainment    1.0
# automobile	automobile    0.9999719579181635	automobile    0.9999956235466286
# world	science    0.9946563031841783	science    0.9999936355448523
# world	politics    0.9967690522401179	politics    0.9999785266054216
# politics	politics    0.9984921062693635	politics    0.999999582103739
# automobile	automobile    0.9999216566477248	automobile    0.9999999976644048
# world	politics    0.9622604394625479	politics    0.9959273220526297
# automobile	automobile    0.9999004402800508	automobile    0.9999999907625486
# entertainment	entertainment    0.9997958844284633	entertainment    1.0
# entertainment	entertainment    0.999924591719076	entertainment    0.9999999936512055
# entertainment	entertainment    0.9997962419688495	entertainment    1.0
# entertainment	entertainment    0.999924830105154	entertainment    1.0
# technology	politics    0.9976363079662418	politics    0.9990882746826133
# world	politics    0.9941084195808731	politics    0.9997338268381648
# sports	sports    0.9999829988145424	sports    0.9999999999675576
# world	entertainment    0.9997884209499837	entertainment    0.9999829519623579
# entertainment	entertainment    0.9999415175046573	entertainment    1.0
# technology	technology    0.9909120464141931	technology    0.9882292391068713
# sports	sports    0.9999779923584998	sports    0.9999999999960211
# politics	politics    0.9998366409276824	politics    0.99999999940947
# sports	entertainment    0.9999860236548829	entertainment    0.9999977362783633
# technology	politics    0.9980564765120217	politics    1.0
# sports	sports    0.9999822836057812	sports    0.9999999999877439
# world	science    0.932138522184076	science    0.939894406213013
# entertainment	entertainment    0.9999249493054897	entertainment    1.0
# politics	entertainment    0.9522405191749013	entertainment    0.9769817836593281
# technology	technology    0.9999766811478773	technology    0.9999980420022012
# sports	sports    0.9999412791109633	sports    1.0
# technology	politics    0.9994140150811371	politics    0.9999775153709402
# entertainment	entertainment    0.999924591719076	entertainment    1.0
# entertainment	entertainment    0.9996152054792449	entertainment    0.9999991356771034
# entertainment	entertainment    0.9999034200817891	entertainment    1.0
# sports	sports    0.9999639267365671	sports    0.9999999999877439
# world	technology    0.9884533575707446	technology    0.994640010272878
# technology	technology    0.9999934886143899	technology    0.9999999980622487
# sports	sports    0.999989435724597	sports    1.0
# world	science    0.5683459877428401	science    0.7771061986445511
# entertainment	entertainment    0.9999782307609466	entertainment    1.0
# sports	sports    0.9999411599159564	sports    0.9999999999969011
# sports	sports    0.9999895549271773	sports    0.9999999918479731
# world	politics    0.9761382006456502	politics    0.9770189344294112
# technology	technology    0.9997336930712344	technology    0.9999767421893001
# world	science    0.891764868947036	world    0.8956236555208155
# politics	politics    0.9989777176600138	politics    0.9999724421523871
# technology	politics    0.9925589181233712	politics    0.9932977463978742
# sports	sports    0.9999863364670418	sports    1.0
# entertainment	entertainment    0.9998763263659391	entertainment    1.0
# world	entertainment    0.8806166112244113	entertainment    0.9769880230843593
# automobile	automobile    0.9928894366613985	automobile    0.9996626267017197
# world	politics    0.9247888112899744	politics    0.9044883109198173
# technology	technology    0.9999777539561099	technology    0.9999999123575818
# world	world    0.9999335313595774	world    0.9999977139780268
# entertainment	entertainment    0.9997381024463875	entertainment    1.0
# politics	politics    0.9994409286811378	politics    0.99999999614926
# science	science    0.9999845484374018	science    0.9999989322966697
# technology	technology    0.9998373560231834	technology    0.9999962428466219
# entertainment	entertainment    0.9997699073482937	entertainment    0.9998189015855804
# entertainment	entertainment    0.9999415175046573	entertainment    1.0
# entertainment	entertainment    0.999796122783857	entertainment    1.0
# entertainment	entertainment    0.9999238765539077	entertainment    0.9999999928058683
# entertainment	entertainment    0.9998410507423425	entertainment    1.0
# entertainment	entertainment    0.9999639267365671	entertainment    1.0
# entertainment	entertainment    0.9999644035311729	entertainment    1.0
# world	politics    0.9984789166286545	politics    0.9990887852286364
# world	politics    0.998047226046219	politics    0.9999930055047752
# politics	politics    0.9914121137501333	politics    1.0
# technology	politics    0.9994970145015075	politics    0.9998414362378092
# technology	politics    0.9997824209254534	politics    0.9999916895031639
# entertainment	entertainment    0.999840931566558	entertainment    1.0
# technology	politics    0.9959166623084282	politics    0.9999999827421598
# entertainment	entertainment    0.9998760879914247	entertainment    1.0
# world	politics    0.996816909807551	politics    0.9999998555019914
# world	politics    0.9982944929770872	politics    0.9998172051590396
# technology	politics    0.9869273837191236	politics    0.9671505140446024
# entertainment	entertainment    0.9999247109121079	entertainment    1.0
# entertainment	entertainment    0.9999036136757582	entertainment    1.0
# entertainment	entertainment    0.9992890129967623	entertainment    1.0
# technology	technology    0.7306861115099752	technology    0.777053565419433
# world	politics    0.9992644967938221	politics    0.9999967112523825
# world	world    0.4729763944035203	world    0.6636089635319539
# politics	politics    0.99979266658529	politics    1.0
# technology	politics    0.9926408947889784	politics    0.9932977463978742
# automobile	technology    0.9234673589674575	technology    0.7309484891615944
# world	politics    0.6219300984968487	politics    0.8175398239956054
# entertainment	entertainment    0.9997364340421032	entertainment    0.9999974102653288
# sports	sports    0.9999714363225806	sports    0.9999999999294712
# sports	sports    0.9999539140405167	sports    0.9999999994788586
# sports	sports    0.999978111559716	sports    1.0
# technology	technology    0.99965458436741	technology    0.9999999363740555
# entertainment	entertainment    0.9996641174254821	entertainment    1.0
# world	politics    0.8775886039095646	politics    0.5621418785472655
# entertainment	entertainment    0.9998763263659391	entertainment    1.0
# entertainment	entertainment    0.9998759688114638	entertainment    0.9999999317439667
# politics	politics    0.999285439499818	politics    1.0
# science	science    0.9999230421891283	science    0.9999251385758876
# technology	technology    0.9998970581311167	technology    0.9999992565945668
# world	politics    0.999082650704612	politics    0.9999999979388456
# sports	sports    0.9999815683975317	sports    0.9999999999877439
# sports	sports    0.9999779923584998	sports    1.0
# technology	automobile    0.9398341890264488	automobile    0.9398933038086353
# politics	politics    0.9996628066227882	politics    1.0
# world	politics    0.9979103327544684	politics    0.9988300457263218
# sports	sports    0.9999539140405167	sports    1.0
# technology	technology    0.9994743884280238	technology    0.9993971059332709
# politics	politics    0.9980683518023405	politics    1.0
# sports	sports    0.9999410407209638	sports    0.9999999999797932
# entertainment	entertainment    0.9999543908303483	entertainment    1.0
# entertainment	entertainment    0.9999828796127436	entertainment    0.9999999966017323
# science	science    0.9996662587211285	science    0.9999983588600252
# technology	entertainment    0.9999722707259225	entertainment    0.9999810748503706
# sports	sports    0.9999541524335852	sports    1.0
# entertainment	entertainment    0.9997383408025537	entertainment    1.0
# world	politics    0.999231643326107	politics    0.9999962733599699
# entertainment	entertainment    0.9999773963544504	entertainment    0.9995622892771698
# technology	politics    0.9990671795802732	politics    0.9999999123574762
# sports	entertainment    0.9999541524335852	entertainment    1.0
# technology	politics    0.9980367784769294	politics    0.9999994956527792
# technology	politics    0.6781517115081624	technology    0.8806971391320896
# technology	technology    0.9999237573536999	technology    0.9999999865595909
# technology	technology    0.9999524836614722	technology    0.9999999898069006
# politics	politics    0.9998392629761186	politics    1.0
# world	politics    0.9947528998108603	politics    0.9999952149055207
# world	politics    0.9995644191075025	politics    0.9999999123575432
# technology	technology    0.9976387980142051	technology    0.9984141332364919
# automobile	automobile    0.9994327566783742	automobile    0.9997876368631223
# world	world    0.9986152468933159	world    0.9947609099548987
# world	politics    0.9958933669968479	politics    0.9998760340784939
# entertainment	entertainment    0.999941398309622	entertainment    1.0
# technology	politics    0.9990757468868416	politics    0.9999999960591808
# technology	technology    0.9999411599159564	technology    0.999999999808283
# politics	politics    0.9994368780140078	politics    1.0
# world	politics    0.9998377135784003	politics    0.9999999450231792
# sports	sports    0.9999828796127436	sports    1.0
# science	science    0.999981926000683	science    0.9999999998837176
# world	politics    0.9988135702964362	politics    0.9999998874647521
# world	politics    0.9990839607465796	politics    0.9999999990263791
# entertainment	entertainment    0.9999644035311729	entertainment    1.0
# politics	politics    0.9990859744007998	politics    1.0
# entertainment	entertainment    0.9999033752947384	entertainment    1.0
# science	science    0.9999534372472745	science    0.9999999993151873
# entertainment	entertainment    0.999840931566558	entertainment    1.0
# sports	sports    0.9999406831287956	sports    0.9999999991407827
# technology	politics    0.9997241630157417	politics    0.9984941515831626
# automobile	automobile    0.9995265890044701	automobile    0.9996286652027954
# science	science    0.9888356553814257	science    0.9947795443600178
# science	science    0.9999083813735681	science    0.9999832851370196
# world	world    0.5581748393214957	world    0.6181389632329919
# world	politics    0.998037016659842	politics    0.999993855826685
# world	world    0.9984725005306537	world    0.9959284783138069
# world	world    0.8172685655999287	world    0.8173697066843334
# entertainment	politics    0.9842479836944282	politics    0.985731050913994
# politics	politics    0.995916781494713	politics    1.0
# automobile	business    0.8225687003745066	business    0.8253211798895957
# sports	sports    0.9999864556683433	sports    0.999999999990455
# world	world    0.5383615246615094	politics    0.6037084944742582
# sports	sports    0.9999638075352227	sports    0.9999999999797932
# world	politics    0.9998366409276824	politics    0.9999999945332788
# entertainment	entertainment    0.9998410507423425	entertainment    1.0
# politics	politics    0.9958841435359514	politics    0.9999951735078932
# science	world    0.9625429839321976	world    0.9626436918687725
# technology	politics    0.9974951410058923	politics    0.9999869928736413
# automobile	automobile    0.9996557016358029	automobile    0.9999951466483603
# automobile	technology    0.9996405269639129	technology    0.9997385246807209
# technology	politics    0.9994254483437707	politics    0.9999999439095192
# politics	politics    0.9998735850260785	politics    0.9999999979388453
# entertainment	entertainment    0.9999249493054897	entertainment    1.0
# politics	politics    0.9975236091415532	politics    1.0
# technology	politics    0.9806097116118522	politics    0.981925050776227
# entertainment	entertainment    0.9998763263659391	entertainment    1.0
# world	entertainment    0.7973136002882336	entertainment    0.869697977624345
# politics	politics    0.9994441417562824	politics    1.0
# world	politics    0.9990825316046861	politics    0.9999999586006106
# world	politics    0.999430210089348	politics    0.9999999468421993
# automobile	technology    0.9240168647340837	technology    0.9241321078552938
# politics	politics    0.9980669241713224	politics    1.0
# sports	sports    0.9999714363225806	sports    1.0
# science	science    0.9999859788604943	science    0.9999999990801882
# entertainment	entertainment    0.9999415175046573	entertainment    1.0
# world	politics    0.9988110773163101	politics    0.9999974387167695
# science	science    0.9999860980626626	science    0.9999999748277688
# entertainment	entertainment    0.999841169932691	entertainment    1.0
# entertainment	entertainment    0.9942323652171535	entertainment    0.9968054420358002
# automobile	automobile    0.840287536011187	automobile    0.9621785801820574
# science	science    0.9999859788604943	science    0.9999999998506904
# entertainment	entertainment    0.9998763263659391	entertainment    1.0
# sports	sports    0.9999827604127778	sports    1.0
# world	world    0.9999759659436347	world    0.9999846134393643
# technology	technology    0.9999703635211259	technology    0.9999999586006364
# world	entertainment    0.9997117693572259	entertainment    0.999841171141483
# politics	politics    0.9968142992680179	politics    0.9999999827421692
# world	world    0.9913107692951971	world    0.9889818847947482
# automobile	automobile    0.9999954406568946	automobile    0.9999998500754207
# entertainment	entertainment    0.999566802170355	entertainment    1.0
# sports	sports    0.9998919329125632	sports    0.9999998555020564
# politics	politics    0.9968053016855728	politics    0.999999352405426
# entertainment	politics    0.990869432481639	politics    0.9988296320267414
# entertainment	entertainment    0.9995657297811864	entertainment    1.0
# sports	sports    0.9999541524335852	sports    1.0
# technology	politics    0.9995577466373773	politics    0.9999986290430039
# entertainment	entertainment    0.9998752536737965	entertainment    1.0
# entertainment	entertainment    0.9999539140405167	entertainment    0.9999999804443133
# technology	technology    0.9995782334862401	technology    0.9997939371626605
# technology	technology    0.5618467433131561	technology    0.7308859627954435
# technology	entertainment    0.9999865748705684	entertainment    0.9999999961492583
# world	politics    0.9990850325898026	politics    1.0
# entertainment	entertainment    0.9999034944852412	entertainment    1.0
# world	politics    0.9984856899703688	politics    1.0
# automobile	automobile    0.9999325033894579	automobile    0.9999956245395507
# world	politics    0.9994389033746082	politics    0.999999887464815
# entertainment	entertainment    0.9999722707259225	entertainment    1.0
# sports	sports    0.9999639267365671	sports    0.9999999999915765
# entertainment	entertainment    0.9997382216244636	entertainment    1.0
# science	science    0.9999852636486923	science    0.9999991684723075
# politics	politics    0.9947629317718498	politics    0.9999999970010367
# world	politics    0.9990836034463759	politics    0.9999986290423329
# politics	politics    0.9996566101691926	politics    0.9999999980375264
# automobile	automobile    0.9999037776573207	automobile    0.9999896933580293
# entertainment	entertainment    0.9997384599806582	entertainment    1.0
# entertainment	entertainment    0.9996641174254821	entertainment    1.0
# sports	sports    0.9999638075352227	sports    0.9999999999925664
# science	science    0.9299121613253334	science    0.952565562878431
# sports	sports    0.9999782307609466	sports    1.0
# automobile	technology    0.9967780532086479	technology    0.9968265826937065
# world	world    0.7956673620455941	world    0.6780046398974766
# sports	sports    0.9999412791109633	sports    1.0
# world	politics    0.8765337211566709	politics    0.88053640439528
# science	technology    0.7764688605768941	science    0.7298004680819405
# sports	sports    0.9999717939239275	sports    1.0
# technology	technology    0.9999708403224388	technology    0.9999996114737157
# sports	sports    0.9999918197752665	sports    0.999999999987744
# entertainment	entertainment    0.9999030177160398	entertainment    1.0
# politics	politics    0.9999024217566767	politics    1.0
# sports	sports    0.9999720323248966	sports    0.9999999999682796
# technology	world    0.7103349258073367	crime 0.0
# politics	politics    0.9996616149907568	politics    0.9999999927199527
# entertainment	entertainment    0.9999543908303483	entertainment    1.0
# technology	politics    0.9985734010549849	politics    0.999784793159823
# politics	politics    0.9980669241713224	politics    1.0
# automobile	automobile    0.9999428734622954	automobile    0.9999999715466439
# world	politics    0.9984852138554035	politics    0.9999999317440069
# science	science    0.999948311739736	science    0.9999898581462004
# sports	sports    0.9999410407209638	sports    1.0
# sports	sports    0.9999720323248966	sports    1.0
# automobile	politics    0.6335681181939233	politics    0.4787667707859235
# entertainment	entertainment    0.999873465846416	entertainment    0.9999986290431997
# world	world    0.9999806147849026	world    0.9999998971275166
# sports	sports    0.9999721515254024	sports    1.0
# entertainment	politics    0.9995676361869676	politics    1.0
# science	science    0.9626221821097908	science    0.9626711570233125
# entertainment	entertainment    0.999664355734915	entertainment    1.0
# world	politics    0.994746180857682	politics    0.9914171854695586
# world	politics    0.998056119579275	politics    0.9999994284993518
# sports	sports    0.9999403255403931	sports    0.9999999119993025
# entertainment	entertainment    0.9999832372181827	entertainment    1.0
# automobile	technology    0.9857266232156799	technology    0.9706750137013077
# automobile	automobile    0.8948197418647065	business    0.6269310122564773
# automobile	automobile    0.9999849508371957	automobile    0.9999999918479846
# politics	politics    0.9994441417562824	politics    1.0
# sports	sports    0.9999720323248966	sports    1.0
# sports	sports    0.9999974223173038	sports    0.9999999999981205
# politics	politics    0.9984730919472161	politics    0.999995522129431
# entertainment	entertainment    0.9999405639338598	entertainment    1.0
# sports	sports    0.9999059975254592	sports    0.999996855360126
# world	politics    0.9992769857374171	politics    0.9999999226557662
# automobile	automobile    0.9996807187503789	automobile    0.9998796047286673
# entertainment	entertainment    0.9995688277360644	entertainment    1.0
# world	science    0.8166073350184915	science    0.9238747965771784
# science	science    0.9999776347567552	science    0.9999910184344117
# automobile	automobile    0.9999931757999936	automobile    0.9999998874648027
# politics	politics    0.9992793680859164	politics    0.9999992598301689
# entertainment	politics    0.9947629317718498	politics    0.9999999865595917
# entertainment	entertainment    0.9999034944852412	entertainment    0.9999999966017348
# technology	politics    0.9509829908522192	politics    0.9525486595719885
# politics	politics    0.9975236091415532	politics    1.0
# politics	politics    0.9996628066227882	politics    1.0
# automobile	automobile    0.9998295346826981	automobile    0.9998755561545586
# entertainment	entertainment    0.9999036136757582	entertainment    1.0
# sports	sports    0.9999883629029224	sports    1.0
# sports	sports    0.9999027793278867	sports    0.9999999999877439
# automobile	automobile    0.9999875732816766	automobile    0.9999999976644054
# sports	sports    0.9999720323248966	sports    1.0
# world	politics    0.9984834358655864	politics    0.9999415962128672
# sports	sports    0.9998408123762379	sports    0.9999545987413849
# politics	politics    0.999565610623673	politics    0.9999999981810398
# world	politics    0.8802739388182581	politics    0.9046250034687754
# sports	sports    0.9999975415208362	sports    0.999999999994891
# science	science    0.9996271920492561	science    0.9992902224687225
# science	science    0.999871558828247	science    0.9999633669820807
# world	politics    0.9994433114204889	politics    0.9999999983947722
# entertainment	entertainment    0.999841289108504	entertainment    1.0
# world	politics    0.998826652613969	politics    0.9999999847700272
# entertainment	entertainment    0.9997382216244636	entertainment    1.0
# world	world    0.9998521313151989	world    0.9974961034123181
# sports	sports    0.999840931566558	sports    1.0
# world	politics    0.9965463213372726	politics    0.9991314383166511
# entertainment	entertainment    0.9997384599806582	entertainment    1.0
# world	science    0.9991422674080581	science    0.9999994241402795
# world	world    0.7767369672778386	world    0.8174044888645012
# sports	sports    0.9999639267365671	sports    1.0
# world	politics    0.8783115283799283	politics    0.8166351568063328
# technology	technology    0.9996641174254821	technology    0.9999999998308102
# world	world    0.9239788013270045	world    0.8173697066843334
# sports	sports    0.9999865748705684	sports    1.0
# sports	sports    0.9999976607241555	sports    1.0
# entertainment	entertainment    0.9995851443836734	entertainment    0.9996848549882974
# science	science    0.9999720323248966	science    0.9999898656389011
# politics	politics    0.9992823460586108	politics    0.9999999950555483
# sports	sports    0.9999932502087602	sports    1.0
# entertainment	politics    0.8150088789089316	politics    0.816964266426885
# world	world    0.9046107330281362	world    0.9682254310984123
# world	politics    0.999660304191344	politics    0.9999989322971534
# sports	sports    0.9999864556683433	sports    0.9999999999964886
# world	science    0.9972520444950035	science    0.9999646437605069
# world	science    0.9988851713645449	science    0.9998415001530553
# sports	sports    0.999978111559716	sports    1.0
# world	politics    0.9984501540055941	politics    0.9999645968355668
# world	politics    0.9988242749494053	politics    1.0
# entertainment	entertainment    0.9999033752947384	entertainment    1.0
# world	politics    0.9988163050237228	politics    0.9999970976882104
# world	politics    0.9995648957370732	politics    0.9999987901337809
# technology	technology    0.9995648957370732	technology    0.9999972468311543
# world	politics    0.9997357190042349	politics    1.0
# world	technology    0.6841570162690048	technology    0.7119174486501557
# politics	politics    0.9994368780140078	politics    1.0
# sports	sports    0.9999622579445043	sports    1.0
# politics	politics    0.9984797498247381	politics    0.99999800720741
# technology	technology    0.9996287411061934	technology    0.9999785228156615
# entertainment	entertainment    0.9997382216244636	entertainment    1.0
# world	world    0.9894723842927783	world    0.9875199180541144
# world	politics    0.9980670430339881	politics    0.9999999976644076
# world	politics    0.658904485400888	politics    0.6655044346369431
# sports	sports    0.9999642843334096	sports    1.0
# science	science    0.99969794946702	science    0.9993734806230257
# world	politics    0.9942798625475954	politics    0.9914155878139476
# sports	sports    0.9999961110814314	sports    0.9999999999987083
# sports	sports    0.9999239957468543	sports    0.9999999976021677
# world	politics    0.9988242749494053	politics    0.9999999983947706
# politics	politics    0.9997338122487848	politics    0.9999999983947706
# world	world    0.7302969883784927	world    0.7304086656959142
# world	world    0.9972113805821224	world    0.9940709353830824
# sports	sports    0.9999539140405167	sports    0.9999999986949188
# sports	sports    0.9999914621676675	sports    1.0
# politics	politics    0.9994422391928954	politics    1.0
# world	politics    0.6221255618379452	politics    0.6224057384589994
# technology	technology    0.9999670259219393	technology    0.9999997436685115
# entertainment	entertainment    0.9999411599159564	entertainment    1.0
# automobile	technology    0.814862343452846	automobile    0.7768091976958365
# entertainment	politics    0.9995562013268195	politics    0.9999999895326055
# automobile	automobile    0.9999520516617789	automobile    1.0
# science	science    0.9706407524728845	science    0.9859347014626427
# automobile	automobile    0.9997314736041512	automobile    0.9998412407819849
# entertainment	entertainment    0.9999036136757582	entertainment    1.0
# automobile	automobile    0.9999017513943407	automobile    0.9999999987498461
# world	politics    0.9667835280165217	politics    0.9706618215029988
# science	science    0.9986159611588561	science    0.9999535618304286
# entertainment	entertainment    0.9999714363225806	entertainment    1.0
# sports	sports    0.9999717939239275	sports    1.0
# world	politics    0.9947583292028692	politics    0.9999999847700115
# world	science    0.9410251921416626	science    0.9623016252902358
# entertainment	entertainment    0.999924830105154	entertainment    1.0
# sports	sports    0.9999243533330548	sports    1.0
# entertainment	entertainment    0.9996641174254821	entertainment    1.0
# technology	business    0.9650866651070802	business    0.9919269452074374
# technology	politics    0.9992419988253365	politics    0.9999140612052032
# world	world    0.9982229889835629	world    0.9947609099548987
# technology	politics    0.9997064103985385	politics    0.9999796570387175
# technology	business    0.9710237700720215	business    0.9919019179000864
# automobile	automobile    0.9974467869929481	automobile    0.9959257970262794
# entertainment	entertainment    0.9997384599806582	entertainment    1.0
# technology	technology    0.999629813562964	technology    0.9998407696935279
# entertainment	entertainment    0.9996642365947385	entertainment    1.0
# science	science    0.9999070702578599	science    0.9999545928860755
# sports	sports    0.9999721515254024	sports    1.0
# entertainment	entertainment    0.9997375065853166	entertainment    0.9999999760808737
# entertainment	entertainment    0.9999033752947384	entertainment    0.9999999317439741
# entertainment	entertainment    0.999841289108504	entertainment    1.0
# sports	sports    0.9999535564437448	sports    0.9999999961263615
# sports	sports    0.9999722707259225	sports    1.0
# technology	technology    0.993180238192744	technology    0.990284155726818
# entertainment	entertainment    0.9999237573536999	entertainment    1.0
# world	world    0.9935146502857483	world    0.9963364512251168
# technology	politics    0.9937880760709612	politics    0.9947660618268855
# technology	technology    0.9999030177160398	technology    0.9999999998506905
# world	politics    0.987384838323943	politics    0.9959289277657726
# world	entertainment    0.9998473675593947	entertainment    0.9997378607105618
# """


# def compute_ece(confidences, correctness, n_bins=10):
#     """
#     Compute Expected Calibration Error (ECE).

#     Parameters:
#     - confidences: np.array of confidence scores (floats between 0 and 1)
#     - correctness: np.array of correctness indicators (1 if correct, 0 if incorrect)
#     - n_bins: number of bins to partition confidence scores into

#     Returns:
#     - ece: float, the expected calibration error
#     """
#     bins = np.linspace(0, 1, n_bins + 1)
#     ece = 0.0

#     for i in range(n_bins):
#         bin_lower = bins[i]
#         bin_upper = bins[i + 1]

#         # Find indices of samples whose confidence falls into current bin (exclusive lower, inclusive upper)
#         in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
#         prop_in_bin = np.mean(in_bin)  # fraction of samples in this bin

#         if prop_in_bin > 0:
#             avg_confidence = np.mean(confidences[in_bin])
#             avg_accuracy = np.mean(correctness[in_bin])
#             ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin

#     return ece


# for line in data.strip().split("\n"):
#     parts = line.strip().split("\t")
#     if len(parts) != 3:
#         print(f"Skipping malformed line: {line}")
#         continue

#     true_label = parts[0].strip()

#     m1_pred_conf = parts[1].strip().split()
#     m1_pred = m1_pred_conf[0]
#     # Handle multi-word labels in prediction (like "South Africa") by joining all except last element as label
#     if len(m1_pred_conf) > 2:
#         m1_pred = " ".join(m1_pred_conf[:-1])
#     m1_conf = float(m1_pred_conf[-1])
#     m1_corr = 1 if m1_pred == true_label else 0

#     m2_pred_conf = parts[2].strip().split()
#     m2_pred = m2_pred_conf[0]
#     if len(m2_pred_conf) > 2:
#         m2_pred = " ".join(m2_pred_conf[:-1])
#     m2_conf = float(m2_pred_conf[-1])
#     m2_corr = 1 if m2_pred == true_label else 0

#     model1_confidences.append(m1_conf)
#     model1_correct.append(m1_corr)
#     model2_confidences.append(m2_conf)
#     model2_correct.append(m2_corr)


# print(model1_confidences, model1_correct)
# print(model2_confidences, model2_correct)

# # ece_model1 = compute_ece(np.array(model1_confidences),
# #                          np.array(model1_correct))
# # ece_model2 = compute_ece(np.array(model2_confidences),
# #                          np.array(model2_correct))

# # print(f"ECE for Model 1: {ece_model1:.4f}")
# # print(f"ECE for Model 2: {ece_model2:.4f}")
