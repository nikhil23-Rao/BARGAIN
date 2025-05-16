
import numpy as np
import pandas as pd
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
        # store final result
        classifier_output = {}
        for classifier in classifiers:
            classifier_output[classifier] = {"confidence_score": 0}  # init

        # run LLM
        task_with_data = self.task.format(data_record)
        prompt = [
            {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
            {"role": "user", "content": task_with_data}
        ]
        response = self.client.beta.chat.completions.parse(
            model=self.model, messages=prompt, logprobs=True, seed=0, temperature=0, max_tokens=1002, top_logprobs=10)
        logprobs = response.choices[0].logprobs.content

        # 1.) loop through all tokens for each step and see where all each classifier is mentioned
        # 2.) Append all logprob values to corresponding classifier_key
        # 2a.) if classifier is multiple words -> look to second token to make sure its referring to correct classifier
        for element in logprobs:
            for possible_token in element.top_logprobs:
                matched_classifier = None
                for c in classifiers:
                    if (c.startswith(possible_token.token) and (not self.determine_multi_step_classifier(c))):
                        # tiger
                        matched_classifier = c
                        print(element, possible_token)
                        classifier_output[c]["confidence_score"] += possible_token.logprob
                    elif (self.determine_multi_step_classifier(c)):
                        steps = c.strip().split()
                        matched_classifier = steps[-1]
                        if (matched_classifier.startswith(possible_token.token)):
                            print(possible_token)
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
            model=self.model, messages=prompt, logprobs=True, seed=0, temperature=0, max_tokens=1000, top_logprobs=10)
        res = response.choices[0].message.content
        logprobs = response.choices[0].logprobs.content
        return get_bool_val_prob(res, logprobs)

    def proxy_func(self, data_record):
        print("new starting")
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
        "lion", "tiger", "elephant", "giraffe", "zebra",
                "kangaroo", "panda", "koala", "dolphin", "whale",
                "eagle", "falcon", "bear", "wolf", "fox",
                "rabbit", "deer", "monkey", "hippopotamus", "rhinoceros"
    ]
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
    return pd.DataFrame.from_dict(data)


# Define Data and Task
df = generate_color_or_animal_data(
    n=100, animal_prop=1, hard_prop=0.5, misleading_text_length=600)
task = ''' 
        I will give you a text. Your task is to extract the name of the animal mentioned is the text.

        Here is the text: {}

        You must respond with ONLY the name of the animal:
        '''

# Define oracle and proxy
proxy = OpenAIProxy(task, model='gpt-4o-mini')
oracle = OpenAIOracle(task, model='gpt-4o')


# Call BARGAIN to process
print("starting process")

bargain = BARGAIN_A(proxy, oracle, target=0.9,  delta=0.1, seed=0)
print(proxy.classifiers_proxy_func(df['value'].to_numpy(), [
    "lion", "tiger", "elephant", "giraffe", "zebra",
    "kangaroo", "panda", "koala", "dolphin", "whale",
                "eagle", "falcon", "bear", "wolf", "fox",
                "rabbit", "deer", "monkey", "hippopotamus", "rhinoceros"
]))
# df['output'] = bargain.process(df['value'].to_numpy())

# # Evaluate output
# df['is_correct'] = df['animal_name'] == df['output']
# print(
#     f"Accuracy: {df['is_correct'].mean()}, Used Proxy: {1-oracle.get_number_preds()/len(df):.2f}")
