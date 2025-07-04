
from generate_toy_data import generate_color_or_animal_data
from BARGAIN.process.BARGAIN_A import BARGAIN_A
from BARGAIN.models.GPTModels import OpenAIProxy, OpenAIOracle
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


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
# print(dir(proxy))
df['output'] = bargain.process(df['value'].to_numpy())

# Evaluate output
df['is_correct'] = df['animal_name'] == df['output']


print(df['output'])
print(
    f"Accuracy: {df['is_correct'].mean()}, Used Proxy: {1-oracle.get_number_preds()/len(df):.2f}")
