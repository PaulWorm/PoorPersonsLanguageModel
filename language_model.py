import sys, os
import numpy as np
from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt

STARTTOKEN = '<|start|>'
ENDTOKEN = '\n'
DEFAULTTRAININGDATA = './data/TrainingData.txt'

def prepare_tokens(line):
    tokens = line.rstrip().split(sep=' ')
    tokens.append('\n')
    return tokens

def prepare_prompt(prompt):
    tokens = prompt.split(sep=' ')
    return tokens

# load training data
def get_token_count(fname):
    token_count = defaultdict(int)
    with open(fname) as file:
        for line in file:
            tokens = prepare_tokens(line)
            for token in tokens:
                token_count[token] += 1
    return token_count

def get_counts(prob_table,token_2_int,fname):
    with open(fname) as file:
        for line in file:
            tokens = prepare_tokens(line)
            last = STARTTOKEN
            for curr in tokens:
                i,j = token_2_int[last], token_2_int[curr]
                prob_table[i,j] += 1
                last = curr
    for i in range(prob_table.shape[-1]):
        if sum(prob_table[i,:])>0:
            prob_table[i,:] /= sum(prob_table[i,:])
    return prob_table


def train_model(training_data_file=DEFAULTTRAININGDATA):
    token_count = get_token_count(training_data_file)
    tokens = token_count.keys()
    token_2_int = {token : i+1 for i,token in enumerate(tokens)}
    int_2_token = {i+1 : token for i,token in enumerate(tokens)}
    token_2_int[STARTTOKEN] = 0
    int_2_token[0] = STARTTOKEN
    n_tokens = len(tokens)
    prob_table = np.zeros((n_tokens+1,n_tokens+1))
    prob_table = get_counts(prob_table,token_2_int,training_data_file)
    return prob_table, tokens, token_2_int, int_2_token

class PoorPersonsLanguageModel():

    max_tokens = 20
    def __init__(self,training_data_file = DEFAULTTRAININGDATA):
        self.prob_table, self.tokens, self.token_2_int, self.int_2_token = train_model(training_data_file)

    def get_next_token(self,token):
        probs = self.prob_table[self.token_2_int[token],:]
        rand = np.random.choice(np.arange(len(probs)),1,True,probs)[0]
        next_token = self.int_2_token[rand]
        return next_token

    def check_prompt(self,prompt):
        for token in prompt:
            if token not in self.tokens:
                return 'I am not trained on some of the words contained in the prompt.\n'
        return 1

    def create_sentence(self,prompt = None):
        if not prompt:
            prompt = [STARTTOKEN]
            sentence = [self.get_next_token(prompt[-1])]
        else:
            prompt = prepare_prompt(prompt)
            check = self.check_prompt(prompt)
            if  check == 1:
                sentence = prompt
            else:
                return check
        for i in range(self.max_tokens):
            sentence.append(self.get_next_token(sentence[-1]))
            if(sentence[-1] == ENDTOKEN):
                break
        return ' '.join(sentence)
            

def main():
    model = PoorPersonsLanguageModel()
    while True:
        prompt = input('User: ')
        if prompt == 'exit':
            break
        print('')
        print('Model:', model.create_sentence(prompt))



if __name__ == '__main__':
    main()