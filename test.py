import numpy as np
import tensorflow as tf

from collections import Counter

path_train = "./data/sentences/sentences.train"
path_vocab = "./data/vocab.txt"

special = {
    "bos" : "<bos>",
    "eos" : "<eos>",
    "pad" : "<pad>"
}

sentence_length = 30
batch_size = 64

print(tf.__version__)

def build_vocab(input_file, output_file, top_k=None, special=None):  
    '''
    builds a vocubulary output_file of size top_k, taking the most frequent words 
    in the input_file and also adding the special symbols from the given dict
    '''
    with open(input_file) as f:
        wordcount = Counter(f.read().split())
        wordcount = wordcount.most_common(top_k-len(special)-1)
        
    with open(output_file, "w") as f:
        for symbol in special.values():
            f.write(f"{symbol}\n")
            
        for word, _ in wordcount:
            f.write(f"{word}\n")
    
build_vocab(input_file=path_train, output_file=path_vocab, top_k=20000, special=special)