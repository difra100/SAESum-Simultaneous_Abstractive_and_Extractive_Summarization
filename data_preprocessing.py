from src.MemSum.src.data_preprocessing.MemSum.utils import greedy_extract
import json
from tqdm import tqdm
import sys

def extract_dataset(filename, target_filename):
    train_corpus = [ json.loads(line) for line in open(filename) ]
    for data in tqdm(train_corpus):
        high_rouge_episodes = greedy_extract( data["text"], data["summary"], beamsearch_size = 2 )
        indices_list = []
        score_list  = []

        for indices, score in high_rouge_episodes:
            indices_list.append( indices )
            score_list.append(score)

        data["indices"] = indices_list
        data["score"] = score_list

    with open(target_filename,"w") as f:
        for data in train_corpus:
            f.write(json.dumps(data) + "\n")

if __name__ == '__main__':
    filename = sys.argv[1]
    target_filename = sys.argv[2]
    extract_dataset(filename, target_filename)

