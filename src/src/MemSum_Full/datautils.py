import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from datasets import load_dataset

from tqdm import tqdm

pubmed_dataset = load_dataset("ccdv/pubmed-summarization")


# the corpus has been preprocessed, so here only lower is needed
# all digits are kept, since sent2vec unigram embedding has digit embedding
# no stemming, no lemmatization
class SentenceTokenizer:
    def __init__(self):
        pass

    def tokenize(self, sen):
        return sen.lower()


class Vocab:
    def __init__(self, words, eos_token="<eos>", pad_token="<pad>", unk_token="<unk>"):
        self.words = words
        self.index_to_word = {}
        self.word_to_index = {}
        for idx in range(len(words)):
            self.index_to_word[idx] = words[idx]
            self.word_to_index[words[idx]] = idx
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eos_index = self.word_to_index[self.eos_token]
        self.pad_index = self.word_to_index[self.pad_token]

        self.tokenizer = SentenceTokenizer()

    def index2word(self, idx):
        return self.index_to_word.get(idx, self.unk_token)

    def word2index(self, word):
        return self.word_to_index.get(word, -1)
    # The sentence needs to be tokenized

    def sent2seq(self, sent, max_len=None, tokenize=True):
        if tokenize:
            sent = self.tokenizer.tokenize(sent)
        seq = []
        for w in sent.split():
            if w in self.word_to_index:
                seq.append(self.word2index(w))
        if max_len is not None:
            if len(seq) >= max_len:
                seq = seq[:max_len - 1]
                seq.append(self.eos_index)
            else:
                seq.append(self.eos_index)
                seq += [self.pad_index] * (max_len - len(seq))
        return seq

    def seq2sent(self, seq):
        sent = []
        for i in seq:
            if i == self.eos_index or i == self.pad_index:
                break
            sent.append(self.index2word(i))
        return " ".join(sent)
    


class ExtractionTrainingDataset(Dataset):
    def __init__(self,  corpus, vocab, max_seq_len, max_doc_len, peg_tokenizer=None, two_heads = False):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
        # corpus is a list
        self.corpus = corpus
        self.peg_tokenizer = peg_tokenizer
        self.two_heads = two_heads
        
    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):

        if not self.two_heads:
            doc_data = self.corpus[idx]
        else:
            doc_data = self.corpus[idx][1]
        sentences = doc_data["text"]
        
        # print("\nEXTRACTIVE TEXT: \n", sentences[0])
        

        indices = doc_data["indices"]
        scores = np.array(doc_data["score"])
        summary = doc_data["summary"]

        sentences = sentences[:self.max_doc_len]
        num_sentences_in_doc = len(sentences)

        # This is for RL training
        rand_idx = np.random.choice(len(indices))
        valid_sen_idxs = np.array(indices[rand_idx])

        np.random.shuffle(valid_sen_idxs)

        valid_sen_idxs = valid_sen_idxs[valid_sen_idxs < num_sentences_in_doc]
        selected_y_label = np.zeros(num_sentences_in_doc)
        selected_y_label[valid_sen_idxs] = 1
        selected_score = scores[rand_idx]

        valid_sen_idxs = valid_sen_idxs[:self.max_doc_len]
        valid_sen_idxs = np.array(valid_sen_idxs.tolist(
        ) + [-1] * (self.max_doc_len - len(valid_sen_idxs)))

        selected_y_label = np.array(selected_y_label.tolist(
        ) + [0] * (self.max_doc_len - num_sentences_in_doc))
        sentences += [""] * (self.max_doc_len - num_sentences_in_doc)

        doc_mask = np.array(
            [True if sen.strip() == "" else False for sen in sentences])
        # print("sent: ", sentences[0])
        # print(tokenize_sentences([sentences[0]], self.peg_tokenizer))

        if self.peg_tokenizer != None:
            seqs = tokenize_sentences(
                sentences, self.peg_tokenizer, self.max_seq_len)
            seqs['doc_mask'] = doc_mask
            seqs['selected_y_label'] = selected_y_label
            seqs['selected_score'] = selected_score
            seqs['valid_sen_idxs'] = valid_sen_idxs

            if self.two_heads:
                seqs['abstract'] = self.corpus[idx][0]['abstract']
                # article = self.corpus[idx][0]['article']

            # print("ARTICLE: \n", self.abs_dataset['train'][idx]['article'])

            # print("ABSTRACT: \n", seqs['abstract'][:200])
            # print("Article: \n", article[:200])

            # print("---------------------------------------------------------")

            return seqs

        else:
            seqs = [self.vocab.sent2seq(sen, self.max_seq_len)
                    for sen in sentences]

            seqs = np.asarray(seqs)
            return seqs, doc_mask, selected_y_label, selected_score, valid_sen_idxs


class ExtractionValidationDataset(Dataset):
    def __init__(self,  corpus, vocab, max_seq_len, max_doc_len, peg_tokenizer=None, two_heads = False):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
        # corpus is a list
        self.corpus = corpus
        self.peg_tokenizer = peg_tokenizer
        self.two_heads = two_heads

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):

        if not self.two_heads:
            doc_data = self.corpus[idx]
        else:
            doc_data = self.corpus[idx][1]
        sentences = doc_data["text"]
        # print("EXTRACTIVE val: \n", sentences[0])


        summary = doc_data["summary"]

        sentences = sentences[:self.max_doc_len]
        num_sentences_in_doc = len(sentences)

        sentences += [""] * (self.max_doc_len - num_sentences_in_doc)

        doc_mask = np.array(
            [True if sen.strip() == "" else False for sen in sentences])

        if self.peg_tokenizer != None:
            seqs = tokenize_sentences(
                sentences, self.peg_tokenizer, self.max_seq_len)
            seqs['doc_mask'] = doc_mask
            seqs['sentences'] = sentences

        else:
            seqs = [self.vocab.sent2seq(sen, self.max_seq_len)
                    for sen in sentences]


        summary = summary[:self.max_doc_len]
        if len(summary) < self.max_doc_len:
            summary = summary + [""] * (self.max_doc_len - len(summary))

        if self.peg_tokenizer:
            seqs['summary'] = summary
            if self.two_heads:
                seqs['article'] = self.corpus[idx][0]['article']
                # print("ARTICLE val: \n", seqs['article'][:200])

            return seqs
        else:
            seqs = np.asarray(seqs)
            return seqs, doc_mask, sentences, summary


# pa
def tokenize_sentences(list_of_sentences, tokenizer, max_len, tensor_type = 'np'):
    ''' This function takes as input a list of sentences in the form of

        sentences = [
      "This is the first sentence.",
      "Here is the second sentence.",
      "And another sentence.",
      "A third sentence."
                    ]
    '''
    # Tokenize the sentences as a batch
    desired_length = max_len  # Desired length for padding

    batch_encoding = tokenizer.batch_encode_plus(
        list_of_sentences,
        truncation=True,
        max_length=desired_length,
        padding='max_length',
        return_tensors=tensor_type
    )

    input_ids = batch_encoding["input_ids"]
    attention_mask = batch_encoding["attention_mask"]

    return batch_encoding
