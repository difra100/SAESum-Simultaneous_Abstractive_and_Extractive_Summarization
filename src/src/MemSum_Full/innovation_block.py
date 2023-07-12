import torch
import torch.nn.functional as F
import re
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def create_list_of_sentences(abstracts):

    # Regular expression is used to distinguish what are the sentences
    sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'
    batch_of_lists = []
    for abstract in abstracts:
    # Use the regular expression to split the text into sentences
        list_of_sentences = re.split(sentence_pattern, abstract)

        # Remove leading and trailing spaces from each sentence
        list_of_sentences = [sentence.strip() for sentence in list_of_sentences]
        batch_of_lists.append(list_of_sentences)

    

    return batch_of_lists


def tokenize_list_of_sentences(sentences, tokenizer, max_len):
    
    desired_length = max_len  # Desired length for padding

    batch_encoding = tokenizer.batch_encode_plus(
        sentences,
        truncation=True,
        max_length=desired_length,
        padding='max_length',
        return_tensors='pt'
    )

    input_ids = batch_encoding["input_ids"]
    attention_mask = batch_encoding["attention_mask"]

    return batch_encoding

def get_the_longest_sentence(tokenized_list):
    max_n = 0
    for token in tokenized_list:
        if token['input_ids'].shape[0] > max_n:
            max_n = token['input_ids'].shape[0]
    return max_n


def get_abstract_tensor(tokenized_abstracts, n_sentences, device):
    # tokenized_abstracts: List: [n_sentences x max_len]* batch_size
    abstract_input_ids = torch.tensor([], device = device)
    abstract_attention_mask = torch.tensor([], device = device)
    abstract_tensor = {}

    for abstract in tokenized_abstracts:
        abstract_inp = F.pad(abstract['input_ids'], pad = (0, 0, 0, n_sentences - abstract['input_ids'].shape[0]))
        abstract_att = F.pad(abstract['attention_mask'], pad = (0, 0, 0, n_sentences - abstract['attention_mask'].shape[0]))
       
        abstract_input_ids = torch.cat((abstract_input_ids, abstract_inp.unsqueeze(0).to(device)), dim = 0)
        abstract_attention_mask = torch.cat((abstract_attention_mask, abstract_att.unsqueeze(0).to(device)), dim = 0)
    
    abstract_tensor['input_ids'] = abstract_input_ids
    abstract_tensor['attention_mask'] = abstract_attention_mask

    return abstract_tensor

def get_embeddings(model, dict, batch_size, n_sentences, n_tokens):

    new_dict = {}
    new_dict['input_ids'] = dict['input_ids'].reshape(batch_size, n_sentences*n_tokens)
    new_dict['attention_mask'] = dict['attention_mask'].reshape(batch_size, n_sentences*n_tokens)

    if torch.cuda.is_available():
        new_dict['input_ids'] = new_dict['input_ids'].type(torch.cuda.LongTensor)
        new_dict['attention_mask'] = new_dict['attention_mask'].type(torch.cuda.LongTensor)

    else:
        new_dict['input_ids'] = new_dict['input_ids'].type(torch.LongTensor)
        new_dict['attention_mask'] = new_dict['attention_mask'].type(torch.LongTensor)


    # DIMENSIONS: batch_size x (n_sent*n_tokens) x embedding_dim
    embedded_tensor = model.model.encoder(**new_dict).last_hidden_state
    # print("Embedding dimension: ", embedded_tensor.shape)
    # DIMENSIONS: batch_size x n_sent x n_tokens x embedding_dim
    embedded_tensor = embedded_tensor.reshape(batch_size, n_sentences, n_tokens, -1)
    # print("Embedding dimension after reshape: ", embedded_tensor.shape)


    return embedded_tensor.detach()

class AbstractToExtractConverter(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.lstm = torch.nn.LSTM(dim, dim, 2, batch_first = True, bidirectional = True)
        self.norm_out = torch.nn.LayerNorm(2*dim)
        self.ln_out = torch.nn.Linear(2*dim, dim)

    def forward(self, x):

        B, S, T, dim = x.shape
        x = x.reshape(B*S, T, dim)
        # print("LSTM input: ", x.shape)

        output, (h, c) = self.lstm(x)
        # print("LSTM output: ", output.shape)
        output = output.reshape(B, S, T, 2*dim)
        
        # Permutation invariant transform.: B x S x T x dim ---> B x S x dim
        output = torch.sum(output, dim = 2)
        # print("After LSTM and summ: ", output.shape)


        # Activation
        output = F.relu(output)
        x = self.norm_out(output)
        x = F.relu(self.ln_out(x))

        # print("CONVERTER OUTPUT: ", x.shape)

        return x
class InnovationBlock(torch.nn.Module):

    def __init__(self, dim, n_heads):
        super().__init__()
        self.converter_net = AbstractToExtractConverter(dim)
        self.attn_layer = torch.nn.MultiheadAttention(dim, n_heads, batch_first = True)

    def forward(self, embedded_abs, embedded_global):
        # embedded_global is key, converter_out is associated with valye and key
        converter_out = self.converter_net(embedded_abs)

        output_attention, _ = self.attn_layer(embedded_global, converter_out, converter_out)

        return output_attention


def get_embedded_abstract_from_abs(texts, tokenizer, n_tokens, device, peg_encoder):
    
    # List of sentences
    batch_of_abstracts = create_list_of_sentences(texts)
    # print("Sentences of the abstracts: \n", batch_of_abstracts)
    # get lists of dicts (input_ids: , attention_mask: )
    tokenized_sentences = [tokenize_list_of_sentences(abstract, tokenizer, max_len = n_tokens) for abstract in batch_of_abstracts]
    # print("Tokenized sentences: \n", tokenized_sentences)
    
    # print("Sentences of the abstracts: \n", tokenized_sentences[0]['input_ids'].shape)
    
    # print(tokenized_sentences[0]['input_ids'].shape)
    max_sentences = get_the_longest_sentence(tokenized_sentences)
    batch_size = len(texts)
    # print("max_sentences: ", max_sentences)

    # get dict (input_ids: shape batch_size x n_sentences x max_len, attention_mask .........)
    abstract_tensor = get_abstract_tensor(tokenized_sentences, max_sentences, device)
    # print("Tokenized Diz: \n", abstract_tensor)

    # get embeddings 

    embedded_abstract_tensor = get_embeddings(peg_encoder, abstract_tensor, batch_size, max_sentences, n_tokens)

    return embedded_abstract_tensor

if __name__ == '__main__':

    

    texts = ["This is a sample text. It contains multiple sentences.\
                 Each sentence ends with a period. Or sometimes a question mark?", 
                 "This is a sample text. It contains multiple sentences.\ Or sometimes a question mark?",
                 "In this case, len(x) calculates the length of each tokenized text, which corresponds to \
                the number of sentences in that text. By comparing the lengths. The max function will \
                determine the tokenized text. With the maximum number of sentences. Assign it to the max_sentences variable."]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_checkpoint = "google/pegasus-x-base"  # Use pegasus-x-base-finetuned-xsum
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    max_len = 100
    model_s = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    model_s.to(device)


    embedded_vec = get_embedded_abstract_from_abs(texts, tokenizer, max_len, device, model_s)

    # global_enc MUST be taken from the global sentence encoder
    global_enc = torch.randn((3, 1000, 768), device = device)

    innovation = InnovationBlock(dim = 768, n_heads = 8).to(device)

    innovated_tensor = innovation(embedded_vec, global_enc)

    print("INNOVATED TENSOR: ", innovated_tensor.shape)
    
