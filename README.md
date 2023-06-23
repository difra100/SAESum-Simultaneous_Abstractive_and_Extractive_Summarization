# SAESum: Simultaneous Abstractive and Extractive Summarization.
In this repository we are developing a neural-based system capable of producing both extractive and abstractive summaries of any document, simultaneously. In our method are involved some of the current and past SoTA architecture in text summarization.

## Previous Work on simultaneous A&E
Similar approaches were attempted in the past:  
https://link.springer.com/article/10.1007/s41019-019-0087-7 (Extractive helps abstractive summarization).   
But we do not draw inspiration from these.

## How to use this repository  
In main.ipynb


### Project Contributors
* Andrea Giuseppe Di Francesco, 1836928
* Antonio Scardino, 2020613

doc_mask, selected_y_label, selected_score, valid_sen_idxs = batch['doc_mask'], batch[
        'selected_y_label'], batch['selected_score'], batch['valid_sen_idxs']
    

    seqs = {}

    batch['input_ids'] = batch['input_ids'].to(device)
    batch['attention_mask'] = batch['attention_mask'].to(device)

    seqs['input_ids'] = batch['input_ids'].view(-1, max_doc_len*max_seq_len)
    seqs['attention_mask'] = batch['attention_mask']


    doc_mask = doc_mask.to(device)
    selected_y_label = selected_y_label.to(device)
    selected_score = selected_score.to(device)
    valid_sen_idxs = valid_sen_idxs.to(device)

    num_documents = seqs['input_ids'].size(0)
    num_sentences = seqs['input_ids'].size(1)