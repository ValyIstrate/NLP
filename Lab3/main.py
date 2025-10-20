import pandas as pd
import numpy as np
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from utils import *

# coqa = pd.read_json('http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json')
# coqa.head()
#
# del coqa["version"]
#
# cols = ["text","question","answer"]
# comp_list = []
# for index, row in coqa.iterrows():
#     for i in range(len(row["data"]["questions"])):
#         temp_list = []
#         temp_list.append(row["data"]["story"])
#         temp_list.append(row["data"]["questions"][i]["input_text"])
#         temp_list.append(row["data"]["answers"][i]["input_text"])
#         comp_list.append(temp_list)
#
# new_df = pd.DataFrame(comp_list, columns=cols)
# new_df.to_csv("CoQA_data.csv", index=False)

# data = pd.read_csv("CoQA_data.csv")
# data.head()


if __name__ == '__main__':
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    data = pd.read_csv("CoQA_data.csv")

    question = input("Enter your question here: ")
    context = find_context_for_question(question, data)
    print(f"Context: {context}")

    input_ids = tokenizer.encode(question, context)
    print("The input has a total of {} tokens.".format(len(input_ids)))
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    sep_idx = input_ids.index(tokenizer.sep_token_id)
    print("SEP token index: ", sep_idx)
    num_seg_a = sep_idx + 1
    print("Number of tokens in segment A: ", num_seg_a)
    num_seg_b = len(input_ids) - num_seg_a
    print("Number of tokens in segment B: ", num_seg_b)
    segment_ids = [0] * num_seg_a + [1] * num_seg_b
    assert len(segment_ids) == len(input_ids)

    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    if answer_end >= answer_start:
        answer = " ".join(tokens[answer_start:answer_end + 1])
    else:
        answer = "I am unable to find the answer to this question. Can you please ask another question?"

    print("Question:n{}".format(question.capitalize()))
    print("Answer:n{}.".format(answer.capitalize()))

    from googletrans import Translator

    translator = Translator()

    translated = translator.translate(answer, src='en', dest='ro')
    print("Translated Answer:", translated.text)
