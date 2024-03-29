import torch
import models

import nltk
#nltk.download('punkt')
#nltk.download('brown')
#nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize

def postprocesstext(content):
  final = ""
  for sent in sent_tokenize(content):
    sent = sent.capitalize()
    final = final + " " + sent
  return final

def summarizer(original_text, model, tokenizer, device):
  original_text = original_text.strip().replace("\n"," ")
  original_text = "summarize: " + original_text
  max_len = 512
  encoding = tokenizer.encode_plus(original_text, max_length = max_len, pad_to_max_length = False,
                                   truncation = True, return_tensors = "pt").to(device)

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids = input_ids,
                        attention_mask = attention_mask,
                        early_stopping = True,
                        num_beams = 4,
                        num_return_sequences = 1,
                        no_repeat_ngram_size = 2,
                        min_length = 100,
                        max_length = 300)


  dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
  summary = dec[0]
  summary = postprocesstext(summary)
  summary = summary.strip()

  return summary


def summarize_text(original_text):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    summary_model = models.summary_model.to(device)

    #torch.cuda.empty_cache()

    summarized_text = summarizer(original_text, summary_model, models.summary_tokenizer, device)

    return summarized_text