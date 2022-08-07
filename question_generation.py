import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_model = question_model.to(device)

def get_question(context, answer, model, tokenizer):
    torch.cuda.empty_cache()
    text = "context: {} answer: {}".format(context,answer)
    encoding = tokenizer.encode_plus(text, max_length=384, pad_to_max_length=False, truncation=True, return_tensors="pt").to(device)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(input_ids=input_ids,
                          attention_mask=attention_mask,
                          early_stopping=True,
                          num_beams=5,
                          num_return_sequences=1,
                          no_repeat_ngram_size=2,
                          max_length=72)

    dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]

    question = dec[0].replace("question:","")
    question = question.strip()

    return question

def generate_question(summarized_text, keywords):
    question_answer = {}

    for answer in keywords:
        question = get_question(summarized_text, answer, question_model, question_tokenizer)
        question_answer[question] = answer

    return question_answer

