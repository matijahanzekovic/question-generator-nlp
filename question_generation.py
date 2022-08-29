import torch
import models

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
question_model = models.question_model.to(device)

#torch.cuda.empty_cache()

def get_question(context, answer, model, tokenizer):
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
        question = get_question(summarized_text, answer, question_model, models.question_tokenizer)
        question_answer[question] = answer

    return question_answer

