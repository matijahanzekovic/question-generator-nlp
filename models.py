from sense2vec import Sense2Vec
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

s2v = Sense2Vec().from_disk('s2v_old')

sentence_transformer_model = SentenceTransformer('msmarco-distilbert-base-v3')

question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')

question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')

summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')

summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')