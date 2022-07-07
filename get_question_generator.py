from summarize import summarize_text
from keyword_extractor import get_keywords
from nltk.tokenize import sent_tokenize
from question_generation import generate_question
from generate_distractors import get_distractors
from sense2vec import Sense2Vec
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List

class QuestionGeneratorResponse(BaseModel):
    question: str = ""
    answer: str = ""
    distractors: List[str] = []

def get_question_generator_response(original_text):
    s2v = Sense2Vec().from_disk('s2v_old')
    sentence_transformer_model = SentenceTransformer('msmarco-distilbert-base-v3')

    # torch.cuda.empty_cache()

    i = 0
    final_summarized_text = ""
    partial_text = ""
    final_keywords = []
    question_answer = {}
    sentences = sent_tokenize(original_text)

    question_generator_response = []

    if len(sentences) >= 10:
        for sent in sent_tokenize(original_text):
            partial_text = partial_text + " " + sent
            i += 1
            if len(sentences) >= 10 and i % 10 == 0:
                partial_summarized_text = summarize_text(partial_text)
                final_summarized_text += partial_summarized_text
                keywords = get_keywords(partial_text, partial_summarized_text)
                final_keywords.append(keywords)
                # print("KEYWORDS: ", keywords)
                partial_text = ""

                # print("\nQUESTIONS: \n")
                question_answer = generate_question(partial_summarized_text, keywords)
                # print("\nQUESTION_ANSWER:\n")
                # print(question_answer)
                # print("\n")

                for key, value in question_answer.items():
                    distractors = get_distractors(value, key, s2v, sentence_transformer_model, 40, 0.2)

                    question_generator = QuestionGeneratorResponse()
                    question_generator.question = key
                    question_generator.answer = value
                    question_generator.distractors = distractors

                    question_generator_response.append(question_generator)

                    # print("\nDISTRACTORS: \n")
                    # print(get_distractors(value, key, s2v, sentence_transformer_model, 40, 0.2))
    else:
        final_summarized_text = summarize_text(original_text)
        keywords = get_keywords(final_summarized_text, final_summarized_text)
        final_keywords.append(keywords)
        # print("KEYWORDS: ", keywords)
        partial_text = ""

        # print("\nQUESTIONS: \n")
        question_answer = generate_question(final_summarized_text, keywords)
        # print("\nQUESTION_ANSWER:\n")
        # print(question_answer)
        # print("\n")

        for key, value in question_answer.items():
            distractors = get_distractors(value, key, s2v, sentence_transformer_model, 40, 0.2)

            question_generator = QuestionGeneratorResponse()
            question_generator.question = key
            question_generator.answer = value
            question_generator.distractors = distractors

            question_generator_response.append(question_generator)
            
    return question_generator_response