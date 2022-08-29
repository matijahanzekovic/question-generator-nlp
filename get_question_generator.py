from summarize import summarize_text
from keyword_extractor import get_keywords
from nltk.tokenize import sent_tokenize
from question_generation import generate_question
from generate_distractors import get_distractors
from pydantic import BaseModel
from typing import List
import random
import models

class QuestionGeneratorResponse(BaseModel):
    question: str = ""
    answer: str = ""
    distractors: List[str] = []

def build_question_generator(key, value, distractors):
    question_generator = QuestionGeneratorResponse()

    question_generator.question = key
    question_generator.answer = value
    question_generator.distractors = distractors

    return question_generator

def get_question_generator_response(original_text):
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
                partial_text = ""

                question_answer = generate_question(partial_summarized_text, keywords)

                for key, value in question_answer.items():
                    distractors = get_distractors(value, key, models.s2v, models.sentence_transformer_model, 40, 0.2)

                    if len(distractors) >= 3:
                        question_generator = build_question_generator(key, value, random.sample(distractors, 3))
                        question_generator_response.append(question_generator)
    else:
        final_summarized_text = summarize_text(original_text)
        keywords = get_keywords(original_text, final_summarized_text)
        final_keywords.append(keywords)

        question_answer = generate_question(final_summarized_text, keywords)

        for key, value in question_answer.items():
            distractors = get_distractors(value, key, models.s2v, models.sentence_transformer_model, 40, 0.2)

            if len(distractors) >= 3:
                question_generator = build_question_generator(key, value, random.sample(distractors, 3))
                question_generator_response.append(question_generator)
            
    return question_generator_response
