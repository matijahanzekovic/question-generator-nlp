from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from get_question_generator import get_question_generator_response

app = FastAPI()
class QuestionGeneratorRequest(BaseModel):
    text: str

class QuestionGeneratorResponse(BaseModel):
    question: str = ""
    answer: str = ""
    distractors: List[str] = []

@app.post("/generate-questions", response_model=List[QuestionGeneratorResponse])
def generate_questions(request: QuestionGeneratorRequest):
    return get_question_generator_response(request.text)

@app.post("/test", response_model=List[QuestionGeneratorResponse])
def generate_questions(request: QuestionGeneratorRequest):
    return get_question_generator_response(request.text)


