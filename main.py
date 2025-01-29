from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel
from typing import Dict
import os
from langchain_mistralai import ChatMistralAI

app = FastAPI()

@app.get("/")
async def read_root():
    """Health check endpoint"""
    return {"status": "alive", "message": "Server is running"}

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=2
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ContestantAnswer(BaseModel):
    answer: str

# AI personality definitions
AI_PERSONALITIES = {
    "contestant1": "Confident and ambitious, with a dry sense of humor and passion for adventure",
    "contestant2": "Submissive, pathetic liar, with a low self-esteem and a passion for being a doormat. Also enjoys giving back handed compliments. Loves licking feet"
}

# Templates
ai_intro_template = PromptTemplate(
    input_variables=[],
    template="""You are a charming and witty AI bachelorette on a dating show.
Your personality traits:
- Confident
- Values authenticity and humor
- Loves being sexy
Introduce yourself to the contestants! ONLY ONE SENTENCE ANSWER"""
)

question_generator_template = PromptTemplate(
    input_variables=[],
    template="""You are a witty AI bachelorette host generating a question for your contestants.
The question should be:
- Flirty and playful
- Slightly humorous but not crude
- Revealing of personality
- Original (avoid cliché dating show questions)
- ONE SENTENCE only, ending with a question mark

Examples of the tone we want:
"If you were a pizza topping, which one would you be and why?"
"How would you handle a first date if we suddenly got trapped in an escape room?"

Generate a creative, funny dating show question. ONLY RETURN THE QUESTION."""
)

contestant_answer_template = PromptTemplate(
    input_variables=["question", "personality"],
    template="""You are a contestant on a dating show answering this question: {question}
Your personality type is: {personality}
Give a flirty but authentic answer, staying true to your character. Keep it under 3 sentences."""
)

rating_template = PromptTemplate(
    input_variables=["conversation", "round_number"],
    template="""Based on the following conversation in round {round_number}:
{conversation}
Rate the contestant's response from 0-10 based on compatibility, authenticity, and chemistry.
Only respond with a number from 0 to 10. NO explanations or extra words!"""
)

# Initialize chains
chains = {
    "ai_intro": LLMChain(llm=llm, prompt=ai_intro_template),
    "question_generator": LLMChain(llm=llm, prompt=question_generator_template),
    "contestant_answer": LLMChain(llm=llm, prompt=contestant_answer_template),
    "rating": LLMChain(llm=llm, prompt=rating_template)
}

@app.get("/ai-introduction")
async def get_ai_introduction():
    """Generate AI bachelorette's introduction"""
    try:
        print("Generating AI introduction...")
        response = await chains["ai_intro"].ainvoke({})
        print(f"Generated response: {response}")
        return {"text": response["text"]}
    except Exception as e:
        print(f"Error in get_ai_introduction: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating AI introduction: {str(e)}")

@app.get("/get-question")
async def get_question():
    """Generate a new question for the game"""
    try:
        llm.temperature = 0.9  # Higher temperature for more creative questions
        response = await chains["question_generator"].ainvoke({})
        llm.temperature = 0.7  # Reset temperature
        question = response["text"].strip('"')
        return {"question": question}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating question: {str(e)}")

@app.get("/get-ai-answers")
async def get_ai_answers(question: str):
    """Generate AI contestant responses to the current question"""
    try:
        llm.temperature = 0.8
        ai_answers = {}
        
        for contestant_id, personality in AI_PERSONALITIES.items():
            response = await chains["contestant_answer"].ainvoke({
                "question": question,
                "personality": personality
            })
            ai_answers[contestant_id] = response["text"]
        
        llm.temperature = 0.7
        return ai_answers
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating AI answers: {str(e)}")

class RatingRequest(BaseModel):
    conversation: str
    round_number: int

@app.post("/rate-answer")
async def rate_answer(request: RatingRequest):
    """Rate a single answer based on the conversation"""
    try:
        llm.temperature = 0.6
        response = await chains["rating"].ainvoke({
            "conversation": request.conversation,
            "round_number": request.round_number
        })
        rating = float(response["text"].strip())
        return {"rating": rating}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rating answer: {str(e)}")
