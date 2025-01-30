from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel
import os
from langchain_mistralai import ChatMistralAI
import nest_asyncio

nest_asyncio.apply()

app = FastAPI()

@app.get("/")
async def read_root():
    """Health check endpoint"""
    return {"status": "alive", "message": "Server is running"}

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
llm_host_and_bachelorette = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=2
)

llm_contestant1 = ChatMistralAI(
    model="ministral-3b-latest",
    temperature=0,
    max_retries=2
)
llm_contestant2 = ChatMistralAI(
    model="ministral-8b-latest",
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
    input_variables=["question"],
    template="""You are a contestant on a dating show answering this question: {question}
Give a flirty but authentic answer, staying true to your character. Keep it under 3 sentences."""
)

rating_template = PromptTemplate(
    input_variables=["conversation", "round_number"],
    template="""You are the Bachelorrete for a dating show, rating a contestant's response to your question. You are a hard-to-please AI with high standards and a judgmental streak.
{conversation}
Rate the contestant's response from 0-10 based on compatibility, authenticity, and chemistry, based on the context of the conversation and trying to find the best match for you.
Only respond with a number from 0 to 10. NO explanations or extra words!"""
)

# Initialize chains with correct LLMs
chains = {
    "ai_intro": LLMChain(llm=llm_host_and_bachelorette, prompt=ai_intro_template),
    "question_generator": LLMChain(llm=llm_host_and_bachelorette, prompt=question_generator_template),
    "contestant_answer_1": LLMChain(llm=llm_contestant1, prompt=contestant_answer_template),
    "contestant_answer_2": LLMChain(llm=llm_contestant2, prompt=contestant_answer_template),
    "rating": LLMChain(llm=llm_host_and_bachelorette, prompt=rating_template)
}

@app.get("/ai-introduction")
async def get_ai_introduction():
    """Generate AI bachelorette's introduction"""
    try:
        response = await chains["ai_intro"].ainvoke({})
        return {"text": response["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating AI introduction: {str(e)}")

@app.get("/get-question")
async def get_question():
    """Generate a new question for the game"""
    try:
        llm_host_and_bachelorette.temperature = 0.9  # Higher temperature for more creative questions
        response = await chains["question_generator"].ainvoke({})
        llm_host_and_bachelorette.temperature = 0.7  # Reset temperature
        question = response["text"].strip('"')
        return {"question": question}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating question: {str(e)}")

@app.get("/get-ai-answers")
async def get_ai_answers(question: str, contestant: int):
    """Generate AI contestant responses to the current question"""
    try:
        llm_host_and_bachelorette.temperature = 0.8
        chain_name = f"contestant_answer_{contestant}"
        response = await chains[chain_name].ainvoke({"question": question})
        llm_host_and_bachelorette.temperature = 0.7
        return {"answer": response["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating AI answers: {str(e)}")

class RatingRequest(BaseModel):
    conversation: str
    round_number: int

@app.post("/rate-answer")
async def rate_answer(request: RatingRequest):
    """Rate a single answer based on the conversation"""
    try:
        llm_host_and_bachelorette.temperature = 0.6
        response = await chains["rating"].ainvoke({
            "conversation": request.conversation,
            "round_number": request.round_number
        })
        rating = float(response["text"].strip())
        return {"rating": rating}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rating answer: {str(e)}")
