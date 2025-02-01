import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
import nest_asyncio
load_dotenv()

questions = [
    "Steak or sushi",
    "If you had one hour to spend with a person who is gone - famous or not - who would you spend that hour with, and why?",
    "Would you rather be blind or deaf?",
    "What is your greatest character strength?",
    "What one thing in your life are you trying to improve?",
    "Choose one cartoon character to run the world - who and why.",
    "What’s one thing you absolutely cannot do, as in you’ve tried and are just terrible at it.",
    "You’re stuck on a deserted island for ten years, what 3 foods would you have an endless supply of?",
    "If you had a million dollars to give away, what would that look like? This is a revealing question and always brings interesting answers.",
    "Do you consider yourself a good person? (So this is an intentional question on my part. When a guy quickly responds with a 'yes', that gives me pause. I’m looking for the type of person who says something more along the lines of 'Not always, but I TRY…')",
    "What would your TEDTalk be about?",
    "Your Mom called and told you not to forget _____ when you come over for dinner…fill in that blank!",
    "What’s a skill you’ve always wanted to learn, what’s kept you from doing it?",
    "A movie is made about your life. Which actor plays your part?",
    "What’s the most random fact you know?",
    "Over or under for toilet paper?",
    "Give me an adjective that describes you, for every letter of your first name.",
    "If you could switch lives with one person for a day, who would it be and why?",
    "What’s the craziest adventure you’ve ever been on?",
    "If you could time travel, where and when would you go?",
    "What’s one thing you’ve always wanted to do but never have?",
    "If you could instantly master any skill, what would it be?",
    "What’s your guilty pleasure?",
    "If you had to eat one meal every day for the rest of your life, what would it be?",
    "What’s the best advice you’ve ever received?",
    "What would you do if you won the lottery tomorrow?",
    "What’s your biggest pet peeve?",
    "If you could live anywhere in the world, where would it be and why?",
    "If you could have any superpower, what would it be and how would you use it?",
    "What’s something you’ve always wanted to try but haven’t yet?",
    "What’s the most adventurous thing you’ve done in the last year?",
    "If you could be famous for something, what would it be?",
    "What’s the most interesting place you’ve visited?",
    "What’s the weirdest dream you’ve ever had?",
    "What’s your idea of the perfect date night?",
    "How do you like to be kissed?",
    "What’s the sexiest thing someone could say to you?",
    "Do you like to be in control or let someone else take the lead?",
    "What’s your secret fantasy?",
    "How do you feel about slow dancing, even when there’s no music?",
    "What’s one thing you would never do on a first date?",
    "What’s your idea of an unforgettable kiss?",
    "If I whispered something in your ear right now, would you blush?",
    "What kind of touch drives you crazy?",
    "If I were to surprise you with a romantic getaway, where would we go?"
]



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
- Revealing of personality
- Original (avoid cliché dating show questions)
- ONE SENTENCE only, ending with a question mark

You prefer innovative, spicy/sexy and humorous questions.

Examples of the tone we want:
{questions}


Generate a creative, funny dating show question. ONLY RETURN THE QUESTION."""
)

contestant_answer_template = PromptTemplate(
    input_variables=["question"],
    template="""You are a man-contestant on a dating show answering this question form the bachelorette: {question}.
    You need to impress the bachelorette with your answer, showing off your personality and sense of humor. She is sexy, confident, and values authenticity. She prefers innovative, spicy and humorous answers.
Give a flirty but authentic answer, staying true to your character.  Keep it under 3 sentences."""
)

rating_template = PromptTemplate(
    input_variables=["conversation", "round_number"],
    template="""You are the Bachelorrete for a dating show, rating a contestant's response to your question. You are a woman with high standards and you prefer innovative, spicy and humorous answers.
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
        # Pass the questions to the prompt dynamically
        response = await chains["question_generator"].ainvoke({"questions": random.sample(questions, 3)})
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
