from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from pydantic import BaseModel
from typing import List, Dict, Literal
from enum import Enum
from random import uniform
import os
import re

app = FastAPI()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="mixtral-8x7b-32768"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ContestantType(str, Enum):
    USER = "contestant3"  # User is always contestant3
    AI_ONE = "contestant1"
    AI_TWO = "contestant2"

AI_PERSONALITIES = {
    ContestantType.AI_ONE: "Confident and ambitious, with a dry sense of humor and passion for adventure",
    ContestantType.AI_TWO: "Sensitive and artistic, with a poetic soul and gentle demeanor"
}

class GameState:
    def __init__(self):
        print("\n[GAME STATE] Initializing new game...")
        self.current_round = 1
        self.max_rounds = 3
        self.contestant_ratings: Dict[str, List[float]] = {
            ContestantType.AI_ONE: [],
            ContestantType.AI_TWO: [],
            ContestantType.USER: []
        }
        self.conversation_history = []
        self.questions = []
        self.stage = "host_intro"
        print("[GAME STATE] Game initialized")

    def advance_stage(self):
        stages = [
            "host_intro",
            "ai_intro",
            "question_submission",
            "round_start",
            "answer_submission",
            "rating",
            "next_round",
            "winner_announcement",
            "game_complete"
        ]
        current_index = stages.index(self.stage)
        if current_index < len(stages) - 1:
            self.stage = stages[current_index + 1]
            print(f"[GAME STATE] Stage advanced to: {self.stage}")
        else:
            raise HTTPException(status_code=400, detail="Game is already complete!")

class QuestionInput(BaseModel):
    question: str

class ContestantAnswer(BaseModel):
    answer: str

class ConversationInput(BaseModel):
    conversation: str

game_state = GameState()

# Templates
host_intro_template = PromptTemplate(
    input_variables=[],
    template="You are a charismatic game show host like Steve Harvey. You don't talk a lot. Give an exciting introduction to this dating show called Rizztral where an AI bachelorette will choose between three contestants. ONLY ONE SENTENCE ANSWER"
)

ai_system_prompt = """You are a charming and witty AI bachelorette on a dating show.
Your personality traits:
- Confident but humble
- Values authenticity and humor
- Looking for genuine connection
Keep responses concise and engaging."""

ai_intro_template = PromptTemplate(
    input_variables=[],
    template=ai_system_prompt + "Introduce yourself to the contestants! ONLY ONE SENTENCE ANSWER"
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

winner_announcement_template = PromptTemplate(
    input_variables=["winner"],
    template="""You are a charismatic game show host announcing the winner. 
If the winner is 'contestant1', call them 'our adventurous bachelor'.
If the winner is 'contestant2', call them 'our poetic soul'.
If the winner is 'contestant3', call them 'our charming contestant'.
The winner is: {winner}
Give an exciting announcement. ONLY ONE SENTENCE ANSWER."""
)

chains = {
    "host_intro": LLMChain(llm=llm, prompt=host_intro_template),
    "ai_intro": LLMChain(llm=llm, prompt=ai_intro_template),
    "question_generator": LLMChain(llm=llm, prompt=question_generator_template),
    "contestant_answer": LLMChain(llm=llm, prompt=contestant_answer_template),
    "rating": LLMChain(llm=llm, prompt=rating_template),
    "winner": LLMChain(llm=llm, prompt=winner_announcement_template)
}

@app.get("/host-introduction")
async def get_host_introduction():
    if game_state.stage != "host_intro":
        raise HTTPException(status_code=400, detail="Not the correct stage for host introduction.")
    print("\n[HOST INTRO] Getting host introduction...")
    response = await chains["host_intro"].ainvoke({})
    game_state.advance_stage()
    return {"text": response["text"]}

@app.get("/ai-introduction")
async def get_ai_introduction():
    if game_state.stage != "ai_intro":
        raise HTTPException(status_code=400, detail="Not the correct stage for AI introduction.")
    print("\n[AI INTRO] Getting AI introduction...")
    response = await chains["ai_intro"].ainvoke({})
    game_state.advance_stage()
    return {"text": response["text"]}

@app.get("/get-question")
async def get_question():
    if game_state.stage != "question_submission":
        raise HTTPException(status_code=400, detail="Not the correct stage for getting a question.")
    
    # Use higher temperature for more creative questions
    llm.temperature = uniform(0.8, 0.95)
    response = await chains["question_generator"].ainvoke({})
    llm.temperature = 0.7  # Reset temperature
    
    question = response["text"].strip('"')  # Remove any quotes from the response
    game_state.questions.append(question)
    print(f"\n[QUESTION] Generated question: {question}")
    
    if len(game_state.questions) == game_state.max_rounds:
        game_state.advance_stage()
    
    return {
        "question": question,
        "round": len(game_state.questions),
        "total_rounds": game_state.max_rounds
    }

@app.get("/next-question")
async def get_next_question():
    if game_state.stage != "round_start":
        raise HTTPException(status_code=400, detail="Not the correct stage for starting a round.")
    if game_state.current_round > len(game_state.questions):
        raise HTTPException(status_code=400, detail="No more questions available.")
    question = game_state.questions[game_state.current_round - 1]
    game_state.advance_stage()  # Move to answer_submission stage
    print(f"\n[QUESTION] Returning question for round {game_state.current_round}: {question}")
    return {"text": question}

@app.post("/submit-answer/{contestant_id}")
async def submit_answer(contestant_id: ContestantType, answer: ContestantAnswer = None):
    if game_state.stage != "answer_submission":
        raise HTTPException(status_code=400, detail="Not the correct stage for answering")
    
    if contestant_id != ContestantType.USER:
        raise HTTPException(status_code=400, detail="Only user can submit answers here")
        
    current_question = game_state.questions[game_state.current_round - 1]
    
    # If no answer provided, generate a dummy response
    if answer is None:
        llm.temperature = uniform(0.7, 1.0)
        response = await chains["contestant_answer"].ainvoke({
            "question": current_question,
            "personality": "Friendly and outgoing, enjoys outdoor activities and meaningful conversations"
        })
        answer_text = response["text"]
        llm.temperature = 0.7  # Reset temperature
    else:
        answer_text = answer.answer
    
    game_state.conversation_history.append({
        "round": game_state.current_round,
        "contestant": contestant_id,
        "question": current_question,
        "answer": answer_text
    })
    
    return {
        "message": "Answer submitted successfully",
        "answer": answer_text,
        "was_auto_generated": answer is None
    }

@app.get("/get-ai-answers")
async def get_ai_answers():
    if game_state.stage != "answer_submission":
        raise HTTPException(status_code=400, detail="Not the correct stage for AI answers")
    
    current_question = game_state.questions[game_state.current_round - 1]
    ai_answers = {}
    
    for contestant_id in [ContestantType.AI_ONE, ContestantType.AI_TWO]:
        llm.temperature = uniform(0.7, 1.0)
        
        response = await chains["contestant_answer"].ainvoke({
            "question": current_question,
            "personality": AI_PERSONALITIES[contestant_id]
        })
        
        ai_answers[contestant_id] = response["text"]
        
        game_state.conversation_history.append({
            "round": game_state.current_round,
            "contestant": contestant_id,
            "question": current_question,
            "answer": response["text"]
        })
    
    llm.temperature = 0.7
    game_state.stage = "rating"
    return ai_answers

@app.get("/rate-all-answers")
async def rate_all_answers():
    if game_state.stage != "rating":
        raise HTTPException(status_code=400, detail="Not the correct stage for rating")
    
    current_round_convos = [
        conv for conv in game_state.conversation_history 
        if conv["round"] == game_state.current_round
    ]
    
    ratings = {}
    for conv in current_round_convos:
        conversation = f"Question: {conv['question']}\nAnswer: {conv['answer']}"
        
        llm.temperature = uniform(0.5, 0.8)
        
        response = await chains["rating"].ainvoke({
            "conversation": conversation,
            "round_number": game_state.current_round
        })
        
        rating = float(re.search(r'\d+(?:\.\d+)?', response["text"]).group())
        game_state.contestant_ratings[conv["contestant"]].append(rating)
        ratings[conv["contestant"]] = rating
    
    llm.temperature = 0.7
    game_state.stage = "next_round"
    
    return ratings

@app.get("/next-round")
async def next_round():
    if game_state.stage != "next_round":
        raise HTTPException(status_code=400, detail="Not the correct stage for next round.")
    
    game_state.current_round += 1
    print(f"[NEXT ROUND] Current round is now {game_state.current_round}")
    
    if game_state.current_round > game_state.max_rounds:
        game_state.stage = "winner_announcement"
        print("[NEXT ROUND] Final round completed, moving to winner announcement")
        return {"current_round": game_state.current_round, "game_complete": True}
    
    game_state.stage = "round_start"
    return {"current_round": game_state.current_round, "game_complete": False}

@app.get("/announce-winner")
async def announce_winner():
    print("\n[WINNER ANNOUNCEMENT] Starting winner announcement process...")
    
    if game_state.stage != "winner_announcement":
        print(f"[WINNER ANNOUNCEMENT] Error: Invalid game stage {game_state.stage}")
        raise HTTPException(
            status_code=400, 
            detail=f"Not the correct stage for announcing winner. Current stage: {game_state.stage}"
        )
    
    print("[WINNER ANNOUNCEMENT] Calculating average ratings for all contestants...")
    print(f"[WINNER ANNOUNCEMENT] Raw ratings: {game_state.contestant_ratings}")
    
    # Add error handling for empty ratings
    for contestant, ratings in game_state.contestant_ratings.items():
        if not ratings:
            print(f"[WINNER ANNOUNCEMENT] Warning: No ratings found for {contestant}")
            raise HTTPException(
                status_code=400, 
                detail=f"Missing ratings for contestant: {contestant}"
            )
    
    avg_ratings = {
        contestant: sum(ratings)/len(ratings) 
        for contestant, ratings in game_state.contestant_ratings.items()
    }
    
    print(f"[WINNER ANNOUNCEMENT] Calculated average ratings: {avg_ratings}")
    
    winner = max(avg_ratings.items(), key=lambda x: x[1])[0]
    print(f"[WINNER ANNOUNCEMENT] Winner determined: {winner} with average rating {avg_ratings[winner]}")
    
    print("[WINNER ANNOUNCEMENT] Generating winner announcement message...")
    response = await chains["winner"].ainvoke({"winner": winner})
    print(f"[WINNER ANNOUNCEMENT] Generated announcement: {response['text']}")
    
    game_state.stage = "game_complete"
    print("[WINNER ANNOUNCEMENT] Game stage updated to: game_complete")
    
    return {
        "text": response["text"], 
        "winner": winner,
        "final_ratings": avg_ratings
    }

@app.get("/reset-game")
async def reset_game():
    global game_state
    game_state = GameState()
    return {"message": "Game reset successfully"}