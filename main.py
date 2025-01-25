from fastapi import FastAPI, HTTPException
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from pydantic import BaseModel
from typing import List, Dict
import os

app = FastAPI()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="mixtral-8x7b-32768"
)

class GameState:
    def __init__(self):
        self.current_round = 1
        self.contestant_ratings: Dict[str, List[float]] = {
            "contestant1": [], 
            "contestant2": [],
            "contestant3": []
        }
        self.conversation_history = []

game_state = GameState()

host_intro_template = PromptTemplate(
    input_variables=[],
    template="You are a charismatic game show host. Give an exciting introduction to this dating show where an AI bachelor/bachelorette will choose between three contestants."
)

ai_intro_template = PromptTemplate(
    input_variables=[],
    template="You are an AI bachelor/bachelorette on a dating show. Introduce yourself with your personality traits and what you're looking for in a partner."
)

question_template = PromptTemplate(
    input_variables=["round_number"],
    template="As the AI bachelor/bachelorette on round {round_number} of 3, pose an interesting dating show question to help you know the contestants better."
)

rating_template = PromptTemplate(
    input_variables=["conversation", "round_number"],
    template="""Based on the following conversation in round {round_number}:
{conversation}
Rate the contestant's response from 0-10 based on compatibility, authenticity, and chemistry.
Only respond with a number from 0 to 10. NO explanations or extra words!"""
)

host_interrupt_template = PromptTemplate(
    input_variables=["next_segment", "round_number"],
    template="As the game show host in round {round_number}, create a smooth transition to {next_segment} while maintaining show excitement."
)

winner_announcement_template = PromptTemplate(
    input_variables=["winner"],
    template="As the game show host, announce that {winner} has won the dating show with excitement and flair!"
)

chains = {
    "host_intro": LLMChain(llm=llm, prompt=host_intro_template),
    "ai_intro": LLMChain(llm=llm, prompt=ai_intro_template),
    "question": LLMChain(llm=llm, prompt=question_template),
    "rating": LLMChain(llm=llm, prompt=rating_template),
    "host_interrupt": LLMChain(llm=llm, prompt=host_interrupt_template),
    "winner": LLMChain(llm=llm, prompt=winner_announcement_template)
}

@app.get("/host-introduction")
async def get_host_introduction():
    response = await chains["host_intro"].ainvoke({})
    return {"text": response["text"]}

@app.get("/ai-introduction")
async def get_ai_introduction():
    response = await chains["ai_intro"].ainvoke({})
    return {"text": response["text"]}

@app.get("/ai-question")
async def get_ai_question():
    response = await chains["question"].ainvoke({"round_number": game_state.current_round})
    return {"text": response["text"]}

class ConversationInput(BaseModel):
    conversation: str

@app.post("/rate-contestant/{contestant_id}")
async def rate_contestant(contestant_id: str, conversation_input: ConversationInput):
    if contestant_id not in ["contestant1", "contestant2", "contestant3"]:
        raise HTTPException(status_code=400, detail="Invalid contestant ID")
    
    import re
    response = await chains["rating"].ainvoke({
        "conversation": conversation_input.conversation,
        "round_number": game_state.current_round
    })
    rating_match = re.search(r'\d+(?:\.\d+)?', response["text"])
    if not rating_match:
        raise HTTPException(status_code=500, detail="Could not extract rating from response")
    rating = float(rating_match.group())
    game_state.contestant_ratings[contestant_id].append(rating)
    return {"rating": rating}

@app.get("/host-interrupt/{next_segment}")
async def get_host_interrupt(next_segment: str):
    response = await chains["host_interrupt"].ainvoke({
        "next_segment": next_segment,
        "round_number": game_state.current_round
    })
    return {"text": response["text"]}

@app.get("/next-round")
async def next_round():
    game_state.current_round += 1
    return {"current_round": game_state.current_round}

@app.get("/announce-winner")
async def announce_winner():
    avg_ratings = {
        contestant: sum(ratings)/len(ratings) 
        for contestant, ratings in game_state.contestant_ratings.items()
    }
    winner = max(avg_ratings.items(), key=lambda x: x[1])[0]
    response = await chains["winner"].ainvoke({"winner": winner})
    return {"text": response["text"], "winner": winner}

@app.get("/reset-game")
async def reset_game():
    global game_state
    game_state = GameState()
    return {"message": "Game reset successfully"}

@app.get("/conversation/{contestant_id}")
async def get_conversation(contestant_id: str):
    return {"text": f"Simulated conversation with {contestant_id}"}