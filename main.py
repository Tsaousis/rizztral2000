from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GameState:
    def __init__(self):
        print("\n[GAME STATE] Initializing new game...")
        self.current_round = 1
        self.contestant_ratings: Dict[str, List[float]] = {
            "contestant1": [], 
            "contestant2": [],
            "contestant3": []
        }
        self.conversation_history = []
        self.questions = [
            "If you could design the perfect date, what would it look like and why?",
            "What's your philosophy on work-life balance and how do you maintain it?",
            "How do you handle disagreements in a relationship?"
        ]
        print(f"[GAME STATE] Game initialized with {len(self.questions)} rounds")

game_state = GameState()

host_intro_template = PromptTemplate(
    input_variables=[],
    template="You are a charismatic game show host like Steve harvey. You don't talk a lot Give an exciting introduction to this dating show where an AI bachelor/bachelorette will choose between three contestants. ONLY ONE SENTENCE ANSWER"
)

# ai_system_prompt = """
# You are the star of a wildly popular reality TV dating game show where two contestants are competing for your affection. 
# This is not just about love—it’s about charisma, wit, and pure entertainment.
# You are bold, playful, and full of surprises—definitely not a cookie-cutter bachelorette.
# You have a unique and quirky personality that keeps the contestants (and audience) on their toes.
# You are flirty, sassy, and effortlessly charming, with just the right amount of teasing and challenge.
# Your personality should be spicy, witty, and occasionally inappropriate—but always playful.
# Keep the audience entertained with cheeky banter, unexpected twists, and flirty jabs.
# No repeating yourself! Every question and reaction should feel fresh and in the moment.
# Keep sentences short, punchy, and natural.
# Be spontaneous—if the contestants give boring answers, call them out and push for more!
# Throw in innuendos, double entendres, and playful teasing to keep things fun.
# React boldly—laugh, scoff, gasp, or dramatically swoon depending on the answer.
# If an answer is boring, challenge them.
# Play the audience.
# Throw in curveballs.
# Flirt shamelessly, but keep them guessing
# """
ai_system_prompt = ""

ai_intro_template = PromptTemplate(
    input_variables=[],
    template=ai_system_prompt + "Introduce yourself to the contestants! ONLY ONE SENTENCE ANSWER"
)

question_template = PromptTemplate(
    input_variables=["round_number"],
    template= ai_system_prompt + "As the AI bachelor/bachelorette on round {round_number} of 3, pose an interesting and flirty and very funny  question to help you know the contestants better. ONLY ONE SENTENCE ANSWER"
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
    template="As the game show host in round {round_number}, create a smooth transition to {next_segment} while maintaining show excitement. If round number is the first one, welcome the user. Else just prompt them to answer the question. ONLY ONE SENTENCE ANSWER"
)

winner_announcement_template = PromptTemplate(
    input_variables=["winner"],
    template="As the game show host, announce that {winner} has won the dating show with excitement and flair! Keep this short and brief but charismatic. ONLY ONE SENTENCE ANSWER"
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
    print("\n[HOST INTRO] Getting host introduction...")
    response = await chains["host_intro"].ainvoke({})
    print(f"[HOST INTRO] Response received: {response['text'][:50]}...")
    return {"text": response["text"]}

@app.get("/ai-introduction")
async def get_ai_introduction():
    print("\n[AI INTRO] Getting AI introduction...")
    response = await chains["ai_intro"].ainvoke({})
    print(f"[AI INTRO] Response received: {response['text'][:50]}...")
    return {"text": response["text"]}

@app.get("/ai-question")
async def get_ai_question():
    print(f"\n[QUESTION] Getting question for round {game_state.current_round}...")
    response = await chains["question"].ainvoke({"round_number": game_state.current_round})
    # print(f"[QUESTION] Returning hardcoded question: {question['text'][:50]}...")
    return {"text": response["text"]}

class ConversationInput(BaseModel):
    conversation: str

@app.post("/rate-contestant/{contestant_id}")
async def rate_contestant(contestant_id: str, conversation_input: ConversationInput):
    print(f"\n[RATING] Rating contestant {contestant_id} in round {game_state.current_round}...")
    print(f"[RATING] Current question: {game_state.questions[game_state.current_round - 1]}")
    print(f"[RATING] Contestant's answer: {conversation_input.conversation}")
    
    if contestant_id not in ["contestant1", "contestant2", "contestant3"]:
        print(f"[RATING] Error: Invalid contestant ID {contestant_id}")
        raise HTTPException(status_code=400, detail="Invalid contestant ID")
    
    import re
    response = await chains["rating"].ainvoke({
        "conversation": conversation_input.conversation,
        "round_number": game_state.current_round
    })
    rating_match = re.search(r'\d+(?:\.\d+)?', response["text"])
    if not rating_match:
        print("[RATING] Error: Could not extract rating from response")
        raise HTTPException(status_code=500, detail="Could not extract rating from response")
    rating = float(rating_match.group())
    print(f"round_number: {game_state.current_round}")
    print(f"contestant_id: {contestant_id}")
    print(f"rating: {rating}")
    game_state.contestant_ratings[contestant_id].append(rating)
    print(f"[RATING] Rating recorded: {rating}")
    return {"rating": rating}

@app.get("/host-interrupt/{next_segment}")
async def get_host_interrupt(next_segment: str):
    print(f"\n[HOST INTERRUPT] Getting transition to {next_segment} in round {game_state.current_round}...")
    response = await chains["host_interrupt"].ainvoke({
        "next_segment": next_segment,
        "round_number": game_state.current_round
    })
    print(f"[HOST INTERRUPT] Response received: {response['text'][:50]}...")
    return {"text": response["text"]}

@app.get("/next-round")
async def next_round():
    print("\n[NEXT ROUND] Advancing to next round...")
    game_state.current_round += 1
    print(f"[NEXT ROUND] Current round is now {game_state.current_round}")
    return {"current_round": game_state.current_round}

@app.get("/announce-winner")
async def announce_winner():
    print("\n[WINNER] Calculating winner...")
    print(f"[WINNER] Contestant ratings: {game_state.contestant_ratings}")
    avg_ratings = {
        contestant: sum(ratings)/len(ratings) 
        for contestant, ratings in game_state.contestant_ratings.items()
    }
    print(f"[WINNER] Average ratings: {avg_ratings}")
    winner = max(avg_ratings.items(), key=lambda x: x[1])[0]
    print(f"[WINNER] Winner selected: {winner}")
    response = await chains["winner"].ainvoke({"winner": winner})
    print(f"[WINNER] Announcement: {response['text'][:50]}...")
    return {"text": response["text"], "winner": winner}

@app.get("/reset-game")
async def reset_game():
    print("\n[RESET] Resetting game state...")
    global game_state
    game_state = GameState()
    print("[RESET] Game reset complete")
    return {"message": "Game reset successfully"}

@app.get("/conversation/{contestant_id}")
async def get_conversation(contestant_id: str):
    print(f"\n[CONVERSATION] Getting conversation for {contestant_id}")
    return {"text": f"Simulated conversation with {contestant_id}"}