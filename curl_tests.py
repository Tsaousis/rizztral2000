import requests
import json
from time import sleep

BASE_URL = "http://localhost:8000"

def print_response(description, response):
    print(f"\n{'-' * 50}")
    print(f"{description}:")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")
    print(f"{'-' * 50}\n")

def test_game_flow():
    """Test the complete game flow from start to finish"""
    
    print("\n=== Starting Dating Game Test ===\n")
    
    try:
        # 1. Reset the game
        reset_response = requests.get(f"{BASE_URL}/reset-game")
        print_response("Reset Game", reset_response)

        # 2. Get host introduction
        host_intro = requests.get(f"{BASE_URL}/host-introduction")
        print_response("Host Introduction", host_intro)

        # 3. Get AI introduction
        ai_intro = requests.get(f"{BASE_URL}/ai-introduction")
        print_response("AI Introduction", ai_intro)

        # 4. Get AI-generated questions
        print("\n=== Getting AI Questions ===")
        for i in range(3):  # Get 3 questions
            question = requests.get(f"{BASE_URL}/get-question")
            print_response(f"Generated Question {i+1}", question)

        # 5. Play each round
        for round_num in range(1, 4):
            print(f"\n=== Playing Round {round_num} ===")
            
            # Get the question for this round
            question = requests.get(f"{BASE_URL}/next-question")
            print_response(f"Question for Round {round_num}", question)
            
            # Submit user's answer (let it auto-generate)
            user_answer = requests.post(f"{BASE_URL}/submit-answer/contestant3")
            print_response(f"User Answer for Round {round_num}", user_answer)
            
            # Get AI contestants' answers
            ai_answers = requests.get(f"{BASE_URL}/get-ai-answers")
            print_response(f"AI Answers for Round {round_num}", ai_answers)
            
            # Rate all answers
            ratings = requests.get(f"{BASE_URL}/rate-all-answers")
            print_response(f"Ratings for Round {round_num}", ratings)
            
            # Always get next round (even after last round to transition to winner stage)
            next_round = requests.get(f"{BASE_URL}/next-round")
            print_response(f"Moving to Next Round/Stage", next_round)
            
            # If game is complete, announce winner
            if next_round.json().get("game_complete", False):
                winner = requests.get(f"{BASE_URL}/announce-winner")
                print_response("Winner Announcement", winner)
                break

        print("\nGame flow completed successfully! 🎉")
        
    except requests.exceptions.RequestException as e:
        print(f"Network Error: {e}")
    except Exception as e:
        print(f"Test Error: {e}")

def test_error_cases():
    """Test various error cases and invalid sequences"""
    
    print("\n=== Starting Error Case Tests ===\n")
    
    try:
        # Reset game first
        requests.get(f"{BASE_URL}/reset-game")
        
        # 1. Test skipping stages
        print("\nTesting stage skipping...")
        skip_test = requests.get(f"{BASE_URL}/announce-winner")
        print_response("Trying to skip to winner announcement", skip_test)
        
        # 2. Test invalid contestant ID
        print("\nTesting invalid contestant...")
        invalid_contestant = requests.post(
            f"{BASE_URL}/submit-answer/contestant4"
        )
        print_response("Submitting answer with invalid contestant", invalid_contestant)
        
        # 3. Test AI answering as user
        print("\nTesting AI answering as user...")
        ai_as_user = requests.post(
            f"{BASE_URL}/submit-answer/contestant1"
        )
        print_response("AI trying to submit answer", ai_as_user)
        
        # 4. Test getting next question without getting AI questions first
        print("\nTesting insufficient questions...")
        early_question = requests.get(f"{BASE_URL}/next-question")
        print_response("Getting question before generation", early_question)

        # 5. Test getting more than max questions
        print("\nTesting too many questions...")
        # First get 3 valid questions
        for _ in range(3):
            requests.get(f"{BASE_URL}/get-question")
        # Try to get one more
        extra_question = requests.get(f"{BASE_URL}/get-question")
        print_response("Getting extra question", extra_question)

        print("\nAll error cases tested successfully! 🎉")
        
    except requests.exceptions.RequestException as e:
        print(f"Network Error: {e}")
    except Exception as e:
        print(f"Test Error: {e}")

if __name__ == "__main__":
    print("\n=== Running Complete Game Flow Test ===")
    test_game_flow()
    
    print("\n=== Running Error Case Tests ===")
    test_error_cases()