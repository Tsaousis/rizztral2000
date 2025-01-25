# test_api.py
import requests
import json

BASE_URL = "http://localhost:8000"

def print_response(description, response):
    print(f"\n{description}:")
    print(json.dumps(response.json(), indent=2))

def test_dating_game():
    requests.get(f"{BASE_URL}/reset-game")

    print_response("Host Introduction", 
                  requests.get(f"{BASE_URL}/host-introduction"))

    print_response("AI Introduction",
                  requests.get(f"{BASE_URL}/ai-introduction"))

    conversations = {
        1: [
            "I love Mcdonalds and I am really stingy.",
            "I love KFC and I am a very insecure person.",
            "I'm a balanced person who enjoys both adventure and quiet time. I value emotional intelligence and can adapt to different situations."
        ],
        2: [
            "Family is everything to me. I come from a big family and hope to have one of my own. I believe in traditional values while being open-minded.",
            "I'm career-focused but know how to maintain work-life balance. I believe in supporting each other's dreams while building something together.",
            "I'm passionate about personal growth and helping others. I volunteer at local shelters and believe in making the world better together."
        ],
        3: [
            "My ideal date would be cooking together, then watching the sunset. I believe small moments create lasting memories.",
            "I love spontaneous trips and surprises. Life's too short to plan everything - sometimes you need to just go with the flow.",
            "I believe in open communication and emotional honesty. Trust and understanding are the foundations of any relationship."
        ]
    }

    for round in range(1, 4):
        print(f"\n=== Round {round} ===")
        
        print_response("AI Question",
                      requests.get(f"{BASE_URL}/ai-question"))

        for contestant in range(1, 4):
            print_response(f"Rate Contestant {contestant}",
                          requests.post(
                              f"{BASE_URL}/rate-contestant/contestant{contestant}",
                              json={"conversation": conversations[round][contestant-1]}
                          ))
            
            if contestant < 3:
                print_response("Host Interrupt",
                             requests.get(f"{BASE_URL}/host-interrupt/next_contestant"))
        
        if round < 3:
            requests.get(f"{BASE_URL}/next-round")

    print_response("Announce Winner",
                  requests.get(f"{BASE_URL}/announce-winner"))

if __name__ == "__main__":
    test_dating_game()