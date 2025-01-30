import subprocess
import json

def run_curl(command):
    print(f"Running: {command}\n")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        print("Response:", result.stdout.strip())
    if result.stderr.strip():
        print("Error:", result.stderr.strip())
    print("-" * 50)

def test_endpoints():
    # 1. Health Check
    run_curl("curl -X GET http://localhost:8000/ -H 'accept: application/json'")

    # 2. Get AI Introduction
    run_curl("curl -X GET http://localhost:8000/ai-introduction -H 'accept: application/json'")

    # 3. Generate a New Question
    run_curl("curl -X GET http://localhost:8000/get-question -H 'accept: application/json'")

    # 4. Get AI Contestant Answers
    question = "If you were a pizza topping, which one would you be and why?"
    run_curl(f"curl -G http://localhost:8000/get-ai-answers --data-urlencode 'question={question}'")

    # 5. Rate an Answer (multiple cases)
    rating_payloads = [
        {
            "conversation": "What is your ideal date?\nContestant1: A long walk on the beach with deep conversations!",
            "round_number": 1
        },
        {
            "conversation": "What is your ideal date?\nContestant1: A McDonalds drive-thru",
            "round_number": 1
        }
    ]

    for payload in rating_payloads:
        json_payload = json.dumps(payload)
        run_curl(f"curl -X POST http://localhost:8000/rate-answer -H 'Content-Type: application/json' -d '{json_payload}'")

if __name__ == "__main__":
    test_endpoints()
