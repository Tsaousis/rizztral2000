import subprocess
import json

def run_curl(command):
    print(f"Running: {command}\n")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    response = result.stdout.strip()

    if response:
        print("Response:", response)
    if result.stderr.strip():
        print("Error:", result.stderr.strip())

    print("-" * 50)
    return response  # Return the response text

def test_endpoints():
    # 1. Health Check
    run_curl("curl -X GET http://localhost:8000/ -H 'accept: application/json'")

    # 2. Get AI Introduction
    run_curl("curl -X GET http://localhost:8000/ai-introduction -H 'accept: application/json'")

    # 3. Generate a New Question
    question_response = run_curl("curl -X GET http://localhost:8000/get-question -H 'accept: application/json'")

    # Parse the question from the JSON response
    try:
        question_data = json.loads(question_response) if question_response else {}
        question = question_data.get("question", "")
    except json.JSONDecodeError:
        print("Error: Failed to parse JSON response for the question.")
        question = ""

    if not question:
        print("Error: No question received, skipping further tests.")
        return

    # 4. Get AI Contestant Answers
    answer_response = run_curl(f'curl -G http://localhost:8000/get-ai-answers --data-urlencode "question={question}" --data-urlencode "contestant=1"')

    # Parse the answer from the JSON response
    try:
        answer_data = json.loads(answer_response) if answer_response else {}
        answer = answer_data.get("answer", "")
    except json.JSONDecodeError:
        print("Error: Failed to parse JSON response for the answer.")
        answer = ""

    if not answer:
        print("Error: No answer received, skipping rating tests.")
        return

    # 5. Rate an Answer (multiple cases)
    conv = f"Question: {question}\nContestant1: {answer}"
    rating_payloads = [
        {
            "conversation": conv,
            "round_number": 1
        },
        {
            "conversation": "What is your ideal date?\nContestant1: A McDonalds drive-thru. I'm joking, I'd love to go to a fancy restaurant!",
            "round_number": 1
        }
    ]

    for payload in rating_payloads:
        json_payload = json.dumps(payload)
        run_curl(f"curl -X POST http://localhost:8000/rate-answer -H 'Content-Type: application/json' -d '{json_payload}'")

if __name__ == "__main__":
    test_endpoints()
