"""
Test script to verify the /normalize endpoint is working correctly
"""
import requests
import json

ML_SERVICE_URL = "http://localhost:8001"

def test_normalize():
    """Test the /normalize endpoint with various inputs"""
    
    test_cases = [
        {
            "name": "Cow with fever and loss of appetite",
            "payload": {
                "animal": "cow",
                "symptom_text": "My cow has fever and loss of appetite"
            }
        },
        {
            "name": "No animal, auto-detect from text",
            "payload": {
                "animal": "",
                "symptom_text": "The goat seems depressed and has chills"
            }
        },
        {
            "name": "Sheep with multiple symptoms",
            "payload": {
                "animal": "sheep",
                "symptom_text": "swelling in limb and difficulty walking, also has sweats"
            }
        },
        {
            "name": "Free text with varied spelling",
            "payload": {
                "animal": "",
                "symptom_text": "animal: cow. symptoms: not eating, looks tired, shivering"
            }
        }
    ]
    
    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {test['name']}")
        print(f"{'='*60}")
        print(f"Payload: {json.dumps(test['payload'], indent=2)}")
        
        try:
            response = requests.post(
                f"{ML_SERVICE_URL}/normalize",
                json=test['payload'],
                timeout=10
            )
            print(f"\nStatus: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
        except requests.exceptions.ConnectionError:
            print(f"ERROR: Could not connect to {ML_SERVICE_URL}")
            print("Make sure the ML service is running with: python -m uvicorn ml_service:app --reload")
            return False
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
            return False
    
    print(f"\n{'='*60}")
    print("✓ All tests completed successfully!")
    print(f"{'='*60}")
    return True

if __name__ == "__main__":
    success = test_normalize()
    exit(0 if success else 1)
