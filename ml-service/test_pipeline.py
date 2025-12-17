"""
Test script for the advanced pipeline operations
Run this after starting the FastAPI server
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_pipeline_operations():
    """Test the run-pipeline endpoint"""
    
    # Test payload
    payload = {
        "file_path": "backend/uploads/1765435818111-spotify_2015_2025_85k.csv",
        "operation_type": "split",
        "test_size": 0.2,
        "n_folds": 5,
        "shuffle": True,
        "random_state": 42,
        "cleaning_pipeline": {
            "steps": []
        }
    }
    
    print("Testing run-pipeline endpoint...")
    print(f"Operation: {payload['operation_type']}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(f"{BASE_URL}/run-pipeline", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ SUCCESS!")
            print(f"Output files: {len(result.get('output_files', []))}")
            for file in result.get('output_files', []):
                print(f"  - {file['name']}: {file['rows']} rows")
            print(f"\nTransformation log:")
            for log in result.get('transformation_log', []):
                print(f"  {log}")
        else:
            print(f"\n❌ ERROR: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"\n❌ EXCEPTION: {str(e)}")

if __name__ == "__main__":
    print("=" * 60)
    print("Advanced Pipeline Operations Test")
    print("=" * 60)
    print("\nMake sure:")
    print("1. FastAPI server is running (python main.py)")
    print("2. A CSV file exists in backend/uploads/")
    print("\nStarting tests...\n")
    
    test_pipeline_operations()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
