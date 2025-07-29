#!/usr/bin/env python3
"""
Test script to verify LangSmith integration is working correctly.
"""

import os
from dotenv import load_dotenv
from app.core import get_response

# Load environment variables
load_dotenv()

def test_langsmith_integration():
    """Test that LangSmith tracing is working."""
    
    # Check if LangSmith environment variables are set
    required_vars = [
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_ENDPOINT", 
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_PROJECT"
    ]
    
    print("🔍 Checking LangSmith Environment Variables:")
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {'*' * len(value)} (set)")
        else:
            print(f"  ❌ {var}: Not set")
    
    print("\n🧪 Testing LangSmith Integration:")
    
    # Test a simple query
    test_query = "Apa itu Program Studi Teknik Informatika?"
    test_user_id = "test_user_123"
    
    print(f"  📝 Test Query: {test_query}")
    print(f"  👤 Test User ID: {test_user_id}")
    
    try:
        # This should trigger LangSmith tracing
        response = get_response(
            query=test_query,
            user_id=test_user_id,
            conversation_has_started=True,
            is_initial_greeting_sent=True
        )
        
        print(f"  ✅ Response received: {len(response)} characters")
        print(f"  📄 Response preview: {response[:100]}...")
        
        print("\n🎉 LangSmith integration test completed successfully!")
        print("📊 Check your LangSmith dashboard to see the traces.")
        
    except Exception as e:
        print(f"  ❌ Error during test: {e}")
        print("\n🔧 Troubleshooting:")
        print("  1. Make sure all environment variables are set")
        print("  2. Verify your LangSmith API key is valid")
        print("  3. Check that langsmith package is installed")
        print("  4. Ensure your project has proper permissions")

if __name__ == "__main__":
    test_langsmith_integration() 