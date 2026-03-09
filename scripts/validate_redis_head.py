"""
Script to validate the current active Gemini API key (head of the Redis list).
Run from the repo root:
    python scripts/validate_redis_head.py
"""
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from upstash_redis import Redis as UpstashRedis
except ImportError:
    print("Error: upstash-redis is not installed. Run `pip install upstash-redis`.")
    sys.exit(1)

try:
    from google import genai
except ImportError:
    print("Error: google-genai is not installed.")
    sys.exit(1)


def main():
    load_dotenv(override=False)

    redis_url = os.getenv("UPSTASH_REDIS_REST_URL", "").strip()
    redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()
    list_key = os.getenv("GEMINI_KEYS_REDIS_LIST", "gemini_api_keys")

    if not redis_url or not redis_token:
        print("Error: UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN must be set.")
        sys.exit(1)

    redis = UpstashRedis(url=redis_url, token=redis_token)

    try:
        # 1. Get the current key at the head of the list
        key = redis.lindex(list_key, 0)
        
        if not key:
            print(f"Error: No keys found in the Redis list '{list_key}'.")
            sys.exit(1)
            
        key_str = key.decode() if isinstance(key, bytes) else str(key)
        masked_key = f"{key_str[:6]}...{key_str[-4:]}"
        print(f"Retrieved key at head of list: {masked_key}")
        
        # 2. Call the Gemini API with this key
        print("Initializing Gemini Client and sending a validation ping...")
        client = genai.Client(api_key=key_str)
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Reply with exactly the word PONG and nothing else."
        )
        
        reply = (response.text or "").strip()
        print(f"Verification successful! Received response: {reply}")
            
    except Exception as exc:
        print(f"Verification Failed! An error occurred: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
