import requests

API_KEY = "AIzaSyDa2W0leYWylnmNXkygbmjhOTiKm8a3YRM"

print("Step 1 — Listing available models on your API key...")
print("="*60)

# List all models
r = requests.get(
    "https://generativelanguage.googleapis.com/v1/models",
    params={"key": API_KEY},
    timeout=15
)

if r.status_code == 200:
    models = r.json().get("models", [])
    print(f"Found {len(models)} models:\n")
    generateContent_models = []
    for m in models:
        name    = m.get("name", "")
        methods = m.get("supportedGenerationMethods", [])
        if "generateContent" in methods:
            generateContent_models.append(name)
            print(f"  ✅ {name}  — supports generateContent")
        else:
            print(f"  ❌ {name}  — does NOT support generateContent")

    print("\n" + "="*60)
    print("Step 2 — Testing generateContent on available models...")
    print("="*60)

    for full_name in generateContent_models:
        # full_name is like "models/gemini-2.0-flash"
        model_id = full_name.replace("models/", "")
        url = f"https://generativelanguage.googleapis.com/v1/models/{model_id}:generateContent"
        test = requests.post(
            url,
            params={"key": API_KEY},
            json={"contents": [{"parts": [{"text": "Say hello in 3 words."}]}]},
            timeout=15
        )
        if test.status_code == 200:
            reply = test.json()["candidates"][0]["content"]["parts"][0]["text"]
            print(f"\n✅ WORKING MODEL: {model_id}")
            print(f"   Response: {reply.strip()}")
            print(f"\n👉 Use this in your app.py:")
            print(f'   url = "https://generativelanguage.googleapis.com/v1/models/{model_id}:generateContent"')
            break
        else:
            print(f"❌ {model_id}: {test.status_code} — {test.text[:100]}")

elif r.status_code == 400:
    print("❌ API key is invalid. Please regenerate at aistudio.google.com")
elif r.status_code == 403:
    print("❌ API key does not have Gemini API enabled.")
    print("\nFix:")
    print("1. Go to https://aistudio.google.com")
    print("2. Click 'Get API Key'")
    print("3. Create a NEW API key")
    print("4. Replace the key in airsense_app.py")
else:
    print(f"❌ Error {r.status_code}: {r.text[:300]}")
