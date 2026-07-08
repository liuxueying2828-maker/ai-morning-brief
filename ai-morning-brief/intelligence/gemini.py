import os

from intelligence.prompt import build_executive_brief_prompt


GEMINI_MODEL = "gemini-2.5-flash"


def generate_intelligence_brief(items):
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "Missing google-genai package. Run: python3 -m pip install -r requirements.txt"
        ) from exc

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable.")

    prompt = build_executive_brief_prompt(items)
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    return response.text.strip()
