"""
llm_engine.py
-------------
Handles all communication with the local Ollama LLM.
"""

import ollama


DEFAULT_MODEL = "llama3"


def is_ollama_available() -> bool:
    """Check if the Ollama server is reachable."""
    try:
        ollama.list()
        return True
    except Exception:
        return False


def list_models() -> list:
    """Return a list of locally available Ollama model names."""
    try:
        response = ollama.list()
        print(f"[DEBUG] type={type(response)}")
        print(f"[DEBUG] raw={response}")

        if hasattr(response, "models"):
            models = response.models
        elif isinstance(response, dict):
            models = response.get("models", [])
        else:
            models = []

        names = []
        for m in models:
            name = (
                getattr(m, "model", None)
                or getattr(m, "name", None)
                or (m.get("model") if isinstance(m, dict) else None)
                or (m.get("name") if isinstance(m, dict) else None)
                or str(m)
            )
            names.append(name)
        return names
    except Exception as e:
        print(f"[list_models error] {type(e).__name__}: {e}")
        return []


def ask_llm(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send a prompt to the LLM and return the text response."""
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
    except Exception as e:
        return f"[LLM ERROR] Could not get response: {e}"


if __name__ == "__main__":
    print("Checking Ollama...")
    if not is_ollama_available():
        print("Ollama is NOT running. Start it with: ollama serve")
    else:
        print("Ollama is running.")
        print("Available models:", list_models())
        print("\nTest prompt:")
        print(ask_llm("Say hello in one short sentence."))