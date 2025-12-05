# assistant.py
"""
Features:
 - Listens on the microphone and speaks back using pyttsx3.
 - Safely evaluates arithmetic expressions (no eval()).
 - Fetches short definitions from Wikipedia, with a DuckDuckGo fallback.
 - Opens Google/YouTube on request.
 - Uses polite confirmations, retries, and clear logs so it feels more "human".
"""

import speech_recognition as sr
import pyttsx3
import wikipedia
import requests
import webbrowser
import ast
import operator as op
import sys
import time

# -------------------------
# Configuration (tweakable)
# -------------------------
WAKE_WORD = "assistant"               # Say this before your command
LISTEN_TIMEOUT = 5                    # Seconds to wait for speech to begin
PHRASE_TIME_LIMIT = 8                 # Max seconds to listen per phrase
WIKI_SENTENCES = 2                    # Sentences when returning Wikipedia summary
MAX_RETRIES = 2                       # How many times to ask again if not understood

# -------------------------
# TTS setup (friendly voice)
# -------------------------
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 160)

def speak(text: str, say_aloud: bool = True):
    """
    Central speak/log helper.
    Prints the assistant text to console and optionally speaks it aloud.
    """
    # Normalize whitespace and trim
    text = " ".join(str(text).split())
    print(f"[Assistant] {text}")
    if say_aloud:
        tts_engine.say(text)
        tts_engine.runAndWait()

# -------------------------
# Speech recognizer setup
# -------------------------
recognizer = sr.Recognizer()

def listen_from_mic(timeout=LISTEN_TIMEOUT, phrase_time_limit=PHRASE_TIME_LIMIT):
    """
    Listen once from the default microphone and return recognized text (lowercased),
    or None if nothing understood.
    This function handles ambient-noise adjustment and timeouts.
    """
    with sr.Microphone() as source:
        # Calibrate to ambient noise for a moment so we don't hear room hiss as speech
        recognizer.adjust_for_ambient_noise(source, duration=0.6)
        print("[Listening] Ready — please speak now...")
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            print("[Listening] No speech detected (timeout).")
            return None

    try:
        # Use Google's free speech recognition (requires internet)
        text = recognizer.recognize_google(audio)
        print(f"[You] {text}")
        return text.lower()
    except sr.UnknownValueError:
        print("[Listening] Sorry — I couldn't understand that.")
        return None
    except sr.RequestError as e:
        # Network / API error
        print(f"[Listening] Speech service error: {e}")
        speak("I am having trouble reaching the speech service. Please check your internet connection.")
        return None

# -------------------------
# Safe arithmetic evaluator
# -------------------------
ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

def safe_eval(expr: str):
    """
    Evaluate a numeric expression safely. Allowed: numbers, + - * / % ** and parentheses.
    Raises ValueError on bad input.
    """
    expr = expr.replace(",", "").strip()
    if not expr:
        raise ValueError("Empty expression")

    try:
        node = ast.parse(expr, mode="eval")
    except Exception:
        raise ValueError("Invalid expression syntax")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):  # Py3.8+
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants are allowed")
        if isinstance(node, ast.Num):  # older AST node
            return node.n
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type in ALLOWED_OPERATORS:
                return ALLOWED_OPERATORS[op_type](left, right)
            raise ValueError("Operator not allowed")
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type in ALLOWED_OPERATORS:
                return ALLOWED_OPERATORS[op_type](operand)
            raise ValueError("Unary operator not allowed")
        raise ValueError("Expression contains disallowed elements")

    return _eval(node)

# -------------------------
# Definition lookup (Wikipedia -> DuckDuckGo fallback)
# -------------------------
def wikipedia_summary(term: str, sentences=WIKI_SENTENCES):
    try:
        summary = wikipedia.summary(term, sentences=sentences, auto_suggest=True, redirect=True)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        # Try the first sensible option
        if e.options:
            try:
                return wikipedia.summary(e.options[0], sentences=sentences)
            except Exception:
                return None
        return None
    except Exception:
        return None

def duckduckgo_instant_answer(term: str):
    try:
        params = {"q": term, "format": "json", "no_html": 1, "skip_disambig": 1}
        r = requests.get("https://api.duckduckgo.com/", params=params, timeout=5)
        data = r.json()
        if data.get("Definition"):
            return data["Definition"]
        if data.get("AbstractText"):
            return data["AbstractText"]
        # Try first related topic text as last resort
        related = data.get("RelatedTopics", [])
        for item in related:
            if isinstance(item, dict) and item.get("Text"):
                return item["Text"]
        return None
    except Exception:
        return None

def lookup_definition(term: str):
    term = term.strip()
    if not term:
        return None
    # Try Wikipedia first
    speak(f"Let me look that up for you: {term}", say_aloud=False)
    wiki = wikipedia_summary(term)
    if wiki:
        return wiki
    # Then DuckDuckGo
    ddg = duckduckgo_instant_answer(term)
    return ddg

# -------------------------
# Natural language helpers
# -------------------------
def prettify_number(value):
    """Format numbers to human-friendly strings (drop .0 on integers)."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)

def is_probably_arithmetic(text: str):
    """
    Heuristic: if the text contains digits or common operator words, treat as arithmetic.
    This helps avoid false positives for single-word definitions.
    """
    tokenized = text.lower()
    operators = ["+", "-", "*", "/", "%", "plus", "minus", "times", "divided", "divide", "over", "mod", "power", "**"]
    if any(ch.isdigit() for ch in tokenized):
        return True
    if any(op in tokenized for op in operators):
        return True
    return False

# -------------------------
# Command handling
# -------------------------
def handle_user_command(command_text: str):
    """
    Decide what the user meant, run the requested action, and speak a human-friendly reply.
    Returns "exit" string if assistant should stop; otherwise returns None.
    """
    if not command_text:
        speak("I didn't catch that. Could you say it again, please?")
        return None

    text = command_text.strip().lower()

    # polite exit
    if any(word in text for word in ("exit", "quit", "stop", "goodbye", "bye")):
        speak("Okay — I'll be here if you need me. Goodbye!")
        return "exit"

    # Open websites
    if "open youtube" in text or "youtube" == text:
        speak("Opening YouTube for you.")
        webbrowser.open("https://www.youtube.com")
        return None
    if "open google" in text or "google" == text:
        speak("Opening Google for you.")
        webbrowser.open("https://www.google.com")
        return None

    # Arithmetic try
    if is_probably_arithmetic(text) or any(text.startswith(k) for k in ("calculate", "what is", "evaluate", "compute")):
        # remove leading trigger words
        expr = text
        for trig in ("calculate", "evaluate", "compute", "what is"):
            if expr.startswith(trig):
                expr = expr[len(trig):].strip()
                break

        # Replace common spoken words with symbols
        replacements = {
            " plus ": " + ",
            " minus ": " - ",
            " times ": " * ",
            " multiplied by ": " * ",
            " divided by ": " / ",
            " over ": " / ",
            " mod ": " % ",
            " modulo ": " % ",
            " to the power of ": "**",
            " power of ": "**",
        }
        expr_for_eval = f" {expr} "
        for k, v in replacements.items():
            expr_for_eval = expr_for_eval.replace(k, v)
        expr_for_eval = expr_for_eval.strip()

        # Accept only safe characters to avoid code execution
        allowed_chars = set("0123456789+-*/(). %")
        if all(ch in allowed_chars for ch in expr_for_eval):
            try:
                result = safe_eval(expr_for_eval)
                pretty = prettify_number(result)
                speak(f"The result is {pretty}")
                return None
            except Exception as e:
                print(f"[Arithmetic] Couldn't evaluate '{expr_for_eval}': {e}")
                # fallthrough to definition or fallback
        else:
            # If it contains words like "two" we won't attempt conversion here
            print(f"[Arithmetic] Expression contains unsafe characters: {expr_for_eval}")

    # Definitions and "what is X" style queries
    for prefix in ("define ", "definition of ", "what is ", "what's ", "tell me about "):
        if text.startswith(prefix):
            term = text[len(prefix):].strip()
            for art in ("a ", "an ", "the "):
                if term.startswith(art):
                    term = term[len(art):]
            if not term:
                speak("What would you like me to define?")
                return None
            speak(f"Searching for {term} — here's what I found.")
            defn = lookup_definition(term)
            if defn:
                speak(defn)
            else:
                speak("I couldn't find a short definition. Would you like me to open a web search for that?")
                # Here we do not automatically open the browser; we wait for the user
            return None

    # Short single-word or short-phrase attempts: try definition
    if len(text.split()) <= 3:
        defn = lookup_definition(text)
        if defn:
            speak(defn)
            return None

    # If nothing else matched, be helpful and offer options
    speak("Sorry, I didn't quite understand. You can ask me to calculate something (for example: 'calculate 23 times 7') or ask for a definition ('define recursion'). What would you like to do?")
    return None

# -------------------------
# Main assistant loop (human-friendly)
# -------------------------
def main():
    speak("Hi — I'm your assistant. Say 'assistant' before your request. If you want to stop, say 'stop' or 'exit'.")

    try:
        while True:
            # Wait for user speech
            spoken = listen_from_mic()
            if not spoken:
                # didn't hear anything; continue listening
                continue

            # Require wake word to avoid accidental triggers
            if WAKE_WORD in spoken:
                # Extract the part after the wake word
                after = spoken.split(WAKE_WORD, 1)[1].strip()
                # If user only said the wake word, prompt for the real command
                if not after:
                    speak("Yes? What can I help you with?")
                    # Small retry loop
                    for attempt in range(MAX_RETRIES):
                        follow = listen_from_mic()
                        if follow:
                            result = handle_user_command(follow)
                            if result == "exit":
                                return
                            break
                        else:
                            speak("I didn't catch that. Could you say it again?")
                    continue

                # We have a command; handle it
                result = handle_user_command(after)
                if result == "exit":
                    return
            else:
                print("[Note] Wake word not present — say 'assistant' first.")

    except KeyboardInterrupt:
        print("\n[User interrupted]")
        speak("Okay, stopping now. Bye!")
        sys.exit(0)

if __name__ == "__main__":
    main()
