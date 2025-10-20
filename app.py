import os
import io
from flask import Flask, render_template, request, jsonify, send_from_directory
from PyPDF2 import PdfReader
import requests
import openai

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load API key from env if present
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

# Simple fallback "thinking" function (rule-based) if OpenAI is not configured
def fallback_generate_response(prompt, emotion=None, role="general"):
    # Very simple heuristics: if question contains "explain", "how", "why" -> longer answer.
    base = ""
    if "explain" in prompt.lower() or "how" in prompt.lower() or "why" in prompt.lower():
        base = ("Here's a clear explanation:\n\n" +
                "1) State the concept simply.\n2) Provide an example.\n3) Summarize key points.")
    else:
        base = "Short answer: " + (prompt[:200] + ("..." if len(prompt) > 200 else ""))

    # Adjust tone slightly by detected emotion
    if emotion == "sad":
        tone = "\n\nI sense you might be feeling down — I'll keep this gentle and encouraging."
    elif emotion == "angry":
        tone = "\n\nI sense frustration — I'll be direct and concise."
    elif emotion == "happy":
        tone = "\n\nNice! Here's a friendly answer with examples."
    else:
        tone = ""

    return base + tone

# If OPENAI_KEY present, use it for better responses
def openai_generate(prompt, emotion=None, role="general"):
    if not OPENAI_KEY:
        return fallback_generate_response(prompt, emotion=emotion, role=role)

    system_prompt = "You are Emotionix, an empathetic assistant that adapts tone to detected user emotion."
    if role == "alpha_study":
        system_prompt = "You are Alpha Study AI: helpful tutor. Ask clarifying q's when needed, adapt to board and grade info."

    # minor tone guidance
    if emotion == "sad":
        system_prompt += " Keep the tone gentle and encouraging."
    elif emotion == "angry":
        system_prompt += " Be concise and direct."
    elif emotion == "happy":
        system_prompt += " Keep the tone upbeat and positive."

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # change to a model you have access to; or "gpt-4" / "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.6,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        # Failback
        return fallback_generate_response(prompt, emotion=emotion, role=role) + f"\n\n(Note: OpenAI call failed: {e})"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json or {}
    prompt = data.get("prompt", "")
    emotion = data.get("emotion")  # expected: 'happy', 'sad', 'angry', etc or None
    role = data.get("role", "general")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    answer = openai_generate(prompt, emotion=emotion, role=role)
    return jsonify({"answer": answer})

@app.route("/api/alpha_study", methods=["POST"])
def api_alpha_study():
    data = request.json or {}
    board = data.get("board", "Generic Board")
    grade = data.get("grade", "Grade")
    prompt = data.get("prompt", "")
    emotion = data.get("emotion")
    if not prompt:
        return jsonify({"error": "No prompt"}), 400

    # prepend context about board & grade
    full_prompt = f"You are Alpha Study AI for {board}, {grade}. Student asks: {prompt}\nGive an answer targeted to {grade} level, explain step-by-step and give 1 example if appropriate."
    answer = openai_generate(full_prompt, emotion=emotion, role="alpha_study")
    return jsonify({"answer": answer})

@app.route("/api/generate_exam", methods=["POST"])
def api_generate_exam():
    # Accept either raw text in JSON or a PDF upload multipart/form-data
    if request.content_type and "application/json" in request.content_type:
        data = request.json or {}
        text = data.get("text", "")
        topic = data.get("topic", "Exam")
        num_questions = int(data.get("num_questions", 10))
    else:
        # Expect file
        text = ""
        topic = request.form.get("topic", "Exam")
        num_questions = int(request.form.get("num_questions", 10))
        if 'file' in request.files:
            f = request.files['file']
            if f.mimetype == "application/pdf" or f.filename.lower().endswith(".pdf"):
                reader = PdfReader(f.stream)
                pages_text = []
                for p in reader.pages:
                    pages_text.append(p.extract_text() or "")
                text = "\n\n".join(pages_text)
            else:
                # attempt to read as text
                text = f.read().decode("utf-8", errors="ignore")

    if not text.strip():
        return jsonify({"error": "No text or pdf content supplied"}), 400

    # Compose prompt for generating a question paper
    prompt = (f"Generate an exam paper titled '{topic}'. Create {num_questions} questions across "
              "Question types: 20% MCQ, 30% Short answer, 50% long-answer. Indicate marks for each "
              "question and include an answer key. Use the following source material:\n\n" + text[:4000])
    # Note: trimming text to 4000 chars to keep prompt reasonable; you can change strategy in production.
    exam = openai_generate(prompt, emotion=None, role="alpha_exam")
    return jsonify({"exam": exam})

@app.route("/static/<path:path>")
def static_proxy(path):
    return send_from_directory("static", path)

if __name__ == "__main__":
    # for local dev only
    app.run(host="0.0.0.0", port=5000, debug=True)
