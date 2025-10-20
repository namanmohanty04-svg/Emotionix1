import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from PyPDF2 import PdfReader
import openai
import uuid

# Config
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = f"sqlite:///{os.path.join(BASE_DIR, 'emotionix.db')}"

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET", "dev-secret-key-change-this")
app.config['SQLALCHEMY_DATABASE_URI'] = DB_PATH
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# OpenAI setup (optional)
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

### Database models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Chat(db.Model):
    id = db.Column(db.String(36), primary_key=True)  # uuid
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    ai_mode = db.Column(db.String(50), default="emotionix")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.String(36), db.ForeignKey('chat.id'), nullable=False)
    role = db.Column(db.String(10))  # 'user' or 'assistant' or 'system'
    content = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Create DB
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

### Utility: fallback generator (simple)
def fallback_generate(prompt, ai_mode="emotionix", board=None, grade=None):
    # Simple dynamic generation: not pre-canned. Use heuristics to expand prompt.
    p = prompt.strip()
    if ai_mode == "alphaStudy":
        prefix = f"(Alpha Study AI for {board or 'Generic Board'}, grade {grade or 'N/A'}) "
        return prefix + f"\nSummary: {p[:500]}.\nExplanation: Break it into simple steps and give an example."
    if ai_mode == "alphaExam":
        return f"Exam on: {p}\n1) Define the topic.\n2) Short question.\n3) Long question."
    # emotionix general
    return f"I understand: {p}\nHere's an empathetic answer: try reframing, examples, and a short action step."

### Utility: OpenAI wrapper (chat)
def openai_chat(messages, model="gpt-3.5-turbo", max_tokens=512, temperature=0.6):
    if not OPENAI_KEY:
        return None
    try:
        resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        app.logger.error("OpenAI call failed: %s", e)
        return None

### Routes - Auth
@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if not username or not password:
            flash("Username and password required", "danger")
            return redirect(url_for("signup"))
        if User.query.filter_by(username=username).first():
            flash("Username already exists", "danger")
            return redirect(url_for("signup"))
        user = User(username=username, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash("Account created", "success")
        return redirect(url_for("index"))
    return render_template("signup.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid credentials", "danger")
            return redirect(url_for("login"))
        login_user(user)
        flash("Logged in", "success")
        return redirect(url_for("index"))
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out", "info")
    return redirect(url_for("login"))

### Main UI
@app.route("/")
@login_required
def index():
    # load user's chats
    chats = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.created_at.desc()).all()
    return render_template("dashboard.html", chats=chats)

### Create new chat
@app.route("/chats/new", methods=["POST"])
@login_required
def new_chat():
    ai_mode = request.form.get("ai_mode", "emotionix")
    title = request.form.get("title") or f"New chat ({ai_mode})"
    chat_id = str(uuid.uuid4())
    chat = Chat(id=chat_id, user_id=current_user.id, title=title, ai_mode=ai_mode)
    db.session.add(chat)
    db.session.commit()
    # system message for context
    sys_msg = Message(chat_id=chat_id, role="system", content=f"AI mode:{ai_mode}")
    db.session.add(sys_msg)
    db.session.commit()
    return redirect(url_for("view_chat", chat_id=chat_id))

### View chat
@app.route("/chats/<chat_id>")
@login_required
def view_chat(chat_id):
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first_or_404()
    messages = Message.query.filter_by(chat_id=chat_id).order_by(Message.created_at).all()
    return render_template("chat.html", chat=chat, messages=messages)

### API endpoint to send a message and get AI response (sync)
@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    data = request.json or {}
    chat_id = data.get("chat_id")
    prompt = data.get("prompt", "").strip()
    emotion = data.get("emotion")  # optional
    ai_mode = data.get("ai_mode", "emotionix")
    board = data.get("board")
    grade = data.get("grade")

    if not chat_id or not prompt:
        return jsonify({"error":"missing chat_id or prompt"}), 400

    # Save user message
    user_msg = Message(chat_id=chat_id, role="user", content=prompt)
    db.session.add(user_msg)
    db.session.commit()

    # Build messages for OpenAI
    system_text = f"You are {ai_mode}. Be helpful and adapt tone to user's emotion: {emotion or 'neutral'}."
    if ai_mode == "alphaStudy":
        system_text += f" Student board: {board or 'any'}. Grade: {grade or 'any'}."

    # Fetch prior conversation (up to some messages)
    prior = Message.query.filter_by(chat_id=chat_id).order_by(Message.created_at).all()
    messages_for_api = [{"role": m.role, "content": m.content} for m in prior]

    # Ensure a system message at the start
    if not any(m['role']=="system" for m in messages_for_api):
        messages_for_api.insert(0, {"role":"system","content":system_text})
    else:
        # update existing system
        for m in messages_for_api:
            if m['role']=="system":
                m['content'] = system_text
                break

    # Add latest user message
    messages_for_api.append({"role":"user","content":prompt})

    # Try OpenAI
    ai_response_text = openai_chat(messages_for_api, model="gpt-3.5-turbo", temperature=0.7) if OPENAI_KEY else None
    if not ai_response_text:
        ai_response_text = fallback_generate(prompt, ai_mode=ai_mode, board=board, grade=grade)

    # Save assistant message
    assistant_msg = Message(chat_id=chat_id, role="assistant", content=ai_response_text)
    db.session.add(assistant_msg)
    db.session.commit()

    return jsonify({"answer": ai_response_text})

### Generate exam from PDF or text
@app.route("/api/generate_exam", methods=["POST"])
@login_required
def api_generate_exam():
    topic = request.form.get("topic") or "Exam"
    num_q = int(request.form.get("num_questions") or 10)
    text = request.form.get("text", "")

    # If file uploaded
    if 'file' in request.files:
        f = request.files['file']
        if f and (f.filename.lower().endswith(".pdf")):
            reader = PdfReader(f.stream)
            pages = []
            for p in reader.pages:
                pages.append(p.extract_text() or "")
            text = "\n\n".join(pages)

    if not text.strip():
        return jsonify({"error":"no text or pdf provided"}), 400

    prompt = f"Create an exam paper on '{topic}' with {num_q} questions from the following material:\n\n{text[:4000]}"
    # call openai
    answer = None
    if OPENAI_KEY:
        messages = [{"role":"system","content":"You are Alpha Exam AI that generates clear exam papers with marks and answer key."},
                    {"role":"user","content":prompt}]
        answer = openai_chat(messages, model="gpt-3.5-turbo", max_tokens=1200)
    if not answer:
        answer = fallback_generate(topic, ai_mode="alphaExam") + "\n\nSource excerpt:\n" + text[:800]
    return jsonify({"exam": answer})

### Static (favicon)
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
