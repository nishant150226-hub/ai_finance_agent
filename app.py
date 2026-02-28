import os
import json
import base64
from io import BytesIO
from flask import Flask, render_template, request, redirect, url_for, jsonify
import fitz
import pandas as pd
import matplotlib.pyplot as plt
from mistralai import Mistral
from dotenv import load_dotenv

# ------------------ ENV ------------------
load_dotenv()

# ------------------ Config ------------------
app = Flask(__name__)

BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, "storage")
CACHE_FILE = os.path.join(BASE_DIR, "extracted_cache.json")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

ALLOWED_EXTENSIONS = {"pdf", "txt", "csv"}

# ------------------ API Setup ------------------
api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    print("⚠ WARNING: MISTRAL_API_KEY not set. AI features disabled.")
    client = None
else:
    client = Mistral(api_key=api_key)

model_name = "mistral-large-latest"

# ------------------ Utility ------------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def read_file_text(file_path):
    ext = file_path.lower().split(".")[-1]
    text = ""

    try:
        if ext == "pdf":
            with fitz.open(file_path) as pdf:
                for page in pdf:
                    text += page.get_text()

        elif ext == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        elif ext == "csv":
            df = pd.read_csv(file_path)
            df.columns = [col.lower() for col in df.columns]

            required = {"tag", "amount", "date"}
            if not required.issubset(df.columns):
                print(f"CSV missing required columns: {required}")
                return ""

            for _, row in df.iterrows():
                text += f"{row['tag']}, {row['amount']}, {row['date']}\n"

    except Exception as e:
        print("File reading error:", e)

    return text


def clean_json_response(text):
    """
    Extract valid JSON array from AI output safely.
    """
    try:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
    except:
        pass
    return []


def ai_extract_data(text):
    if not text.strip() or not client:
        return []

    prompt = f"""
Extract financial data from the text below.
Return ONLY a JSON array:
[
  {{ "tag": "...", "amount": number, "date": "..." }}
]

Text:
{text}
"""

    try:
        response = client.chat.complete(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            content = response.choices[0].message.content
        except AttributeError:
            content = response.choices[0].content

        data = clean_json_response(content)

        for item in data:
            item["amount"] = float(item["amount"])

        return data

    except Exception as e:
        print("AI extraction failed:", e)
        return []


# ------------------ Caching ------------------

def load_cached_data():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return []


def save_cached_data(data):
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def process_all_files():
    """
    Only process files if cache is empty.
    """
    cached = load_cached_data()
    if cached:
        return cached

    all_data = []

    for file in os.listdir(UPLOAD_FOLDER):
        path = os.path.join(UPLOAD_FOLDER, file)
        text = read_file_text(path)
        all_data.extend(ai_extract_data(text))

    save_cached_data(all_data)
    return all_data


# ------------------ Charts ------------------

def generate_trend_chart(data):
    if not data:
        return None

    df = pd.DataFrame(data)

    if df.empty:
        return None

    try:
        grouped = df.groupby("tag")["amount"].sum()

        chart = BytesIO()
        grouped.plot(kind="bar")
        plt.title("Expenditure by Tag")
        plt.ylabel("Amount")
        plt.tight_layout()
        plt.savefig(chart, format="png")
        plt.close()

        chart.seek(0)
        return base64.b64encode(chart.read()).decode()

    except Exception as e:
        print("Chart error:", e)
        return None


# ------------------ Routes ------------------

@app.route('/')
def home():
    return render_template("dashboard.html")


@app.route('/your_uploads', methods=['GET', 'POST'])
def uploads():
    if request.method == 'POST':
        file = request.files.get("file")

        if not file or file.filename == "":
            return redirect(url_for("uploads"))

        if not allowed_file(file.filename):
            return "Invalid file type. Only PDF, TXT, CSV allowed.", 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Clear cache when new file uploaded
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)

        return redirect(url_for("uploads"))

    files = os.listdir(UPLOAD_FOLDER)
    return render_template("uploads.html", files=files)


@app.route('/results')
def results():
    data = process_all_files()
    return render_template("results.html", data=data)


@app.route('/trends')
def trends():
    data = process_all_files()
    chart = generate_trend_chart(data)
    error = None if chart else "No data available to generate trends."
    return render_template("trends.html", chart=chart, error=error)


@app.route('/suggestions')
def suggestions():
    data = process_all_files()

    if not data or not client:
        suggestion = "No financial data available."
    else:
        prompt = f"""
You are a financial advisor.
Here is structured financial data:
{json.dumps(data, indent=2)}

Give practical suggestions to improve financial habits.
"""
        try:
            response = client.chat.complete(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )

            try:
                suggestion = response.choices[0].message.content
            except AttributeError:
                suggestion = response.choices[0].content

        except:
            suggestion = "Could not generate suggestions."

    return render_template("suggestions.html", suggestion=suggestion)


@app.route('/awareness')
def awareness():
    data = process_all_files()

    if not data or not client:
        lesson = "No financial data available."
    else:
        prompt = f"""
Generate a financial awareness lesson based on this data:
{json.dumps(data, indent=2)}
"""
        try:
            response = client.chat.complete(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )

            try:
                lesson = response.choices[0].message.content
            except AttributeError:
                lesson = response.choices[0].content

        except:
            lesson = "Could not generate awareness content."

    return render_template("awareness.html", lesson=lesson)


@app.route('/chat', methods=['GET'])
def chat_page():
    return render_template("chat.html")


@app.route('/chat', methods=['POST'])
def chat():
    if not client:
        return jsonify({"reply": "AI service unavailable."})

    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"reply": "No message received."})

    data = process_all_files()

    prompt = f"""
You are a helpful financial assistant.
User financial data:
{json.dumps(data, indent=2)}

User question:
{user_message}
"""

    try:
        response = client.chat.complete(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            reply = response.choices[0].message.content
        except AttributeError:
            reply = response.choices[0].content

    except:
        reply = "Sorry, I couldn't process that."

    return jsonify({"reply": reply})


# ------------------ Run ------------------

if __name__ == "__main__":
    app.run(debug=False)