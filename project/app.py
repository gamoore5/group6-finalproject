# Basic Knowledge-Base Chatbot
# - Embeds a small knowledge base with SentenceTransformers
# - Retrieves top-k relevant facts using cosine similarity
# - Generates an answer with FLAN-T5 (using model/tokenizer directly for maximum compatibility)
import numpy as np
import torch
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#Use flask to build web app
app = Flask(__name__)
#CORS to bypass browsers blocking site
CORS(app)


# -----------------------------
# 1) Knowledge base
# -----------------------------
knowledge_base = [
    "The 2025 Bears had an 11-6 record",
    "In 2025 the Bears won their first playoff game since 2018.",
    "The Bears play at Soldier Field.",
    "The Bears are a Chicago-based football team in the NFC North division of the NFL.",
    "The Bears’ logo is an orange bear head.",
    "The Bears' old logo was the letter C.",
    "The official colors of the Chicago Bears are navy blue, orange, and white.",
    "The Bears’ rival team is the Green Bay Packers.",
    "In the 2025 regular season the Chicago Bears won six games after trailing in the final two minutes, the most in NFL history.",
    "For the first time in 24 years, the Bears overcame a double-digit deficit in the last two minutes of a game - twice against the Green Bay Packers.",
    "In the 2025 postseason, the Chicago Bears won their Wild Card game against the Green Bay Packers.",
    "In the 2025 postseason, the Chicago Bears lost the NFC Divisional Round to the Rams.",
    "The Bears' quarterback is Caleb Williams.",
    "The Bears' running back is D’Andre Swift.",
    "The Bears' tight end is Colsten Loveland.",
    "The Bears' offensive line includes Braxton Jones, Joe Thuney, Drew Dalman, Jonah Jackson, and Darnell Wright.",
    "The Bears' wide receivers are DJ Moore, Rome Odunze, and Luther Burden III.",
    "Luther Burden III is a rookie wide receiver for the Bears who ran for 652 receiving yards in the 2025 regular season. He also has an average of 2.67 yards per route run, the 3rd highest in the league after Puka Nacua and Jaxon Smith-Njigba.",
    "DJ Moore is a wide receiver for the Bears who ran for 682 yards and scored 6 touchdowns during the 2025 regular season.",
    "Cole Kmet is the Bears’ backup tight end, who ran for 347 receiving yards and scored 2 touchdowns in the 2025 regular season. He also caught the touchdown that sent the Bears into overtime against the Rams in the postseason Divisional Round.",
    "Kyle Monangai is the Bears’ rookie backup running back. He managed 783 rushing yards and 5 touchdowns during the 2025 regular season.",
    "The Bears' linebackers are Tremaine Edmunds, T.J. Edwards, and Noah Sewell.",
    "The Bears' head coach is Ben Johnson",
    "The Bears' defensive coordinator is Dennis Allen.",
    "The Bears' offensive coordinator is Press Taylor.",
    "The Bears' special teams coordinator is Richard Hightower.",
    "The Bears' senior director of coaching operations is Justin Rudd.",
    "The Bears' director of research and analysis is Harrison Freid.",
    "The Bears' safeties are Kevin Byard III and Jaquan Briskier.",
    "The Bears' defensive linemen are Montez Sweat, Grady Jarrett, Gervon Dexter Sr., and Dayo Odeyingbo.",
    "Kevin Byard III is the Bears’ starting free safety. He made 7 interceptions in the 2025 regular season, the most in the NFL.",
    "The Bears have won nine total championships.",
    "The Bears have been in two Super Bowls, in 1986 and 2007.",
    "The Bears won a Super Bowl against the Patriots in 1986.",
    "The Bears lost a Super Bowl to the Colts in 2007.",
    "The Bears were founded in 1919 as a company team.",
    "The Bears were originally named the Decatur Staleys.",
    "The Bears are the second oldest team in the league.",
    "Good, better, best. Never let it rest",
    "The Bears’ most famous player was Walter Payton.",
    "Walter Payton was a running back",
    "Walter Payton had the second highest career rushing yards of all time with 16,726",
    "The Bears were the first team to buy a player from another team with their purchase of Ed Healey from Rock Island in 1922.",
    "George Halas was the founder and owner of the Bears. In addition to his top executive roles, George Halas served the team as a player, coach, general manager and traveling secretary.",
    "George Halas’ nickname was Papa Bear.",
    "The colors the Chicago Bears use for their logo, orange and navy blue, were selected by the founder George Halas to honor his alma mater, the University of Illinois at Urbana-Champaign.",
    "The Bears formerly played in Wrigley Field, the famous home of the Chicago Cubs baseball team.",
    "The name Chicago Bears is a nod to a famed Chicago baseball team, the Chicago Cubs. Since football players tend to be larger than baseball players, George Halas used the word ‘bears’ to reflect the size difference.",
]

# -----------------------------
# 2) Embedding model + helpers
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize a 2D numpy array."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms

# Precompute and normalize KB vectors once
kb_vectors = embedder.encode(knowledge_base, convert_to_numpy=True)
kb_vectors = normalize(kb_vectors)

# -----------------------------
# 3) Generator model (robust)
# -----------------------------
MODEL_NAME = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def generate_answer(prompt: str, max_new_tokens: int = 80) -> str:
    """Generate an answer from a seq2seq model (deterministic)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,     # deterministic
            num_beams=1
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

# -----------------------------
# 4) Retrieval + Chat loop
# -----------------------------

TOP_K = 2
CONF_THRESHOLD = 0.5  # tune (0.35-0.60 typical)
DEBUG_SHOW_RETRIEVAL = False  # set True to print retrieved facts + scores

#Placed into a function to return the response instead of printing it for use on frontend
def bot_message(user_input):
    if not user_input:
        return "I'm a Chicago football fan! Ask me anything about the Bears."

    if user_input.lower() in {"hi","hello","hi there!"}:
      return "Hello! I'm a local football fan, here to provide you with knowledge about the Chicago Bears!"

    if user_input.lower() in {"tell me a fun fact","give me a fun fact","fun fact"}:
      return random.choice(knowledge_base)

    # Embed question and normalize
    q_vec = embedder.encode([user_input], convert_to_numpy=True)
    q_vec = normalize(q_vec)

    # Cosine similarity (dot product of normalized vectors)
    scores = (q_vec @ kb_vectors.T).flatten()

    # Retrieve top-k
    top_indices = np.argsort(scores)[::-1][:TOP_K]
    top_score = float(scores[top_indices[0]])

    if top_score < CONF_THRESHOLD:
        return "I don't know that one. Ask me about the Chicago Bears!\n"

    retrieved = [(knowledge_base[i], float(scores[i])) for i in top_indices]

    if DEBUG_SHOW_RETRIEVAL:
        print("\n[DEBUG] Retrieved facts:")
        for fact, sc in retrieved:
            print(f"  score={sc:.3f} | {fact}")
        print()

    context_block = "\n- " + "\n- ".join([fact for fact, _ in retrieved])

    prompt = (
        "You are a helpful assistant.\n"
        "Answer ONLY using the context below.\n"
        "If the answer is not in the context, reply exactly: "
        "\"I don’t know based on my knowledge base.\"\n\n"
        f"Context:{context_block}\n\n"
        f"Question: {user_input}\n"
        "Answer:"
    )

    response = generate_answer(prompt, max_new_tokens=100)

    #Edited to capitalize responses and add punctuation
    return response[:1].upper() + response[1:] + "."

#Send chatbot's response to the frontend in json format
@app.route('/bears_chatbot', methods=['POST'])
def send_chatbot_response():
    data = request.json
    user_input = data.get("message")
    chatbot_response = bot_message(user_input)
    return jsonify({"response":chatbot_response})

if (__name__) == '__main__':
    app.run(debug=True)