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

    "The Chicago Bears play their home games at Soldier Field in Chicago.",
    "Soldier Field is the home stadium of the Chicago Bears.",
    "Where do the Chicago Bears play? They play at Soldier Field in Chicago.",
    "Where do they play? The Chicago Bears play at Soldier Field.",
    "The Bears' home stadium is Soldier Field.",

    "The Chicago Bears are a football team in the NFC North division of the NFL.",
    "The Chicago Bears are a professional American football team based in Chicago, Illinois.",
    "What team are the Chicago Bears? They are an NFL team in the NFC North division.",

    "The Bears' logo is an orange bear head.",
    "The Bears' old logo was the letter C.",
    "The official colors of the Chicago Bears are navy blue, orange, and white.",
    "What are the Chicago Bears' colors? They are navy blue, orange, and white.",
    "The Chicago Bears team colors are navy blue, orange, and white.",

    "The Bears' rival team is the Green Bay Packers.",

    "In the 2025 regular season the Chicago Bears won six games after trailing in the final two minutes, the most in NFL history.",
    "The Bears overcame a double-digit deficit in the final two minutes of a game twice against the Green Bay Packers.",
    "In the 2025 postseason, the Chicago Bears won their Wild Card game against the Green Bay Packers.",
    "In the 2025 postseason, the Chicago Bears lost the NFC Divisional Round to the Rams.",

    "The Bears' quarterback is Caleb Williams.",
    "Who is the quarterback of the Chicago Bears? Caleb Williams is the starting quarterback.",
    "The Chicago Bears quarterback is Caleb Williams.",

    "The Bears' running back is D’Andre Swift.",
    "The Bears' tight end is Colsten Loveland.",
    "The Bears' wide receivers are DJ Moore, Rome Odunze, and Luther Burden III.",

    "DJ Moore is a wide receiver for the Chicago Bears.",
    "Luther Burden III is a rookie wide receiver for the Chicago Bears with strong 2025 production.",
    "Cole Kmet is a tight end for the Chicago Bears and contributed key postseason plays.",
    "Kyle Monangai is a rookie running back for the Chicago Bears.",

    "The Chicago Bears offense includes Caleb Williams, D’Andre Swift, DJ Moore, Rome Odunze, and Luther Burden III.",
    "The Chicago Bears defense includes Montez Sweat, Tremaine Edmunds, T.J. Edwards, and Kevin Byard III.",

    "The Bears' offensive line includes Braxton Jones, Joe Thuney, Drew Dalman, Jonah Jackson, and Darnell Wright.",
    "The Bears' linebackers are Tremaine Edmunds, T.J. Edwards, and Noah Sewell.",
    "T.J. Edwards is a linebacker for the Chicago Bears.",

    "The Chicago Bears' head coach is Ben Johnson.",
    "Who is the Bears head coach? Ben Johnson is the head coach of the Chicago Bears.",
    "Ben Johnson is the head coach of the Chicago Bears.",

    "The Bears' defensive coordinator is Dennis Allen.",
    "The Bears' offensive coordinator is Press Taylor.",
    "The Bears' special teams coordinator is Richard Hightower.",

    "The Bears' safeties include Kevin Byard III and Jaquan Brisker.",
    "Kevin Byard III is a safety for the Chicago Bears and recorded multiple interceptions in the 2025 season.",
    "Montez Sweat is a defensive end for the Chicago Bears known for pass rushing.",
    "The Bears' defensive line includes Montez Sweat, Grady Jarrett, Gervon Dexter Sr., and Dayo Odeyingbo.",

    "The Chicago Bears have won nine total championships.",
    "The Bears have appeared in two Super Bowls: 1986 and 2007.",
    "The Bears won a Super Bowl against the Patriots in 1986.",
    "The Bears lost a Super Bowl to the Colts in 2007.",

    "The Chicago Bears were founded in 1919 as a company team originally called the Decatur Staleys.",
    "The Chicago Bears are one of the oldest teams in the NFL.",

    "Walter Payton was a legendary running back for the Chicago Bears.",
    "Walter Payton ranks second in all-time career rushing yards with 16,726.",
    "George Halas founded and owned the Chicago Bears and served as a player, coach, and executive.",
    "George Halas was nicknamed 'Papa Bear'.",

    "The Chicago Bears previously played at Wrigley Field.",
    "The team's name reflects strength, inspired by Chicago baseball culture and larger player size compared to baseball athletes."
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
CONF_THRESHOLD = 0.35  # tune (0.35-0.60 typical)
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
    "You are a Chicago Bears information assistant.\n"
    "You must ONLY answer using the context below.\n"
    "If the question is not directly answered in the context, respond exactly:\n"
    "\"I don't know that one. Ask me about the Chicago Bears!\"\n\n"
    f"Context:\n{context_block}\n\n"
    f"Question: {user_input}\n"
    "Answer:"
    )

    response = generate_answer(prompt, max_new_tokens= 200)

    #Edited to capitalize responses and add punctuation
    return response[:1].upper() + response[1:]

#Send chatbot's response to the frontend in json format
@app.route('/bears_chatbot', methods=['POST'])
def send_chatbot_response():
    data = request.json
    user_input = data.get("message")
    chatbot_response = bot_message(user_input)
    return jsonify({"response":chatbot_response})

if (__name__) == '__main__':
    app.run(debug=True)