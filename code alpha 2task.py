import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Step 1: FAQs (Question-Answer Pairs)
faq_data = {
    "What is Artificial Intelligence?": "Artificial Intelligence is the simulation of human intelligence processes by machines, especially computer systems.",
    "What is machine learning?": "Machine learning is a subset of AI that involves training algorithms to learn patterns from data and make decisions.",
    "What is deep learning?": "Deep learning is a subset of machine learning that uses neural networks with many layers to analyze various factors of data.",
    "What are neural networks?": "Neural networks are computing systems inspired by the biological neural networks of animal brains.",
    "What is natural language processing?": "Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language.",
    "What is computer vision?": "Computer vision is a field of AI that trains computers to interpret and understand the visual world.",
    "What is the difference between AI and ML?": "AI is the broader concept of machines being able to carry out tasks in a smart way, while ML is a subset of AI focused on learning from data.",
    "What are some real-life applications of AI?": "Examples include voice assistants, recommendation systems, autonomous vehicles, and fraud detection."
}

# Step 2: Preprocessing function
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Preprocess the questions
preprocessed_questions = [preprocess(q) for q in faq_data.keys()]

# Step 3: Vectorize the questions using TF-IDF
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(preprocessed_questions)

# Chatbot function
def get_response(user_input):
    user_input_processed = preprocess(user_input)
    user_vector = vectorizer.transform([user_input_processed])

    # Compute cosine similarity
    similarities = cosine_similarity(user_vector, question_vectors)
    best_match_index = similarities.argmax()
    best_score = similarities[0][best_match_index]

    if best_score < 0.3:  # Threshold for uncertain match
        return "I'm not sure I understand your question. Could you please rephrase?"
    else:
        matched_question = list(faq_data.keys())[best_match_index]
        return faq_data[matched_question]

# Step 4: Chat Loop
print("🤖 AI FAQ Chatbot (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break
    response = get_response(user_input)
    print("Bot:", response)
