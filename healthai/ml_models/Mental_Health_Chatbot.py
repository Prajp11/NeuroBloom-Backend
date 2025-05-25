# import os
# import json

# # Get the current script's directory
# script_dir = os.path.dirname(__file__)

# # Create the full path to the intents.json file
# file_path = os.path.join(script_dir, 'intents.json')

# # Load the dataset
# with open(file_path, 'r', encoding='utf-8') as file:
#     data = json.load(file)

# # Display dataset structure
# print(json.dumps(data, indent=4))


# from langdetect import detect
# import nltk
# import numpy as np
# from nltk.tokenize import word_tokenize
# from nltk.stem.porter import PorterStemmer


# # In[5]:


# nltk.download("punkt")
# stemmer = PorterStemmer()


# # In[6]:


# words, tags, xy = [], [], []


# # In[7]:


# for intent in data["intents"]:
#     tag = intent["tag"]
#     tags.append(tag)

#     for pattern in intent["patterns"]:
#         tokenized_words = word_tokenize(pattern)
#         words.extend(tokenized_words)
#         xy.append((tokenized_words, tag))

# words = sorted(set(stemmer.stem(w) for w in words if w not in ["?", ".", "!"]))
# tags = sorted(tags)


# # In[8]:


# # Function to create a bag of words
# def bag_of_words(tokenized_sentence, words):
#     tokenized_sentence = [stemmer.stem(w) for w in tokenized_sentence]
#     bag = np.zeros(len(words), dtype=np.float32)
#     for idx, w in enumerate(words):
#         if w in tokenized_sentence:
#             bag[idx] = 1
#     return bag


# # In[9]:


# # Prepare training data
# X_train, y_train = [], []


# # In[10]:


# for (pattern_sentence, tag) in xy:
#     X_train.append(bag_of_words(pattern_sentence, words))
#     y_train.append(tags.index(tag))

# X_train, y_train = np.array(X_train), np.array(y_train)

# print("Training Data Processed Successfully!")


# # In[11]:


# import torch
# import torch.nn as nn


# # In[12]:


# class ChatbotRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(ChatbotRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
#         out, _ = self.rnn(x.unsqueeze(1), h0)
#         out = self.fc(out[:, -1, :])
#         return out


# # In[13]:


# import torch.optim as optim


# # In[14]:


# # Hyperparameters
# input_size = len(X_train[0])
# hidden_size = 8
# output_size = len(tags)
# learning_rate = 0.01
# num_epochs = 1000


# # In[15]:


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # In[16]:


# # Convert data to PyTorch tensors
# X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
# y_train = torch.tensor(y_train, dtype=torch.long).to(device)


# # In[17]:


# # Initialize model
# model = ChatbotRNN(input_size, hidden_size, output_size).to(device)


# # In[18]:


# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# # In[19]:


# # Training loop
# for epoch in range(num_epochs):
#     outputs = model(X_train)
#     loss = criterion(outputs, y_train)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # Calculate accuracy
#     _, predicted = torch.max(outputs, dim=1)
#     correct = (predicted == y_train).sum().item()
#     accuracy = 100 * correct / y_train.size(0)

#     # Print for every epoch
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")


# # In[21]:


# # Save model
# torch.save(model.state_dict(), "Mental_Health_chatbot_rnn.pth")
# print("Training complete. Model saved as Mental_Health_chatbot_rnn.pth")


# # In[22]:


# # Load trained model
# model.load_state_dict(torch.load("Mental_health_chatbot_rnn.pth", map_location=device))
# model.eval()


# # In[23]:


# def chatbot_response(text):
#     # Detect the language of the input text
#     detected_language = detect(text)

#     # Map detected language to the corresponding language tag
#     language_map = {
#         "en": "English",
#         "es": "Spanish",
#         "fr": "French",
#         "ja": "Japanese",
#         "de": "German",
#         "pt": "Portuguese"
#     }

#     # Default to English if the language is not in the map
#     language = language_map.get(detected_language, "English")

#     # Preprocess the input text
#     bow = bag_of_words(word_tokenize(text), words)
#     bow = torch.tensor(bow, dtype=torch.float32).to(device)
    
#     output = model(bow.unsqueeze(0))
#     _, predicted = torch.max(output, dim=1)
#     tag = tags[predicted.item()]
    
#     # Find the intent matching the tag and language
#     for intent in data["intents"]:
#         if intent["tag"] == tag:
#             responses = intent["responses"]
#             # Filter responses based on detected language
#             if language == "Spanish":
#                 responses = [r for r in responses if r.startswith("¡Hola") or "¿Cómo te sientes?"]
#             elif language == "French":
#                 responses = [r for r in responses if r.startswith("Bonjour") or "Comment te sens-tu?"]
#             elif language == "Japanese":
#                 responses = [r for r in responses if r.startswith("こんにちは") or "今日はどんな気分ですか？"]
#             elif language == "German":
#                 responses = [r for r in responses if r.startswith("Guten Tag") or "Wie fühlst du dich heute?"]
#             elif language == "Portuguese":
#                 responses = [r for r in responses if r.startswith("Olá") or "Como você está se sentindo?"]

#             return np.random.choice(responses)

#     # Default response
#     return np.random.choice(intent["responses"])


# # In[24]:


# print(chatbot_response("Morning"))


# # In[25]:


# print(chatbot_response("Good Night"))


# # In[26]:


# print(chatbot_response("Tell me a joke "))


# # In[26]:


# print(chatbot_response("Should i go on bike ride to relif the stress"))


# # In[27]:


# print(chatbot_response("How can I deal with exam stress?"))


# # In[28]:


# print(chatbot_response("I'm stressed about grades. What can I do?"))


# # In[29]:


# print(chatbot_response("from few days I am not feeling well"))


# # In[23]:


# print(chatbot_response("what activities are good to improve mental health"))


# # In[ ]:

import os
import json

# Get the current script's directory
script_dir = os.path.dirname(__file__)

# Create the full path to the intents.json file
file_path = os.path.join(script_dir, 'intents.json')

# Load the dataset
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Display dataset structure
print(json.dumps(data, indent=4))

from langdetect import detect
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

nltk.download("punkt")
stemmer = PorterStemmer()

words, tags, xy = [], [], []

for intent in data["intents"]:
    tag = intent["tag"]
    tags.append(tag)

    for pattern in intent["patterns"]:
        tokenized_words = word_tokenize(pattern)
        words.extend(tokenized_words)
        xy.append((tokenized_words, tag))

words = sorted(set(stemmer.stem(w) for w in words if w not in ["?", ".", "!"]))
tags = sorted(tags)

# Function to create a bag of words
def bag_of_words(tokenized_sentence, words):
    tokenized_sentence = [stemmer.stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in tokenized_sentence:
            bag[idx] = 1
    return bag

# Prepare training data
X_train, y_train = [], []

for (pattern_sentence, tag) in xy:
    X_train.append(bag_of_words(pattern_sentence, words))
    y_train.append(tags.index(tag))

X_train, y_train = np.array(X_train), np.array(y_train)
print("Training Data Processed Successfully!")

import torch
import torch.nn as nn

class ChatbotRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x.unsqueeze(1), h0)
        out = self.fc(out[:, -1, :])
        return out

import torch.optim as optim

# Hyperparameters
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.01
num_epochs = 210

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)

# Initialize model
model = ChatbotRNN(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == y_train).sum().item()
    accuracy = 100 * correct / y_train.size(0)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

# Save model
torch.save(model.state_dict(), "Mental_Health_chatbot_rnn.pth")
print("Training complete. Model saved as Mental_Health_chatbot_rnn.pth")

# Load trained model
model.load_state_dict(torch.load("Mental_Health_chatbot_rnn.pth", map_location=device))
model.eval()

def chatbot_response(text):
    detected_language = detect(text)
    language_map = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "ja": "Japanese",
        "de": "German",
        "pt": "Portuguese"
    }
    language = language_map.get(detected_language, "English")

    bow = bag_of_words(word_tokenize(text), words)
    bow = torch.tensor(bow, dtype=torch.float32).to(device)
    
    output = model(bow.unsqueeze(0))
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    for intent in data["intents"]:
        if intent["tag"] == tag:
            responses = intent["responses"]
            if language == "Spanish":
                responses = [r for r in responses if r.startswith("¡Hola") or "¿Cómo te sientes?"]
            elif language == "French":
                responses = [r for r in responses if r.startswith("Bonjour") or "Comment te sens-tu?"]
            elif language == "Japanese":
                responses = [r for r in responses if r.startswith("こんにちは") or "今日はどんな気分ですか？"]
            elif language == "German":
                responses = [r for r in responses if r.startswith("Guten Tag") or "Wie fühlst du dich heute?"]
            elif language == "Portuguese":
                responses = [r for r in responses if r.startswith("Olá") or "Como você está se sentindo?"]

            return np.random.choice(responses)

    return np.random.choice(intent["responses"])


# ==========================
# 🔊 Voice Assistant Section
# ==========================
import speech_recognition as sr
import pyttsx3

engine = pyttsx3.init()

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def listen_to_user():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            print(f"🧑 You said: {text}")
            return text
        except sr.UnknownValueError:
            print("❗ Sorry, I didn't understand that.")
            speak_text("Sorry, I didn't understand that.")
            return ""
        except sr.RequestError:
            print("❗ Speech service unavailable.")
            speak_text("Speech service is unavailable.")
            return ""

def voice_chatbot():
    print("🔈 Voice Mental Health Assistant is now running. Say 'quit' to stop.")
    speak_text("Hi, I'm your mental health assistant. How can I help you today?")

    while True:
        user_input = listen_to_user()
        if user_input.lower() == "quit":
            speak_text("Take care. Goodbye!")
            break
        elif user_input.strip():
            response = chatbot_response(user_input)
            print(f"🤖 Chatbot: {response}")
            speak_text(response)

# Uncomment to run voice assistant
# voice_chatbot()






# Mental_Health_Chatbot.py - new one
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import random
# import numpy as np
# import nltk
# import json
# from nltk.tokenize import word_tokenize
# from torch.utils.data import Dataset, DataLoader
# import os

# nltk.download('punkt')

# # ----------------------------- Data Preprocessing -----------------------------

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# with open(os.path.join(BASE_DIR, 'intents.json'), encoding='utf-8') as f:
#     intents = json.load(f)

# all_words = []
# tags = []
# xy = []

# for intent in intents['intents']:
#     tag = intent['tag']
#     tags.append(tag)
#     for pattern in intent['patterns']:
#         w = word_tokenize(pattern)
#         all_words.extend(w)
#         xy.append((w, tag))

# ignore_words = ['?', '!', '.', ',']
# all_words = [w.lower() for w in all_words if w not in ignore_words]
# all_words = sorted(set(all_words))
# tags = sorted(set(tags))

# def bag_of_words(tokenized_sentence, words):
#     sentence_words = [w.lower() for w in tokenized_sentence]
#     bag = np.zeros(len(words), dtype=np.float32)
#     for idx, w in enumerate(words):
#         if w in sentence_words:
#             bag[idx] = 1
#     return bag

# X_train = []
# y_train = []

# for (pattern_sentence, tag) in xy:
#     bag = bag_of_words(pattern_sentence, all_words)
#     X_train.append(bag)
#     label = tags.index(tag)
#     y_train.append(label)

# X_train = np.array(X_train)
# y_train = np.array(y_train)

# print(f"Total training samples: {len(X_train)}")
# print(f"Vocabulary size: {len(all_words)}")

# # ----------------------------- Dataset and Model -----------------------------

# class ChatDataset(Dataset):
#     def __init__(self):
#         self.n_samples = len(X_train)
#         self.x_data = X_train
#         self.y_data = y_train

#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]

#     def __len__(self):
#         return self.n_samples

# class NeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(NeuralNet, self).__init__()
#         self.l1 = nn.Linear(input_size, hidden_size)
#         self.l2 = nn.Linear(hidden_size, hidden_size)
#         self.l3 = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         out = F.relu(self.l1(x))
#         out = F.relu(self.l2(out))
#         out = self.l3(out)
#         return out

# # ----------------------------- Training -----------------------------

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# input_size = len(all_words)
# hidden_size = 64
# output_size = len(tags)
# batch_size = 8
# learning_rate = 0.001
# num_epochs = 20

# dataset = ChatDataset()
# train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# model = NeuralNet(input_size, hidden_size, output_size).to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# for epoch in range(num_epochs):
#     for (words, labels) in train_loader:
#         words = words.to(dtype=torch.float32).to(device)
#         labels = labels.to(dtype=torch.long).to(device)

#         outputs = model(words)
#         loss = criterion(outputs, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# # ----------------------------- Save the Model -----------------------------

# model_data = {
#     "model_state": model.state_dict(),
#     "input_size": input_size,
#     "hidden_size": hidden_size,
#     "output_size": output_size,
#     "all_words": all_words,
#     "tags": tags
# }

# torch.save(model_data, "healthai/ml_models/Mental_health_chatbot_rnn.pth")
# print("Model training complete and saved.")

# # ----------------------------- Inference -----------------------------

# def tokenize(sentence):
#     return word_tokenize(sentence)

# def predict_class(sentence, threshold=0.75):
#     sentence = tokenize(sentence)
#     X = bag_of_words(sentence, all_words)
#     X = torch.from_numpy(X).float().unsqueeze(0).to(device)  # Add batch dimension
#     model.eval()
#     with torch.no_grad():
#         output = model(X)
#         probabilities = torch.softmax(output, dim=1)
#         top_prob, top_class = torch.max(probabilities, dim=1)

#     if top_prob.item() > threshold:
#         tag = tags[top_class.item()]
#         return tag, top_prob.item()
#     else:
#         return None, top_prob.item()

# def chatbot_response(msg):
#     tag, confidence = predict_class(msg)
#     if tag:
#         for intent in intents['intents']:
#             if intent['tag'] == tag:
#                 return random.choice(intent['responses'])
#     return "I'm not sure I understand. Could you try saying it another way?"

