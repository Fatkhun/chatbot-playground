from flask import Flask, jsonify, render_template, request
import speech_recognition as srec
from gtts import gTTS
import os
import tempfile
import pygame


app = Flask(__name__)
app.static_folder = 'static'


import nltk
nltk.download('popular')
nltk.download('punkt_tab')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np


from keras.models import load_model
model = load_model('model/models.h5')
import json
import random
intents = json.loads(open('model/data.json').read())
words = pickle.load(open('model/texts.pkl','rb'))
classes = pickle.load(open('model/labels.pkl','rb'))


def say(phrase):
    tts = gTTS(text=phrase, lang="id")
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=True) as f:
        tmpfile = f.name
        print(tmpfile)
    tts.save(tmpfile)
    play_mp3(tmpfile)
    os.remove(tmpfile)


def play_mp3(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()

def perintah():
    recognizer = srec.Recognizer()
    with srec.Microphone() as source:
        print("Mendengarkan...")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Pindahkan ke dalam with
        try:
            audio = recognizer.listen(source, phrase_time_limit=5)
            print("Diterima...")
            text = recognizer.recognize_google(audio, language="id-ID")
            print(f"Text: {text}")
            # we need some special handling here to correctly print unicode characters to standard output
            if str is bytes: # this version of Python uses bytes for strings (Python 2)
                say(u"You said {}".format(text).encode("utf-8"))
            else: # this version of Python uses unicode for strings (Python 3+)
                say("You said {}".format(text))
            return text
        except srec.UnknownValueError:
            return "Maaf, suara tidak dikenali."
        except srec.RequestError:
            return "Maaf, terjadi kesalahan pada layanan pengenalan suara."
        except Exception as e:
            return f"Error: {str(e)}"
       
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    print(ints)
    if not ints:  # If ints is empty, return a default response
        return "Maaf, saya tidak mengerti pertanyaan Anda."
    
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
   ints = predict_class(msg, model)
   res = getResponse(ints, intents)
   return res


@app.route("/")
def home():
   return render_template("index.html")


@app.route("/get")
def get_bot_response():
   userText = request.args.get('msg')
   return chatbot_response(userText)


@app.route("/start-recording")
def start_recording():
    text = perintah()
    return jsonify({"response": chatbot_response(text), "message": text})

@app.route("/stop-recording")
def stop_recording():
    return jsonify({"message": "Berhenti..."})
  
if __name__ == "__main__":
   app.run()