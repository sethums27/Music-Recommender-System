import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import shutil
from flask import Flask, render_template, request, Response, g, url_for
import os
import cv2
import nltk
import numpy as np
from tensorflow import keras
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import json
import random
from PIL import Image
from skimage import transform
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotify_credentials

# python -m spacy download en_core_web_sm
# pip install --upgrade pyyaml
# nltk.download('stopwords')
# nltk.download('wordnet')
# lemmatizer = WordNetLemmatizer()

# Create a chatbot instance
chatbot = ChatBot('My Chatbot')
# Train the chatbot on some example data
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train("chatterbot.corpus.english")
trainer.train("chatterbot.corpus.english.greetings")
trainer.train("chatterbot.corpus.english.conversations")

app = Flask(__name__)


counter = 0
current_frame = None
query = {}
emo = []
emo_r = -1
emo_f = []
emo_t = []
songs = None

#HOME PAGE
@app.route('/')
def index():
    output = os.path.join(os.getcwd(), 'static\output')
    if os.path.exists(output):
        shutil.rmtree(output)
    return render_template('index.html')

#ABOUT PAGE
@app.route('/about')
def about():
    output = os.path.join(os.getcwd(), 'static\output')
    if os.path.exists(output):
        shutil.rmtree(output)
    return render_template('about.html')

#INTERSET FORM
@app.route('/page2')
def page2():
    output = os.path.join(os.getcwd(), 'static\output')
    if os.path.exists(output):
        shutil.rmtree(output)
    return render_template('page2.html')

#INTERST ACTION
@app.route('/chat', methods=['POST', 'GET'])
def chat():
    global i, query
    if request.method == 'POST':
        query['lang'] = request.form['lang']
        query['singer'] = request.form['singer']
        query['musicdir'] = request.form['musicdir']
        output = os.path.join(os.getcwd(), 'static/output')
        if os.path.exists(output):
            shutil.rmtree(output)
        if not os.path.exists(output):
            os.mkdir(output)
        return render_template('chat.html')
    else:
        return render_template('index.html')


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#FACE DETECTION 
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

#CAMERA UTILITY
def get_camera():
    with app.app_context():
        # Check if the camera has already been opened
        if 'camera' not in g:
            # Open the default camera (usually 0)
            g.camera = cv2.VideoCapture(0)

        return g.camera

#GENERATE FRAME
def generate_frames():
    global current_frame
    camera = get_camera()
    while True:
        success, frame = camera.read()
        current_frame = frame
        if not success:
            break
        else:
            framef = detect_faces(frame)
            ret, buffer = cv2.imencode('.jpg', framef)
            framef = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + framef + b'\r\n')

#LIVE FRAMES
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#CHATBOT
@app.route('/chat2', methods=['POST', 'GET'])
def chat2():
    global counter, current_frame, emo, emo_r, emo_f, emo_t, songs
    emo_f = []
    if request.method == 'POST':
        print("counter", counter)
        if counter > 4:
            model = load_model('models/cnn_new.h5')
            for x in os.listdir('static/output'):
                pt = os.path.join(os.getcwd(), 'static/output', x)
                image = load(pt)
                pred = model.predict(image) #ARRAY FORM WITH EACH EMOTION
                c = np.argmax(pred)
                emo.append(c)
                emo_f.append(c)
            ecount = []
            if 0 in emo:
                ecount.append(emo.count(0))
            else:
                ecount.append(0)
            if 1 in emo:
                ecount.append(emo.count(1))
            else:
                ecount.append(0)
            if 2 in emo:
                ecount.append(emo.count(2))
            else:
                ecount.append(0)
            emo_r = np.argmax(ecount)
            print(emo_f)
            print(emo_t)
            print(emo)
            print(ecount)
            print(emo_r)
            songs = search_song(emo_r)
            return url_for('result')
        else:
            output = os.path.join(os.getcwd(), 'static/output')
            pt = output+"/"+str(counter)+".png"
            print(pt)
            save_frame(current_frame, pt)
            counter += 1
            r = txt_to_emo(request.form['message'])
            emo.append(r)
            emo_t.append(r)
            print(emo_t)
            response = chatbot_res(request.form['message'])
            return response
    else:
        return render_template('index.html')

#RESULT PAGE
@app.route('/result')
def result():
    global emo_r, query, songs, emo_f, emo_t
    if songs == None:
        return render_template("index.html")
    else:
        output = os.path.join(os.getcwd(), 'static/output')
        dir_f = sorted(os.listdir(output))
        print(emo_r)
        return render_template("result.html", emo_r=emo_r, songs=songs, emo_f=emo_f, emo_t=emo_t, dir_f=dir_f)

#SAVING FRAME
def save_frame(frame, path):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_img = frame[y:y+h, x:x+w]
        face_filepath = path
        cv2.imwrite(face_filepath, face_img)


# chatbot
def chatbot_res(msg):
    res = chatbot.get_response(msg)
    return str(res)

# text to emotion
def txt_to_emo(txt):
    txt = [preprocess(x) for x in [txt]]
    cvn = pickle.load(open('models/cv.pkl', 'rb'))
    data_cv2 = cvn.transform(txt).toarray()
    model = keras.models.load_model('models/txt_emo.h5')
    preds = model.predict(data_cv2)
    preds_class = np.argmax(preds)
    return preds_class


def preprocess(line):
    ps = PorterStemmer()
    # leave only characters from a to z
    review = re.sub('[^a-zA-Z]', ' ', line)
    review = review.lower()  # lower the text
    review = review.split()  # turn string into list of words
    # apply Stemming
    review = [ps.stem(word) for word in review if not word in stopwords.words(
        'english')]  # delete stop words like I, and ,OR   review = ' '.join(review)
    # trun list into sentences
    return " ".join(review)


# cnn


def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image.convert('L')).astype('float32') / 255
    np_image = transform.resize(np_image, (48, 48, 1))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


# spotify

def search_song(emo):
    global query
    EMOTION_MAP = {
        0: {
            'tempo_range': '120-140',
            'valence_range': '0.7-1.0'
        },
        2: {
            'tempo_range': '60-80',
            'valence_range': '0.0-0.3'
        },
        1: {
            'tempo_range': '80-120',
            'valence_range': '0.4-0.6'
        }
    }
    tempo_range = EMOTION_MAP[emo]['tempo_range']
    valence_range = EMOTION_MAP[emo]['valence_range']

    ls = []
    c = 0
    client_credentials_manager = SpotifyClientCredentials(
        client_id=spotify_credentials.client_id, client_secret=spotify_credentials.client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    query1 = f'tempo:{tempo_range} valence:{valence_range} artists:{query["musicdir"]}&&{query["singer"]} language:{query["lang"]}'
    results = sp.search(q=query1, market='IN', type='track')
    while c < 10:
        ls2 = []
        track = results['tracks']['items'][c]
        ls2.append(track['name'])
        artist = []
        for x in track['artists']:
            artist.append(x['name'])
        ls2.append(",".join(artist))
        ls2.append(track['preview_url'])
        ls2.append(track['external_urls']['spotify'])

        ls.append(ls2)
        c += 1

    query1 = f'tempo:{tempo_range} valence:{valence_range} artists:{query["singer"]} language:{query["lang"]}'
    results = sp.search(q=query1, market='IN', type='track')
    while c < 10:
        ls2 = []
        track = results['tracks']['items'][c]
        ls2.append(track['name'])
        artist = []
        for x in track['artists']:
            artist.append(x['name'])
        ls2.append(",".join(artist))
        ls2.append(track['preview_url'])
        ls2.append(track['external_urls']['spotify'])

        ls.append(ls2)
        c += 1

    query1 = f'tempo:{tempo_range} valence:{valence_range} language:{query["lang"]}'
    results = sp.search(q=query1, market='IN', type='track')
    while c < 10:
        ls2 = []
        track = results['tracks']['items'][c]
        ls2.append(track['name'])
        artist = []
        for x in track['artists']:
            artist.append(x['name'])
        ls2.append(",".join(artist))
        ls2.append(track['preview_url'])
        ls2.append(track['external_urls']['spotify'])

        ls.append(ls2)
        c += 1

    query1 = f'tempo:{tempo_range} valence:{valence_range}'
    results = sp.search(q=query1, market='IN', type='track')
    while c < 10:
        ls2 = []
        track = results['tracks']['items'][c]
        ls2.append(track['name'])
        artist = []
        for x in track['artists']:
            artist.append(x['name'])
        ls2.append(",".join(artist))
        ls2.append(track['preview_url'])
        ls2.append(track['external_urls']['spotify'])

        ls.append(ls2)
        c += 1

    print(ls)
    return ls
