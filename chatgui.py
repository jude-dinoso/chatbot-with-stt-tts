import nltk, os
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import speech_recognition as sr
import sounddevice as sd
from scipy.io.wavfile import write
from noisereduction import noisereduction
from featureextraction import extract_features
from gtts import gTTS
import pygame
import time
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
with open('intents.json',  encoding='utf8',  errors="ignore") as jsonfile:
    data = jsonfile.read()
 #   print(data)
    intents = json.loads(str(data))
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
speaker = ""

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
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    print(ints)
    plan = ints[0]['intent']
    if float(ints[0]['probability']) < 0.80:
        ints[0]['intent'] = 'noanswer'
    if plan == "plans":
        res = get_plans()
    elif plan == "make plan":
        res = make_plans(msg)
    else:
        res = getResponse(ints, intents)
    return res

def get_plans():
    currspeaker = SpeakerTitle['text']
    if os.path.isfile('{}_plans.json'.format(currspeaker)):
        with open('{}_plans.json'.format(currspeaker),  encoding='utf8',  errors="ignore") as jsonfile1:
            data = jsonfile1.read()
            print(data)
            plans = json.loads(str(data))
        if plans is not None:    
            #month = plans['Month'][0]
            #day = plans['Day'][0]
            activity = plans['Activity'][0]
            res = "予定があります。　{}".format(activity) + "を行います。"
    else:
        
        print('hello')
        res = "すみませんが予定がないです。"
    return res

def make_plans(msg):
    currspeaker = SpeakerTitle['text']
    res = "はい、どうぞ"
    ChatLog.insert(END, res + '\n\n')
    tts = gTTS(text=res, lang='ja')   
    tts.save('abc.mp3')
    pygame.mixer.init()
    pygame.mixer.music.load('abc.mp3')
    pygame.mixer.music.play()
    time.sleep(3)
    pygame.mixer.music.unload()
    os.remove('abc.mp3')
    
    response = recognize_speech_from_mic()
    print("{}".format(response["transcription"]))
    msg = response["transcription"]
    
    if os.path.isfile('{}_plans.json'.format(currspeaker)):
        os.remove('{}_plans.json'.format(currspeaker))
    data = {}
    data['Activity'] = []
    data['Activity'].append(msg)
    with open('{}_plans.json'.format(currspeaker), 'w') as outfile:
        json.dump(data, outfile)
    return "わかりました。　予定の内容は" + msg + "ことです。"
        
def speaker_test(myrecording):
        #path where training speakers will be saved
        modelpath = "VoiceModels/"
        gmm_files = [os.path.join(modelpath,fname) for fname in 
                 os.listdir(modelpath) if fname.endswith('.gmm')]
        fs = 44100  # Sample rate
        seconds = 5  # Duration of recording
    
        #Load the Gaussian gender Models
        models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
        speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname 
                  in gmm_files]
    
        error = 0
        total_sample = 0.0
        
        myrecording_clean = noisereduction(sound_array=myrecording, samplingrate=fs,seconds=seconds+1)
        write('testsample.wav', fs, myrecording_clean*2)
        vector   = extract_features(myrecording_clean,fs)
        
        log_likelihood = np.zeros(len(models)) 
        Actual_score = np.zeros(len(models)) 
    
        for i in range(len(models)):
            gmm = models[i]  #checking with each model one by one
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
            Actual_score[i] = (log_likelihood[i] + 30)*100/60
            
    
        Actual_score.sort()
        winner = np.argmax(log_likelihood)
        print(Actual_score[-1])
        print ("detected as {}さん ".format(speakers[winner]))
        if Actual_score[-1] > 100 or (Actual_score[-1] - Actual_score[-2]) > 20:
            return speakers[winner]
        else:
            return ' '
def recognize_speech_from_rec(myrecording):
    
    r = sr.Recognizer()
    with sr.AudioFile(myrecording) as source:
        r.adjust_for_ambient_noise(source)
        audio = r.record(source)
        

    # set up the response object
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    try:
        response["transcription"] = r.recognize_google(audio, language="ja-JP")
    except sr.RequestError:
        # API was unreachable or unresponsive
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        # speech was unintelligible
        response["error"] = "Unable to recognize speech"

    return response

def recognize_speech_from_mic():
    
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)
    # check that recognizer and microphone arguments are appropriate type

    # set up the response object
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    # try recognizing the speech in the recording
    # if a RequestError or UnknownValueError exception is caught,
    #     update the response object accordingly
    try:
        response["transcription"] = r.recognize_google(audio, language="ja-JP")
    except sr.RequestError:
        # API was unreachable or unresponsive
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        # speech was unintelligible
        response["error"] = "Unable to recognize speech"

    return response



#Creating GUI with tkinter
import tkinter
from tkinter import *
from mpg123 import Mpg123, Out123


def identify():
    fs = 44100  # Sample rate
    seconds = 5  # Duration of recording
    myrecording = sd.rec(int((seconds + 1) * fs), samplerate=fs, channels=1)    
    sd.wait()  # Wait until recording is finished
    speaker = speaker_test(myrecording)
    print(speaker)
    if speaker != ' ':
        SpeakerTitle['text'] = speaker 
        text = "Bot:{}さん おはようございます！".format(speaker)
        
        speech = "{}さん おはようございます！".format(speaker)
    else:
        SpeakerTitle['text'] = speaker 
        text = "すみませんが音声認識出来ません。"
        
        speech = "すみませんが音声認識出来ません。"
    ChatLog.config(state=NORMAL)
    ChatLog.insert(END, text + '\n')
    
    tts = gTTS(speech, lang='ja')
    tts.save('zzz.mp3')
    pygame.mixer.init()
    pygame.mixer.music.load('zzz.mp3')
    pygame.mixer.music.play()
    time.sleep(5)
    pygame.mixer.music.unload()
    os.remove('zzz.mp3')
    
def send():
    #input("Press Enter to start recording test sample...")
    
    response = recognize_speech_from_mic()
    print("{}".format(response["transcription"]))
    msg = response["transcription"]
    EntryBox.delete("0.0",END)

    if response["transcription"] != 'None':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
    
        res = chatbot_response(msg)
        
        ChatLog.insert(END, res + '\n\n')
        tts = gTTS(text=res, lang='ja')   
        tts.save('abc.mp3')
        pygame.mixer.init()
        pygame.mixer.music.load('abc.mp3')
        pygame.mixer.music.play()
        time.sleep(8)
        pygame.mixer.music.unload()
        os.remove('abc.mp3')
            
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
 

base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Speaker name
SpeakerTitle = Label(base,text="", font=("Verdana",12,'bold'), bd=0, bg="White", height="2", width="10")

#Speaker name
Speaker = Label(base, bd=0, bg="White", height="2", width="20", text="スピーカー")

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Record", width="12", height=3,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )
#Create Button to send message
IdentifyButton = Button(base, font=("Verdana",12,'bold'), text="Identify", width="12", height=3,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= identify )
#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=376,y=36, height=386)
Speaker.place(x=110,y=0, height=36, width=60)
SpeakerTitle.place(x=190,y=0, height=36, width=50)
ChatLog.place(x=6,y=36, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=30)
IdentifyButton.place(x=6, y=459, height=30)


base.mainloop()

