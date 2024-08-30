# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 18:48:11 2020

@author: jude.dinoso
"""

import time

import speech_recognition as sr


def recognize_speech_from_rec(recognizer, recording):
    
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    
    with recording as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.record(source)
        

    # set up the response object
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    try:
        response["transcription"] = recognizer.recognize_google(audio, language="ja-JP")
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

def speechrecognizer(file_name, samplingrate):
    
    # create recognizer and mic instances
    recognizer = sr.Recognizer()

    for i in range(1):
        # if a transcription is returned, break out of the loop and
        #     continue
        # if no transcription returned and API request failed, break
        #     loop and continue
        # if API request succeeded but no transcription was returned,
        #     re-prompt the user to say their guess again. Do this up
        #     to PROMPT_LIMIT times
        for j in range(3):
            print('Translating...')
            #recording = sr.AudioData(frame_data = raw_recording, 
            #                         sample_rate = samplingrate,
            #                         sample_width = 2)
            recording = sr.AudioFile(file_name)
            transcription = recognize_speech_from_rec(recognizer, recording)
            if transcription["transcription"]:
                break
            if not transcription["success"]:
                break
            print("I didn't catch that. What did you say?\n")

        # if there was an error, stop the game
        if transcription["error"]:
            print("ERROR: {}".format(transcription["error"]))
            break

        # show the user the transcription
        print("You said: {}".format(transcription["transcription"]))
        return transcription["transcription"]

       