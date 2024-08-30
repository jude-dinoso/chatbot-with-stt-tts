import numpy as np
import scipy as sp
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc

def calculate_delta(array):
    """Calculate and returns the delta of given feature vector matrix"""

    rows,cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
              first =0
            else:
              first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

def extract_features(audio,rate):
    """extract 20 dim mfcc features from an audio, performs CMS and combines 
    delta to make it 40 dim feature vector"""    
    
    mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature,delta)) 
    return combined

def my_dtw (x, y):
    cost_matrix = sp.spatial.distance.cdist(x, y,metric='seuclidean')
    m,n = np.shape(cost_matrix)
    for i in range(m):
        for j in range(n):
            if ((i==0) & (j==0)):
                cost_matrix[i,j] = cost_matrix[i,j]

            elif (i==0):
                cost_matrix[i,j] = cost_matrix[i,j] + cost_matrix[i,j-1]

            elif (j==0):
                cost_matrix[i,j] = cost_matrix[i,j] + cost_matrix[i-1,j]

            else:
                min_local_dist = cost_matrix[i-1,j]

                if min_local_dist > cost_matrix[i,j-1]:
                    min_local_dist = cost_matrix[i,j-1]

                if min_local_dist > cost_matrix[i-1,j-1]:
                    min_local_dist = cost_matrix[i-1,j-1]

                cost_matrix[i,j] = cost_matrix[i,j] + min_local_dist
    return cost_matrix[m-1,n-1]

def test_dtw():
    features = np.asarray(())
    sr,audio = read('VoiceData/Jude_0.wav')
    # extract 40 dimensional MFCC & delta MFCC features
    vector   = extract_features(audio,sr)
    
    sr,audio = read('VoiceData/Jude_1.wav')
    # extract 40 dimensional MFCC & delta MFCC features
    vector2   = extract_features(audio,sr)
    
    sr,audio = read('VoiceData/Jude_2.wav')
    # extract 40 dimensional MFCC & delta MFCC features
    vector3   = extract_features(audio,sr)
    
    sr,audio = read('VoiceData/Jude_3.wav')
    # extract 40 dimensional MFCC & delta MFCC features
    vector4   = extract_features(audio,sr)
    
    sr,audio = read('VoiceData/Jude_4.wav')
    # extract 40 dimensional MFCC & delta MFCC features
    vector5   = extract_features(audio,sr)
    
    sr,audio = read('VoiceData/Pam_0.wav')
    # extract 40 dimensional MFCC & delta MFCC features
    vector6   = extract_features(audio,sr)
    
    dtw_score = my_dtw(vector2,vector)
    dtw_score2 = my_dtw(vector3,vector2)
    dtw_score3 = my_dtw(vector5,vector3)
    dtw_score4 = my_dtw(vector5,vector4)
    dtw_score5 = my_dtw(vector,vector6)
    dtw_score6 = my_dtw(vector2,vector6)
