import numpy as np
from collections import namedtuple

subsystem_result = namedtuple('subsystem_result', ['prediction', 'lastlayer'])
vid_start_end = namedtuple('start_end', ['start', 'end'])
bounding_box = namedtuple('bounding_box', ['x', 'y', 'h', 'w'])

class SceneSplitter():
    def split(self, video_path):
        # video_path is the path to the source video file
        # should return a list of vid_start_end namedtuple with
        # start and end frames in integers
        return [vid_start_end(start=24, end=35), vid_start_end(start=56, end=96)]

class BoundingBoxer():
    def __init__(self, device=None):
        self.device = device
    
    def box(self, video_path):
        # video_path is the path to the source video file
        # should return a list of bounding_box namedtuple.
        # This list should have the same length as the number of frames
        # in the video
        return [bounding_box(x=5, y=50, h=280, w=160)]

class SASSubsystem():
    def __init__(self, device=None):
######################### NOTICE FOR ALL SUBSYSTEMS #########################
        # device here will be a string like 'cuda:0'.
        # Initialise the GPU device using the library based on the library
        # that the subsystem uses.
#############################################################################
        self.device = device

    def __call__(self, input_va):
        # input_va here is a string containing the path to the audio or
        # video file to perform the prediction on
        # This function should return a subsystem_result namedtuple as specified above.
        # lastlayer should be a numpy array
        return self.get_prediction_lastlayer(input_va)
    
    def speech_to_text(self,input_va):
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.AudioFile(input_va) as source:
            audio = r.record(source)
        # Transcribe audio file
        text = r.recognize_sphinx(audio)
#         text = " ".join(text.split()) #only returns 500 words
        return(text) 
    
    def subsystem_core(self, text, model, tokenize):
        import pandas as pd
        from keras.models import Sequential
        from keras.layers import Dense, Activation
        import numpy as np
                
        text = pd.Series(text) 
        seq = tokenize.sequences_to_matrix(tokenize.texts_to_sequences(text))
        word_index = tokenize.word_index
        vocab_size = max(word_index.values())+1
        
        #extracting fully connected layer
        model2 = Sequential()
        model2.add(Dense(512, input_shape=(vocab_size,), weights = model.layers[0].get_weights() ))
        model2.add(Activation('relu'))
        model2.add(Dense(256,weights = model.layers[2].get_weights()))
        model2.add(Activation('relu'))        
        model2.add(Dense(128,weights = model.layers[4].get_weights()))
        model2.add(Activation('relu'))        
        model2.add(Dense(64,weights = model.layers[6].get_weights()))
        model2.add(Activation('relu'))
        last_layer = model2.predict(seq)
        
        #getting class prediction
        prediction = model.predict(seq)
#         print("Softmax predictions: ",np.argmax(prediction[0]),"\n")
        return (np.argmax(prediction[0]),last_layer)

                
    
    def model_iterator(self,text):
        from keras.models import load_model
        import keras
        import pickle
        model_tokenizer_pairs = [('CNNsst1.h5','sst1_tokenizer.pickle'), #5 class sentiment analysis
                             ('CNNtrec.h5','trec_tokenizer.pickle'), #7 class topic classification
                             ('CNNsubj.h5','subj_tokenizer.pickle')  #binary subjectivity analysis
                            ] #these are filepaths so need to change in case the files are not in the same directory
        predictions = []
        last_layers = []    
        
        for pair in model_tokenizer_pairs:
            model = load_model(pair[0]) #load model
            with open(pair[1], 'rb') as handle:
                tokenize = pickle.load(handle) #load tokenizer                
            prediction, last_layer = self.subsystem_core(text, model, tokenize)
            predictions.append(prediction)
            last_layers.append(last_layer)
            
        return predictions, last_layers    

    def get_prediction_lastlayer(self, input_va):
        text = self.speech_to_text(input_va)
        predictions,last_layers = self.model_iterator(text)  
        return subsystem_result(predictions,last_layers) #these are lists containing predictions and last layers of 3 models

class ActionSubsystem():
    def __init__(self, device=None):
        self.device = device

    def __call__(self, input_va, characters):
        # characters will be a dictionary of character to a list of bounding_box
        # namedtuple for that character. The list will have same length as number
        # of frames in the video.
        # prediction field should be a dictionary of character
        # to prediction value for each possible prediction.
        # If any characters are involved in an action requiring > 1 person,
        # create a new character name like 'pair0'
        # lastlayer should be a dictionary of numpy arrays with keys corresponding
        # to those in prediction
        return self.get_prediction_lastlayer(input_va)

    def get_prediction_lastlayer(self, input_va):
        return subsystem_result(prediction={'person0' : 'running', 'person1': 'sitting',
                                            'pair0' : 'hugging'},
                                lastlayer={'person0' : np.random.rand(5), 'person1' : np.random.rand(5),
                                           'pair0' : np.random.rand(5)})

class EmotionSubsystem():
    def __init__(self, device=None):
        self.device = device

    def __call__(self, input_va, characters):
        # characters will be a dictionary of character to a list of bounding_box
        # namedtuple for that character. The list will have same length as number
        # of frames in the video.
        # prediction field should be a dictionary of character
        # to prediction value for each possible prediction.
        # lastlayer should be a dictionary of numpy arrays with keys corresponding
        # to those in prediction
        return self.get_prediction_lastlayer(input_va)

    def get_prediction_lastlayer(self, input_va):
        return subsystem_result(prediction={'person0' : 'angery', 'person1' : 'happy'},
                                lastlayer={'person0' : np.random.rand(5), 'person1' : np.random.rand(5)})

class ContextSubsystem():
    def __init__(self, device=None):
        self.device = device

    def __call__(self, input_va):
        return self.get_prediction_lastlayer(input_va)

    def get_prediction_lastlayer(self, input_va):
        return subsystem_result(prediction='office', lastlayer=np.random.rand(5))
