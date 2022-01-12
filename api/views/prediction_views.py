from datetime import date, datetime
from os import stat
from librosa.core import audio
from rest_framework import generics, status, views
import google.cloud.speech as speech
from rest_framework.response import Response
from rest_framework.utils import serializer_helpers
import tensorflow as tf
import h5py
import librosa
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import gluonnlp as nlp
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm, tqdm_notebook
from tqdm.notebook import tqdm
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from api.models import Emo_db, User, VoiceVector, TextVector
from api.serializers import Emo_dbSerializer, TextVectorSerializer, VoiceVectorSerializer


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
        bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair) 

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))
 
    
class BERTClassifier(nn.Module):
    def __init__(self,
        bert,
        hidden_size = 768,
        num_classes = 5, # softmax 사용 <- binary일 경우는 2
        dr_rate=None,
        params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

device = torch.device("cuda:0")
bertmodel, vocab = get_pytorch_kobert_model()
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

speech_model = tf.keras.models.load_model('C:/Users/usr/Desktop/1116_model.h5')


class CNN(generics.GenericAPIView):
    queryset = VoiceVector.objects.all()
    serialzer_class = VoiceVectorSerializer
    
    def get_serializer_class(self):
        if self.request.method == 'POST':
            return VoiceVectorSerializer
    
    def post(self, request):
        data = request.data
        
        X, sample_rate = librosa.load(data['filename'], res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis = 0)
        mfccs = np.expand_dims(mfccs, axis=0)
        mfccs = np.expand_dims(mfccs, axis=2)
        speech_res = speech_model.predict(mfccs)
        speech_res = speech_res[0].tolist()
        
        username = User.objects.get(username=data['username'])  
        voice_vector = VoiceVector(
            user_id=username,
            vector_voice=speech_res
        )
        
        voice_vector.save()
        
        return Response(status=status.HTTP_201_CREATED)
    

class KoBERT(generics.GenericAPIView):    
    queryset = TextVector.objects.all()
    serializer_class = TextVectorSerializer
    
    def get_serializer_class(self):
        if self.request.method == 'POST':
            return TextVectorSerializer
    
    def post(self, request): 
        data = request.data
        file = data['filename'].read()
                      
        device = torch.device('cpu')
        text_model = torch.load('1209_model2.pt', map_location=device, pickle_module=pickle)
        text_model.load_state_dict(torch.load('1209_model2_state_dict.pt', map_location=device))
        checkpoint= torch.load('12092_all.tar', map_location=device)
        text_model.load_state_dict(checkpoint['model'])  
        
        client = speech.SpeechClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="ko-KR",
            audio_channel_count = 1,
        )
        
        audiofile = speech.RecognitionAudio(content=file)
        response = client.recognize(config=config, audio=audiofile)

        for result in response.results:
            best_alternative = result.alternatives[0]
            transcript = best_alternative.transcript
              
        label = 0
        unseen_test = pd.DataFrame([[transcript, label]], columns = ['발화문', 'emotion'])
        print(unseen_test)
        unseen_values = unseen_test.values
        unseen_values.tolist()

        test_set = BERTDataset(unseen_values, 0, 1, tok, 128, True, False)
        test_input = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=0)
        
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_input)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            out = text_model(token_ids, valid_length, segment_ids)

        text_res = []

        text_res.append(float(out[0][4]))
        text_res.append(float(out[0][0]))
        text_res.append(float(out[0][1]))
        text_res.append(float(out[0][2]))
        text_res.append(float(out[0][3]))
        # {0: 'anger', 1: 'depressed', 2: 'fear', 3: 'happy', 4: 'sad'}
        
        nor_res = []
        sum = 0
        for i in range(5):
            nor_res.append((text_res[i]-min(text_res))/(max(text_res)-min(text_res)))
            sum += nor_res[i]
        for i in range(5):
            nor_res[i] = nor_res[i]/sum
        
        username = User.objects.get(username=data['username'])
        text_vector = TextVector(
            user_id=username,
            vector_text=nor_res
        )
        text_vector.save()
        
        return Response(status=status.HTTP_201_CREATED)
    

class Multimodal(generics.GenericAPIView):
    queryset = Emo_db.objects.all()
    serializer_class = Emo_dbSerializer
    
    def post(self, request, *args, **kwargs):     
        data = request.data
        text_res = []
        speech_res = []
        text_vector = TextVector.objects.get(
            user_id=data['username'],
            created=date.today()
        ).vector_text
    
        speech_vector = VoiceVector.objects.get(
            user_id=data['username'],
            created_voice=date.today()
        ).vector_voice
        
        text_vector = text_vector.replace('[', "")
        speech_vector = speech_vector.replace('[', "")  
        
        tmp = ""
        for i in range(len(text_vector)):
            if text_vector[i] != ',' and text_vector[i] != ']':
                tmp = tmp + text_vector[i]
            else:
                tmp = float(tmp)
                text_res.append(tmp)
                tmp = ""
        
        print(text_res)
        
        tmp = ""
        for i in range(len(speech_vector)):
            if speech_vector[i] != ',' and speech_vector[i] != ']':
                tmp = tmp + speech_vector[i]
                
            else:
                tmp = tmp.strip()
                tmp = float(tmp)
                speech_res.append(tmp)
                tmp = ""
                
        res = []
        for i in range(5):
            res.append(text_res[i]*0.3 + speech_res[i]*0.7)
        
        emotion_num = np.argmax(res)
        
        if emotion_num == 0:
            emotion = 'sad'
        elif emotion_num == 1:
            emotion = 'angry'
        elif emotion_num == 2:
            emotion = 'depressed'
        elif emotion_num == 3:
            emotion = 'fear'
        elif emotion_num == 4:
            emotion = 'happy'
        else:
            raise Exception('Not Able to predict emotion')
        
        username = User.objects.get(username=data['username'])
        emo_db = Emo_db(
            voice_data = data['filename'],
            emotion = emotion,
            user_id = username
        )
        emo_db.save()
        
        return Response(status=status.HTTP_201_CREATED)