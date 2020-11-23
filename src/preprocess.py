from emot.emo_unicode import UNICODE_EMO, EMOTICONS
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import pickle

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

class Preprocess():

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.STOPWORDS = stopwords.words('english')

    def remove_punct(self,text):
        text  = "".join([char for char in text if char not in string.punctuation])
        text = re.sub('[0-9]+', '', text)
        return text

    def get_mention(self,text):
        mention = re.findall(r"@([A-Za-z]+)",text)[0]
        
        return mention


    def clean_text(self,text):
    #     #links
        text = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)", '', text, flags=re.MULTILINE)
        text = re.sub(r"[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)", '', text, flags=re.MULTILINE)                                  
        
    #     #mentions @
        text = re.sub(r"@([A-Za-z]+)",'',text)
        
    #     #numbers
        text = re.sub('[0-9]+', '', text)

    #     #punctuations
        text  = "".join([char for char in text if char not in string.punctuation])
        
    #     #stopwords 
        
        text = " ".join([word for word in str(text).split() if word not in self.STOPWORDS])

        return text


    def process_emo(self,text):
        for emot in UNICODE_EMO:
            text = text.replace(emot, "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()))
        return text
        
    def convert_emoticons(self,text):
        for emot in EMOTICONS:
            text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
        return text


    def emoji(self,string):
        emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)

    def remove_emoticons(self,text):
        emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
        return emoticon_pattern.sub(r'', text)

    def word_lem(self,text):
        text = " ".join([self.lemmatizer.lemmatize(word) for word in text.split(" ")])
        return text


    def clean(self,data,emoticon_pattern):

        print("Preprocessing...")

        data['text'] = data['text'].str.lower()
        data['clean_text'] = data['text'].apply(lambda x:self.clean_text(x))
        data['clean_text'] = data['clean_text'].apply(lambda x: self.remove_punct(x))

        if emoticon_pattern==1:
            data['clean_text'] = data['clean_text'].apply(lambda x: self.process_emo(x))
            data['clean_text'] = data['clean_text'].apply(lambda x: self.convert_emoticons(x))
            data['clean_text'] = data['clean_text'].apply(lambda x: self.word_lem(x))

        elif emoticon_pattern ==2:
            data['clean_text'] = data['clean_text'].apply(lambda x: self.emoji(x))
            data['clean_text'] = data['clean_text'].apply(lambda x: self.remove_emoticons(x))     
            data['clean_text'] = data['clean_text'].apply(lambda x: self.word_lem(x))
   
        return data



class make_features():

    def __init__(self, reuse=False):

        self.reuse = reuse

    def get_features(self,data,vectorizer):

        print("Getting features....")

        if self.reuse:
            print("Reusing trained vectorizer...")
            features = vectorizer.transform(data).toarray()

        else:
            features = vectorizer.fit_transform(data).toarray()
            pickle.dump(vectorizer,open("vectorizers/vectorizer.pkl","wb"))

        return features


    def sample_data(self,Xtrain,ytrain,over=.4,under=.7):
        
        print("Sampling data...")
        over = SMOTE(sampling_strategy=over)
        under = RandomUnderSampler(sampling_strategy=under)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        Xtrain,ytrain = pipeline.fit_resample(Xtrain,ytrain)
    
        return Xtrain,ytrain




