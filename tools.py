import re
import string
import nltk
from nltk.corpus import stopwords
from spellchecker import SpellChecker


acrodict = {'lol': 'laugh out loud', 'aamof': 'as a matter of fact', 'afaik': 'as far as i know', 'afair': 'as far as i remember', 
'fyeo': 'for your eyes only', 'aka': 'also known as', 'afk': 'away from keyboard', 'btk': 'back to keyboard', 'btt': 'back to topic', 
'btw': 'by the way', 'bc': 'because', 'cu': 'see you', 'diy': 'do it yourself', 'eobd': 'end of business day', 'eod': 'end of day', 
'eot': 'end of thread', 'faq': 'frequently asked questions', 'fka': 'formerly known as', 'fwiw': "for what its worth", 
'fyi': 'for your information', 'jfyi': 'just for your information', 'ftw': 'for the win', 'hf': 'have fun', 'hth': 'hope this helps', 
'idk': "i don't know", 'iirc': 'if i remember correctly', 'imho': 'in my humble opinion', 'imo': 'in my opinion', 'iow': 'in other words', 
'itt': 'in this thread', 'dgmw': "don't get me wrong", 'lmao': 'laughing my ass off', 'lmfao': 'laughing my fucking ass off', 
':)': 'smiling', 'xd': 'laughing', ':>': 'smiling', ':3': 'smiling', ':d': 'smiling', ':-)': 'smiling', ':/': 'disappointed', 
'smh': 'shaking my head', 'n/a': 'not available', 'nntr': 'no need to reply', 'noyb': 'none of your business', 
'nrn': 'no reply needed', 'omg': 'oh my god', 'omfg': 'oh my fucking god', 'op': 'original poster', 'rofl': 'rolling on the floor laughing', 
'otoh': 'on the other hand', 'tbh': 'to be honest', 'tbfh': 'to be fucking honest', 'irl': 'in real life', 'rsvp': 'reserve', 
'sflr': 'sorry for late reply', 'tba': 'to be announced', 'tbc': 'to be continued', 'tgif': 'thank god its friday', 'thx': 'thanks', 
'tnx': 'thanks', 'ty': 'thank you', 'tysm': 'thank you so much', 'ttyl': 'talk to you later', 'oml': 'oh my lord', 'ong': 'on god', 
'dtf': 'down to fuck', 'lmk': 'let me know', 'ngl': 'not going to lie', 'wfm': 'works for me', 'wtf': 'what the fuck', 'wth': 'what the hell', 
'ymmd': 'you made my day', 'icymi': 'in case you missed it', 'kit': 'keep in touch', 'hmu': 'hit me up', 'jfc': 'jesus fucking christ', 
'dw': "don't worry", 'dne': 'does not exist', 'np': 'no problem', 'jk': 'just kidding', 'kms': 'kill myself', 'kys': 'kill yourself', 
'kmn': 'kill me now', 'idc': "i don't care", 'ily': 'i love you', 'cya': 'see you later', 'gn': 'good night', 'gm': 'good morning', 
'brb': 'be right back', 'bff': 'best friends forever', 'asap': 'as soon as possible', 'asl': 'age sex location', 'tmr': 'tomorrow', 'tmw': 'tomorrow', 
'tmrw': 'tomorrow', 'nsfw': 'not safe for work', 'sfw': 'safe for work', 'nvm': 'never mind', 'omw': 'on my way', 'oic': 'oh i see', 
'rn': 'right now', 'ru': 'are you', 'dm': 'direct message', 'gtg': 'got to go', 'fb': 'facebook', 'ig': 'instagram', 'tfw': 'that feeling when', 
'mfw': 'my face when', 'pov': 'point of view', 'srsly': 'seriously', 'til': 'today i learned', 'tldr': 'too long did not read', 'nsfl': 'not safe for life', 
'nbd': 'no big deal', 'ppl': 'people', 'rt': 'retweet', 'mt': 'modified tweet', 'prt': 'partial retweet', 'ht': 'hat tip', 'cc': 'carbon copy',
'gtfo': 'get the fuck out', 'stfu': 'shut the fuck up', 'nfw': 'no fucking way', 'fml': 'fuck my life', 'yw': 'you', 'yolo': 'you only live once', 
'ffs': 'for fucks sake', 'bf': 'boyfriend', 'gf': 'girlfriend', 'wyd': 'what are you doing', 'kk': 'ok', 'probs': 'probably', 'prolly': 'probably',
'rip': 'rest in peace', 'ite': 'alright', 'aight': 'alright', 'aite': 'alright', 'bs': 'bull shit', 'jn': 'just now', 'w/e': 'whatever', 'w/': 'with',
'usagov': 'USA government', 'amirite': 'am i right', '<3': 'love', 'trfc': 'traffic', 'windstorm': 'wind storm', 'pls': 'please', 'tho': 'though',
'idps': 'internally displaced people', 'wtg' : 'way to go', 'wtpa' : 'where the party at', 'wuf' : 'where are you from', 'wuzup' : 'what is up',
'wywh' : 'wish you were here', 'ygtr' : 'you got that right', 'ynk' : 'you never know', 'zzz' : 'sleeping'}

contractions = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "i'd": "i would",
  "i'd've": "i would have",
  "i'll": "i will",
  "i'll've": "i will have",
  "i'm": "i am",
  "i've": "i have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

stop_words = set(stopwords.words('english'))

# Cleaning Functions

def clean_tags(tweet):
	'''Removes hashtags from a string'''
	tweet = re.sub('\# ', '', tweet)
	tweet = re.sub('\@ ', '', tweet)

	return tweet

def clean_links(tweet):
	tweet = re.sub('(https?:\/\/)?([\da-z\.-]+)\.([a-z]{2,6})([\/\w\.\?\=-]*)', '', tweet)
	return tweet

def clean_stopwords(tweet):
	tweet = ' '.join([word for word in tweet.split() if word not in stop_words])
	return tweet

def clean_punctuation(tweet):
	tweet = re.sub('[%s]' % re.escape(string.punctuation), '', tweet)
	return tweet

def clean_numbers(tweet):
	tweet = re.sub('\w*\d\w*', '', tweet)
	return tweet

def clean_symbols(tweet):
	tweet = re.sub('\w*[ûòóåêªÀÌÏÒÓÊÈ¼¢åÛ÷©¤£Ç]+\w*', ' ', tweet)
	return tweet

def clean_emojis(tweet):
	emojis = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
	tweet = emojis.sub('', tweet)
	return tweet

def clean_doublespace(tweet):

	tweet = re.sub('\s+', ' ', tweet).strip()
	return tweet

# Formatting Functions

def fix_acronyms(tweet):
	for acronym in acrodict.keys():
		if acronym == ':)':
			acronym = ':\)'
		elif acronym == ':-)':
			acronym = ':-\)'
		str = acronym + '+'
		tweet = re.sub(str, acronym, tweet)
		if acronym == 'lol':
			tweet = re.sub('l(ol)+', acronym, tweet)
		else:
			tweet = re.sub('({})+'.format(acronym), acronym, tweet)

	tweet = tweet.split()
	for index, word in enumerate(tweet):
		if word in acrodict:
			tweet[index] = acrodict[word]		

	tweet = ' '.join(tweet)
	return tweet

def fix_spelling(tweet):
	tweet = re.sub('c[0]{3,}l', 'cool', tweet)
	tweet = re.sub('y[e]{1,}[s]{1,}', 'yes', tweet)
	tweet = re.sub('pl[s]{1,}', 'please', tweet)
	tweet = re.sub('y[a]{1,}[s]{1,}', 'yes', tweet)
	tweet = re.sub('d[a]{1,}[m]{1,}[n]{1,}', 'damn', tweet)
	tweet = re.sub('[n]{1,}[o]{2,}', 'no', tweet)
	tweet = re.sub('b[e]{1,}f[o]{1,}r[e]{1,}', 'before', tweet)
	tweet = re.sub('[s]{1,}[o]{1,}', 'so', tweet)
	tweet = re.sub('[a]{1,}[h]{1,}', 'ah', tweet)
	tweet = re.sub('[o]{1,}[h]{1,}', 'oh', tweet)
	tweet = re.sub('[f]{1,}[u]{1,}[c]{1,}[k]{1,}', 'fuck', tweet)
	tweet = re.sub('[w]{1,}[o]{1,}[w]{1,}', 'wow', tweet)
	tweet = re.sub('[o]{1,}[r]{1,}', 'or', tweet)
	tweet = re.sub('[v]{1,}[e]{1,}[r]{1,}[y]{1,}', 'very', tweet)
	tweet = re.sub('w[h]{1,}[a]{1,}[t]{1,}', 'what', tweet)
	tweet = re.sub('br[o]{1,}', 'bro', tweet)
	tweet = re.sub('br[u]{1,}[h]{1,}', 'bro', tweet)

	tweet = re.sub('[#@]?\w*\d*\w*news\w*\d*\w*', 'news', tweet)
	tweet = re.sub('[#@]?\w*\d*\w*radio\w*\d*\w*', 'radio', tweet)

	tweet = re.sub('windstorm', 'wind storm', tweet)
	tweet = re.sub('irandeal', 'iran deal', tweet)

	tweet = re.sub('okwx', 'oklahoma city weather', tweet)
	tweet = re.sub('arwx', 'arkansas weather', tweet)
	tweet = re.sub('gawx', 'georgia weather', tweet)  
	tweet = re.sub('scwx', 'south carolina weather', tweet)  
	tweet = re.sub('cawx', 'california weather', tweet)
	tweet = re.sub('tnwx', 'tennessee weather', tweet)
	tweet = re.sub('azwx', 'arizona weather', tweet)  
	tweet = re.sub('alwx', 'alabama weather', tweet)  
	tweet = re.sub('usnwsgov', 'united states national weather service', tweet)

	tweet = re.sub(r'&gt;', '>', tweet)
	tweet = re.sub(r'&lt;', '<', tweet)
	tweet = re.sub(r'&amp;', '&', tweet)

	spell = SpellChecker()
	fixed_text = []
	misspelled = spell.unknown(tweet.split())

	for word in tweet.split():
		if word in misspelled:
			fixed_text.append(spell.correction(word))
		else:
			fixed_text.append(word)
	return ' '.join(fixed_text)

def fix_contractions(tweet):
	tweet = re.sub("'s", '', tweet)

	fixed_text = []
	for word in tweet.split():
		if word in contractions.keys():
			fixed_text.append(contractions[word])
		else:
			fixed_text.append(word)

	return ' '.join(fixed_text)


def full_clean(tweet):
	tweet = clean_emojis(tweet)
	tweet = clean_symbols(tweet)
	tweet = clean_links(tweet)
	tweet = clean_tags(tweet)
	tweet = tweet.lower()
	tweet = fix_contractions(tweet)
	tweet = clean_punctuation(tweet)
	tweet = fix_acronyms(tweet)
	tweet = clean_numbers(tweet) 
	tweet = clean_doublespace(tweet)
	tweet = fix_spelling(tweet)
	tweet = clean_stopwords(tweet)
	print(tweet)
	return tweet