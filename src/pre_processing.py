"""
    (a) lowercase
    (b) remove and replace url, mentions ("i.e.,@user"), replace lots of whitespace with one instance
    (c) remove numbers, punctuation and stopwords
    (d) tokenise   
    (e) stemming to reduce word flexions

"""
import nltk

nltk.download('stopwords')

import re                                               # library for regular expression operations
from nltk.corpus import stopwords                       # module for stop words that come with NLTK
import string                                           # for string operations
from nltk.stem import PorterStemmer                     # module for stemming
from nltk.tokenize import TweetTokenizer                # module for tokenizing s
from unicodedata import normalize

def clear_replace(text_string):
    """
    Accepts a text string and remove:
    1) urls
    2) mentions 
    3) numbers
    4) special characters

    And replace lots of whitespace with one instance
    """
    rt_regex = 'RT @[\w_]+'
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    number_regex = '[0-9]'
    special_character_regex = '[^A-Za-z0-9 ]+'

    #new code       
    # Removendo tags
    parsed_text = re.sub(space_pattern, ' ', text_string)
    
    #new code remove RT
    parsed_text = re.sub(rt_regex, ' ', parsed_text) 
    
    parsed_text = re.sub(giant_url_regex, ' ', parsed_text)
    parsed_text = re.sub(mention_regex, ' ', parsed_text)
    parsed_text = re.sub(number_regex, ' ', parsed_text)
    # removing special characters
    parsed_text = normalize('NFKD', parsed_text).encode('ASCII', 'ignore').decode('ASCII')
    parsed_text = re.sub(special_character_regex, '', parsed_text)
    
    return parsed_text

def stemming(tweet_tokens):
  """stems tweets. Returns a string."""
  # Instantiate stemming class
  stemmer = PorterStemmer() 

  # Create an empty list to store the stems
  tweets_stem = [] 

  for word in tweet_tokens:
      stem_word = stemmer.stem(word)  # stemming word
      tweets_stem.append(stem_word)  # append to the list
  return tweets_stem

def remove_stopwords_ponct(tweet_tokens):
	""" remove stop words and ponctuation remove_stopwords_ponct(tweet_tokens)"""
	stop_words = set(stopwords.words('english')) 

	filtered_sentence = []

	for word in tweet_tokens: # Go through every word in your tokens list
		if (word not in stop_words and  # remove stopwords
      word not in string.punctuation):  # remove punctuation
				filtered_sentence.append(word)

	return filtered_sentence

def tokenize(tweets):
  tk = TweetTokenizer()
  tokens = [tk.tokenize(t) for t in tweets]
  return tokens

def normalise(tweets_tokens):
  """ Untokenizer 
  input: list of tokens 
  output: list of sentences of strings """

  tweets = []
  for t in tweets_tokens:
    tweets.append(" ".join(t))
  
  return tweets

def pre_processing(tweets):	
	#lower-case the tweets
	tweets = [clear_replace(t) for t in tweets] 
	tweets = [str(t).lower() for t in tweets]           
	tweets_tokens = tokenize(tweets)  
	tweets_tokens = [stemming(t) for t in tweets_tokens]                     
	tweets_tokens = [remove_stopwords_ponct(t) for t in tweets_tokens]	
	tweets = normalise(tweets_tokens)

	return tweets#, tweets_tokens