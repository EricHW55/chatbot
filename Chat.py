import numpy as np 
from tensorflow.keras import preprocessing, utils
from tensorflow.keras.models import load_model as load
import os
import yaml
import re
from time import sleep

# model = load('model.h5')
# model.summary()

dir_path = 'dataset'
files_list = os.listdir(dir_path + os.sep)  
print('Preparing...')
# print(*files_list)

questions = list()
answers = list()

for filepath in files_list:
    stream = open( dir_path + os.sep + filepath , 'rb')
    docs = yaml.safe_load(stream)
    conversations = docs['conversations']
    for con in conversations:
        if len( con ) > 2 :
            questions.append(con[0])
            replies = con[ 1 : ]
            ans = ''
            for rep in replies:
                ans += ' ' + rep
            answers.append( ans )
        elif len( con )> 1:
            questions.append(con[0])
            answers.append(con[1])

answers_with_tags = list()
for n,i in enumerate(answers):
    if type(i) == str:
        answers_with_tags.append(i)
    else:
        questions[n] = '__REMOVE__'
for i in range(questions.count('__REMOVE__')):
    questions.remove('__REMOVE__')

    
_dir1 = 'chatbot_dataset.txt'
_dir2 = 'data\\dialogs.txt'

def read_txt_data(questions, answers_with_tags, _dir):
    with open(_dir, 'r' ,encoding='UTF-8') as f:
        data = f.readlines()

    for i in data:
        [q,a] = i.split('\t')
        if '\n' in a: a.replace('\n', '')
        questions.append(q)
        answers_with_tags.append(a)
    
    return questions, answers_with_tags

questions, answers_with_tags = read_txt_data(questions, answers_with_tags, _dir1)
questions, answers_with_tags = read_txt_data(questions, answers_with_tags, _dir2)



answers = list()
for i in answers_with_tags:
    answers.append('<START> ' + i + ' <END>')

tokenizer = preprocessing.text.Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts( questions + answers )


VOCAB_SIZE = len( tokenizer.word_index )+1
print( 'VOCAB SIZE : {}'.format( VOCAB_SIZE ))
# print(sequences)


from gensim.models import Word2Vec
import re

# Data PreProcessing

vocab = []
for word in tokenizer.word_index:  # 데이터 토큰화
    vocab.append(word)  # 중복 단어 제거 후 vocab에 저장

def tokenize(sentences):
    tokens_list = []
    vocabulary = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        tokens = sentence.split()
        vocabulary += tokens
        tokens_list.append(tokens)
    return tokens_list, vocabulary


#encoder_input_data
tokenized_questions = tokenizer.texts_to_sequences( questions )
maxlen_questions = max( [len(x) for x in tokenized_questions ] )
padded_questions = preprocessing.sequence.pad_sequences( tokenized_questions, maxlen = maxlen_questions, padding = 'post')
encoder_input_data = np.array(padded_questions)
# print('Encoder Input Data :', encoder_input_data.shape, maxlen_questions)


# decoder_input_data
tokenized_answers = tokenizer.texts_to_sequences( answers )
for n,i in enumerate(tokenized_answers):
    tokenized_answers[n] = i[:-1]
maxlen_answers = max( [ len(x) for x in tokenized_answers ] )
padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=maxlen_answers , padding='post' )
decoder_input_data = np.array( padded_answers )
# print('Decoder Input Data :', decoder_input_data.shape , maxlen_answers)



# decoder_output_data
tokenized_answers = tokenizer.texts_to_sequences( answers )
for n,i in enumerate(tokenized_answers):
    tokenized_answers[n] = i[1:]

padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=maxlen_answers , padding='post' )  # padding : post VS pre
onehot_answers = utils.to_categorical( padded_answers , VOCAB_SIZE )
decoder_output_data = np.array( onehot_answers )
# print('Decoder Output Data :', decoder_output_data.shape )


def str_to_tokens( sentence : str ):

    words = sentence.lower().split()
    tokens_list = list()
  
    for word in words:
        try:tokens_list.append( tokenizer.word_index[ word ] ) 
        except: tokens_list.append(1)
    return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=maxlen_questions , padding='post')


def say(text:str):
    for t in text:
        print(t, end='')
        sleep(0.05)
    print()


enc_model = load('Encoder_model.h5')
dec_model = load('Decoder_model.h5')

for _ in range(50):
    states_values = enc_model.predict( str_to_tokens( input( '>>> ' ) ) )
    empty_target_seq = np.zeros( ( 1 , 1 ) )
    empty_target_seq[0, 0] = tokenizer.word_index['start']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition :
        dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
        sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
        sampled_word = None
        for word , index in tokenizer.word_index.items() :
            if sampled_word_index == index :
                decoded_translation += ' {}'.format( word ) if not word == 'end' else '.'
                sampled_word = word
        
        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True
            
        empty_target_seq = np.zeros( ( 1 , 1 ) )  
        empty_target_seq[ 0 , 0 ] = sampled_word_index
        states_values = [ h , c ] 

    # print( decoded_translation )
    say(decoded_translation)