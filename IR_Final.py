#!/usr/bin/env python
# coding: utf-8

import math
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


#Load the questions and their answers
data=pd.read_csv('questions.csv')

#Creating a dataframe only with unique questions and answers
q_corpus=data[['QuestionID','Question']].drop_duplicates()

 #Test question
 print("")
test_q=input("Enter a question :")

#creating a tuple of unique questions
documents=list(q_corpus['Question'])
documents.append(test_q)


#Removes question mark in question
def process_documents(docs):
    for i in range(len(docs)):
        docs[i]=docs[i].replace("?","")

process_documents(documents)

#Constructing the tfidf vector of all the questions
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

#Constructing cosine_similarity matrix with the test question
from sklearn.metrics.pairwise import cosine_similarity
dict_cosine={}
for i in range(len(q_corpus)):
    cd=cosine_similarity(tfidf_matrix[-1], tfidf_matrix[i])
    qid=q_corpus.iloc[i,0]
    dict_cosine[qid]=cd[0][0]


#sorting the dictionary for the top 10 most similar questions
import operator
dict_cosine=sorted(dict_cosine.items(), key=operator.itemgetter(1),reverse=True)

#Gets the top 10 most relevant question's qid
rel_qid=[dict_cosine[i][0] for i in range(0,10) if dict_cosine[i][1]!=0.0]

#gets all answers corresponding to relevant questions
answers=[]
for qid in rel_qid:
    rows=data.loc[data['QuestionID']==qid]
    #print(data[data['QuestionID']==qid])
    sent=rows['Sentence'].tolist()
    answers=answers+sent

#compares possible answers and test__q for relevant answers
answers.append(test_q)

#Removes wh word in the possible answers
processed_answer=[]
wh=['who','what','why','how','whom','when','where','which']
for ans in answers:
    ans=word_tokenize(ans)
    ans_sent=""
    for w in ans:
        if w not in wh:
            ans_sent=ans_sent+w+" "
    processed_answer.append(ans_sent)


#Making the tfidf matrix of possible answers
tfidf_answer_matrix=tfidf_vectorizer.fit_transform(processed_answer)

#Finding the cosine similarity between answers and test_q
dict_final_answers={}
for i in range(len(answers)-1):
    cs=cosine_similarity(tfidf_answer_matrix[-1], tfidf_answer_matrix[i])
    f=answers[i]
    dict_final_answers[f]=cs[0][0]

#Most relevant 10 answers based on cosine similarity
best_answers=dict_final_answers
best_answers=sorted(best_answers.items(), key=operator.itemgetter(1),reverse=True)
rel_ans=[best_answers[i][0] for i in range(0,10)]

#Named entity recognition
import nltk
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

def ner(answer):
    tokenized_ans=nltk.word_tokenize((answer))
    tagged_ans=nltk.pos_tag(tokenized_ans)
    chunked_ans=nltk.ne_chunk(tagged_ans)
    ner_tags=[]
    for tree in chunked_ans:
        if(hasattr(tree,'label')):
            ans_name=' '.join(c[0] for c in tree.leaves())
            ans_type=tree.label()
            ner_tags.append((ans_name,ans_type))
    return ner_tags

#Tokenise test question
question=nltk.word_tokenize(test_q)
question=[q.lower() for q in question]
final_answers = []

# Returns a list of nouns present in a sentence
def getNouns(sentence):
     nouns=[]
     wh=['who','what','why','how','whom','when','where','which']
     for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence.lower()))):
         if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS' or pos=='PRP' or pos=='PRP$'):
             if word not in wh:
                 nouns.append(word)
     return nouns


#Mapping answers with same NER as the test question
if 'where' in question:
    for i in rel_ans:
        tag_ans = ner(i)
        if ('LOCATION' or 'ORGANIZATION') in tag_ans:
            final_answers.append(i)

elif 'who' in question:
    for i in rel_ans:
        tag_ans = ner(i)
        if 'PERSON' in tag_ans:
            final_answers.append(i)

elif 'why' in question:
     for i in rel_ans:
        if 'because' in i:
            final_answers.append(i)

        elif 'for ' in i:
            final_answers.append(i)

        elif 'to ' in i:
            final_answers.append(i)


elif 'when' in question:
    for i in rel_ans:
        tag_ans = ner(i)
        if ('DURATION','DATE')  in tag_ans:
            final_answers.append(i)


elif ('how many') in question:
    for i in rel_ans:
        tag_ans = ner(i)
        if ('CARDINAL')  in tag_ans:
            final_answers.append(i)

elif ('how much') in question:
    for i in rel_ans:
        tag_ans = ner(i)
        if ('CARDINAL','DURATION')  in tag_ans:
            final_answers.append(i)

elif ('how') in question:
    for i in rel_ans:
        #if ('by') in nltk.word_tokenize(i):
        final_answers.append(i)


#Ordering after NER
answer_output =[]
best_answers=dict(best_answers)
for ans in final_answers:
        answer_output.append(ans)


for i in range(len(rel_ans)):
    if rel_ans[i] not in answer_output:
        answer_output.append(rel_ans[i])

#Scoring based on nouns present
score_dict={}
for i in range(len(answer_output)):
    score_dict[answer_output[i]]=0

nouns_question=getNouns(test_q.lower())
nouns_question=list(set(nouns_question))

for sent in answer_output:
    noun_sent=getNouns(sent.lower())
    noun_sent=list(set(noun_sent))
    for noun in nouns_question:
        if noun in noun_sent:
            score_dict[sent]+=1
            
#Sort according to nouns found score
score_dict=sorted(score_dict.items(), key=operator.itemgetter(1),reverse=True)

print("")

for i in range(0,10):
    ans=str(i+1)+". "+ score_dict[i][0])
    print(ans)
