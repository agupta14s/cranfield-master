
from readers import  read_queries, read_documents
from nltk.corpus import stopwords
import math
import nltk
import operator
from collections import Counter
from string import punctuation

inverted_index = {}
unique_inverted_index = {}
ps = nltk.stem.porter.PorterStemmer()   #nltk potter stemming
cnt = Counter()


def remove_not_indexed_toknes(tokens):
    return [token for token in tokens if token in inverted_index]


def merge_two_postings(first, second):
    first_index = 0
    second_index = 0
    merged_list = []
    while first_index < len(first) and second_index < len(second):
        if first[first_index] == second[second_index]:
            merged_list.append(first[first_index])
            first_index = first_index + 1
            second_index = second_index + 1
        elif first[first_index] < second[second_index]:
            merged_list.append(first[first_index])        #OR Merging
            first_index = first_index + 1
        else:
            merged_list.append(second[second_index])     #OR Merging
            second_index = second_index + 1
    merged_list = list(set(merged_list))               #Remove duplicates
    return merged_list


def merge_postings(indexed_tokens):
    first_list = inverted_index[indexed_tokens[0]]
    second_list = []
    for each in range(1, len(indexed_tokens)):
        second_list = inverted_index[indexed_tokens[each]]
        first_list = merge_two_postings(first_list, second_list)
    return first_list


def search_query(query):
    tokens = tokenize(str(query['query']))   #tokenize the search query
    indexed_tokens = remove_not_indexed_toknes(tokens) #remove all unindexed tokens from the query (Not useful)
    if len(indexed_tokens) == 0:  #If no tokens left  return nothing
        return []
    elif len(indexed_tokens) == 1:  #If there is one token left return the posting for that one token
        return inverted_index[indexed_tokens[0]]
    else:                                        #Otherwise here we perform TF.IDF
        #return merge_postings(indexed_tokens)
        return tfidf_cal(tokens)


def tokenize(text):
    text = text.split(" ")
    stop_words = stopwords.words('english') + list(punctuation)
    filtered = [w for w in text if not w in stop_words]  # stopwords.words('english')]#Remove Stop Words
    token_list =[]
    for token in filtered:  # nltk potter stemming
        token = ps.stem(token)
        token_list.append(token)
    return token_list


def add_token_to_index(token, doc_id):

    if token in inverted_index:
        current_postings = inverted_index[token]
        current_postings.append(doc_id)
        inverted_index[token] = current_postings
    else:
        inverted_index[token] = [doc_id]
    cnt[token] +=1
    global docsize
    docsize= sum(cnt.values())
    return docsize




def add_to_index(document):
    for token in tokenize(document['title']):
        add_token_to_index(token, document['id'])


def create_index():
    for document in read_documents():
        add_to_index(document)
    print( "Created index with size {}".format(len(inverted_index)))



def unique_invertedindex(inverted_index):
    for key, value in inverted_index.items():
        unique_inverted_index[key] = (list(set(value)))
    return unique_inverted_index


def term_freq(token, unique_inverted_index):
    if token in unique_inverted_index:
        unique_inverted_index[token] = unique_inverted_index[token]+1
    else:
        unique_inverted_index[token] =1
    return unique_inverted_index


def idf(token,unique_inverted_index):
    if token in unique_inverted_index.items():
        doc_freq_size = len(unique_inverted_index[token])
        if doc_freq_size != 0:
            idf_cal = (1 + math.log(docsize / doc_freq_size))
            return idf_cal
        else:
            return 0



def tfidf(token):
    tfidf = []
    if token in unique_inverted_index.items():
         tfidf = (term_freq(token,unique_inverted_index) *idf(token, unique_inverted_index))
    return tfidf


def tfidf_cal(query):
    tfidf_token = {}
    score = {}
    tfidf_score = {}
    for token in query:
        for d in unique_inverted_index:
            tfidf_token = tfidf(token)
            s = 0
        for m,n in tfidf_token:
            if n !=0:
                s = s+ n
                score[d['id']] = s
                tfidf_score = sorted(score(),key=operator.itemgettr(1),reverse=True)
    return (tfidf_score)


create_index()
unique_invertedindex(inverted_index)


if __name__ == '__main__':
    all_queries = [query for query in read_queries() if query['query number'] != 0]
    for query in all_queries:
        documents = search_query(query)
        print ("Query:{} and Results:{}".format(query, documents))

