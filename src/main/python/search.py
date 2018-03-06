
from readers import  read_queries, read_documents
from nltk.corpus import stopwords
import math
import nltk
import operator


inverted_index = {}
unique_inverted_index = {}
#token_dict = {}
ps = nltk.stem.porter.PorterStemmer()   #nltk potter stemming


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
    tokens = tokenize(str(query['query']))
    indexed_tokens = remove_not_indexed_toknes(tokens)
    if len(indexed_tokens) == 0:
        return []
    elif len(indexed_tokens) == 1:
        return inverted_index[indexed_tokens[0]]
    else:
        return merge_postings(indexed_tokens)
        #return  tfidf_cal(query)


def tokenize(text):
    text = text.split(" ")
    filtered = [w for w in text if not w in stopwords.words('english')]#Remove Stop Words
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


def add_to_index(document):
    for token in tokenize(document['title']): # tokenize(document['title']):
        add_token_to_index(token, document['id'])


def create_index():
    for document in read_documents():
        add_to_index(document)
    print( "Created index with size {}".format(len(inverted_index)))


def unique_invertedindex(inverted_index):
    for k, i in inverted_index.items():
        unique_inverted_index[k] = (list(set(i)))
    return unique_inverted_index


def doc_size(document):
    document_id =[]
    for doc in document:
        document_id.append(doc['id'])
    return max(document_id)


def token_terms(unique_inverted_index ):
    token = []
    for k,i in unique_inverted_index.items():
        token.append(k)
    return token


def doc_freq(unique_inverted_index, token):
    if token in unique_inverted_index:
        return len(unique_inverted_index[token])


def idf(token,document,unique_inverted_index):
    idf_cal = math.log(doc_size(document)/doc_freq(unique_inverted_index,token))
    return idf_cal


def tf_weight(token, document_id):
    tfweight = {}

    for k, i in inverted_index.items():
        if token in k:
            tfreq = i.count(document_id)
            if tfreq == 0:
                tfweight[token] =0
            else:
                tfweight[token] = 1 +math.log(tfreq)
                return tfweight


def tfidf(token, document_id,document):
    tfidf = {}
    for t in token:
        if t in inverted_index:
            tfweight = (tf_weight(t,document_id))
            for k,i in tfweight.items():
                tfidf[k] = (i *idf(t,document, unique_inverted_index))
            return tfidf


def getquery(all_queries,qnum):
    for q in all_queries:
        if q['query number'] == qnum:
            return (q['query'])


def query_tokenize(query):
    query = query.split(" ")
    return  query
    #return (tokenize(query))


def tfidf_cal(all_queries,qnum):
    query_token = []
    tfidf_token = {}
    score = {}
    tfidf_score = {}
    document = read_documents()
    query = getquery(all_queries,qnum)
    query_token.append(query_tokenize(query))
    for d in document:
        for token in query_token:
            tfidf_token = (tfidf(token,d['id'],document))
            s = 0
        for m,n in tfidf_token.items():
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

