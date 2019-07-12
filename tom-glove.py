import pprint
import json
from glove import Glove
from glove import Corpus
from nltk.corpus import stopwords 
from glob import glob
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def isNumber(string_to_check):
    try:
        int(string_to_check)
        return True
    except ValueError:
        return False

def read_corpus(filepattern):

    delchars = [chr(c) for c in range(256)]
    delchars = [x for x in delchars if not x.isalnum()]
    delchars.remove(' ')
    delchars = dict.fromkeys(delchars)
    delchars['\xa0'] = ' '
    delchars['\n'] = ' '
    delchars['“']= None
    delchars['”']= None    
    delchars["'"]= None
    delchars["‘"]= None
    delchars["’"]= None
    
    delchars = str.maketrans(delchars)

    try:
        stop_words = set(stopwords.words('english')) 
    except LookupError:
        import nltk
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english')) 

    filenames = glob(filepattern)
    for filename in filenames:
        print(f'Reading data from f{filename}')
        tom_data = json.load(open(filename, 'r'))
        for article in tom_data:
            article_body = article['article_body']            
            word_tokens = article_body.lower().translate(delchars).split(' ')
            yield [w for w in word_tokens if ( (not w in stop_words) and (not len(w) <= 4) )]  

if __name__ == '__main__':      

    want_TSNE = False
    want_GRAPH = True
    
    # Build the corpus dictionary and the cooccurrence matrix.
    print('Pre-processing corpus')

    filepattern = 'tom_articles_1*.json'

    corpus_model = Corpus()
    corpus_model.fit(read_corpus(filepattern) , window=5)
    corpus_model.save('corpus.model')
    
    print('Dict size: %s' % len(corpus_model.dictionary))
    print('Collocations: %s' % corpus_model.matrix.nnz)  

    print('Training the GloVe model')

    glove = Glove(no_components=50, learning_rate=0.05)
    glove.fit(corpus_model.matrix, epochs=100,
                no_threads=4, verbose=True)
    glove.add_dictionary(corpus_model.dictionary)

    glove.save('glove.model')

    if want_TSNE:
        print("Starting TSNE...")
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
        tsne_pca_results = tsne.fit_transform(glove.word_vectors[:])

        word = []
        tsne_1 = []
        tsne_2 = []
        for (w, v) in zip(glove.dictionary.keys(), tsne_pca_results):
            print(f'{w}: {v}')
            word.append(w)
            tsne_1.append(v[0])
            tsne_2.append(v[1])

        d = {
            'word': word,
            'tsne-one': tsne_1,
            'tsne-two': tsne_2
        }

        df = pd.DataFrame(data=d)


    if want_GRAPH:
        graph_data = {}
        graph_data['nodes']=[]
        graph_data['edges']=[]
        for word in glove.dictionary.keys():
            graph_data['nodes'].append({
                'data': {
                    'id': word,
                    'label': word
                }
            })

        connections_done = set()

        with open('tom_graph.sif', 'w') as result_file:
            for word in glove.dictionary.keys():
                if word == "":
                    continue
                if isNumber(word):
                    continue

                similar_words = []

                for similar_word_data in glove.most_similar(word, number=5):
                    similar_word = similar_word_data[0]

                    if similar_word == "":
                        continue
                    if isNumber(similar_word):
                        continue

                    weight = similar_word_data[1]

                    edge_id = [word, similar_word]
                    edge_id.sort()
                    edge_id = '-'.join(edge_id)
                    if edge_id in connections_done:
                        continue

                    similar_words.append(similar_word)

                similar_words = ' '.join(similar_words)
                result_file.write(f'{word} rel {similar_words}\n')
        
        


    # for (word, vector) in zip(glove.dictionary.keys(), glove.word_vectors[:]):
    #     print (f'{word} : {vector}')
    print(glove.word_vectors[glove.dictionary['muscat']])
    pprint.pprint(glove.most_similar('muscat', number=5))