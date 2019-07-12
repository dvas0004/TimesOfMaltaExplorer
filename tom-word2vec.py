import pprint
import json
from gensim.models import Word2Vec
from gensim import corpora
from nltk.corpus import stopwords 

from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from collections import OrderedDict

memoize_cache = {}

def isNumber(string_to_check):
    try:
        int(string_to_check)
        return True
    except ValueError:
        return False

def getArticleById(id, filepattern):
    filenames = glob(filepattern)
    
    for filename in filenames:
        tom_data = json.load(open(filename, 'r'))
        for article in tom_data:
            if article['id'] == id:
                return article

def sentence_preprocess(sentence):
    global memoize_cache
    if not 'delchars' in memoize_cache:
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
        memoize_cache['delchars'] = delchars
    else:
        delchars = memoize_cache['delchars']

    if not 'stopwords' in memoize_cache:
        try:
            stop_words = set(stopwords.words('english')) 
        except LookupError:
            import nltk
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english')) 

        stop_words.update(["a's" , "able" , "about" , "above" , "according" , "accordingly" , "across" , "actually" , "after" , "afterwards" , "again" , "against" , "ain't" , "all" , "allow" , "allows" , "almost" , "alone" , "along" , "already" , "also" , "although" , "always" , "am" , "among" , "amongst" , "an" , "and" , "another" , "any" , "anybody" , "anyhow" , "anyone" , "anything" , "anyway" , "anyways" , "anywhere" , "apart" , "appear" , "appreciate" , "appropriate" , "are" , "aren't" , "around" , "as" , "aside" , "ask" , "asking" , "associated" , "at" , "available" , "away" , "awfully" , "be" , "became" , "because" , "become" , "becomes" , "becoming" , "been" , "before" , "beforehand" , "behind" , "being" , "believe" , "below" , "beside" , "besides" , "best" , "better" , "between" , "beyond" , "both" , "brief" , "but" , "by" , "c'mon" , "c's" , "came" , "can" , "can't" , "cannot" , "cant" , "cause" , "causes" , "certain" , "certainly" , "changes" , "clearly" , "co" , "com" , "come" , "comes" , "concerning" , "consequently" , "consider" , "considering" , "contain" , "containing" , "contains" , "corresponding" , "could" , "couldn't" , "course" , "currently" , "definitely" , "described" , "despite" , "did" , "didn't" , "different" , "do" , "does" , "doesn't" , "doing" , "don't" , "done" , "down" , "downwards" , "during" , "each" , "edu" , "eg" , "eight" , "either" , "else" , "elsewhere" , "enough" , "entirely" , "especially" , "et" , "etc" , "even" , "ever" , "every" , "everybody" , "everyone" , "everything" , "everywhere" , "ex" , "exactly" , "example" , "except" , "far" , "few" , "fifth" , "first" , "five" , "followed" , "following" , "follows" , "for" , "former" , "formerly" , "forth" , "four" , "from" , "further" , "furthermore" , "get" , "gets" , "getting" , "given" , "gives" , "go" , "goes" , "going" , "gone" , "got" , "gotten" , "greetings" , "had" , "hadn't" , "happens" , "hardly" , "has" , "hasn't" , "have" , "haven't" , "having" , "he" , "he's" , "hello" , "help" , "hence" , "her" , "here" , "here's" , "hereafter" , "hereby" , "herein" , "hereupon" , "hers" , "herself" , "hi" , "him" , "himself" , "his" , "hither" , "hopefully" , "how" , "howbeit" , "however" , "i'd" , "i'll" , "i'm" , "i've" , "ie" , "if" , "ignored" , "immediate" , "in" , "inasmuch" , "inc" , "indeed" , "indicate" , "indicated" , "indicates" , "inner" , "insofar" , "instead" , "into" , "inward" , "is" , "isn't" , "it" , "it'd" , "it'll" , "it's" , "its" , "itself" , "just" , "keep" , "keeps" , "kept" , "know" , "known" , "knows" , "last" , "lately" , "later" , "latter" , "latterly" , "least" , "less" , "lest" , "let" , "let's" , "like" , "liked" , "likely" , "little" , "look" , "looking" , "looks" , "ltd" , "mainly" , "many" , "may" , "maybe" , "me" , "mean" , "meanwhile" , "merely" , "might" , "more" , "moreover" , "most" , "mostly" , "much" , "must" , "my" , "myself" , "name" , "namely" , "nd" , "near" , "nearly" , "necessary" , "need" , "needs" , "neither" , "never" , "nevertheless" , "new" , "next" , "nine" , "no" , "nobody" , "non" , "none" , "noone" , "nor" , "normally" , "not" , "nothing" , "novel" , "now" , "nowhere" , "obviously" , "of" , "off" , "often" , "oh" , "ok" , "okay" , "old" , "on" , "once" , "one" , "ones" , "only" , "onto" , "or" , "other" , "others" , "otherwise" , "ought" , "our" , "ours" , "ourselves" , "out" , "outside" , "over" , "overall" , "own" , "particular" , "particularly" , "per" , "perhaps" , "placed" , "please" , "plus" , "possible" , "presumably" , "probably" , "provides" , "que" , "quite" , "qv" , "rather" , "rd" , "re" , "really" , "reasonably" , "regarding" , "regardless" , "regards" , "relatively" , "respectively" , "right" , "said" , "same" , "saw" , "say" , "saying" , "says" , "second" , "secondly" , "see" , "seeing" , "seem" , "seemed" , "seeming" , "seems" , "seen" , "self" , "selves" , "sensible" , "sent" , "serious" , "seriously" , "seven" , "several" , "shall" , "she" , "should" , "shouldn't" , "since" , "six" , "so" , "some" , "somebody" , "somehow" , "someone" , "something" , "sometime" , "sometimes" , "somewhat" , "somewhere" , "soon" , "sorry" , "specified" , "specify" , "specifying" , "still" , "sub" , "such" , "sup" , "sure" , "t's" , "take" , "taken" , "tell" , "tends" , "th" , "than" , "thank" , "thanks" , "thanx" , "that" , "that's" , "thats" , "the" , "their" , "theirs" , "them" , "themselves" , "then" , "thence" , "there" , "there's" , "thereafter" , "thereby" , "therefore" , "therein" , "theres" , "thereupon" , "these" , "they" , "they'd" , "they'll" , "they're" , "they've" , "think" , "third" , "this" , "thorough" , "thoroughly" , "those" , "though" , "three" , "through" , "throughout" , "thru" , "thus" , "to" , "together" , "too" , "took" , "toward" , "towards" , "tried" , "tries" , "truly" , "try" , "trying" , "twice" , "two" , "un" , "under" , "unfortunately" , "unless" , "unlikely" , "until" , "unto" , "up" , "upon" , "us" , "use" , "used" , "useful" , "uses" , "using" , "usually" , "value" , "various" , "very" , "via" , "viz" , "vs" , "want" , "wants" , "was" , "wasn't" , "way" , "we" , "we'd" , "we'll" , "we're" , "we've" , "welcome" , "well" , "went" , "were" , "weren't" , "what" , "what's" , "whatever" , "when" , "whence" , "whenever" , "where" , "where's" , "whereafter" , "whereas" , "whereby" , "wherein" , "whereupon" , "wherever" , "whether" , "which" , "while" , "whither" , "who" , "who's" , "whoever" , "whole" , "whom" , "whose" , "why" , "will" , "willing" , "wish" , "with" , "within" , "without" , "won't" , "wonder" , "would" , "wouldn't" , "yes" , "yet" , "you" , "you'd" , "you'll" , "you're" , "you've" , "your" , "yours" , "yourself" , "yourselves" , "zero"])
        memoize_cache['stopwords'] = stop_words
    else:
        stop_words = memoize_cache['stopwords']

    word_tokens = sentence.lower().translate(delchars).split(' ')
    return [w for w in word_tokens if ( (not w in stop_words) and (not len(w) <= 4) )]

class MyCorpus():
    def __init__(self, filepattern):
        self.filepattern = filepattern

    def __iter__(self):
        filenames = glob(self.filepattern)
        for filename in filenames:
            print(f'Reading data from {filename}')
            tom_data = json.load(open(filename, 'r'))
            for article in tom_data:
                article_body = article['article_body']            
                yield sentence_preprocess(article_body)

    

if __name__ == '__main__':      

    want_GRAPH = True
    want_ARTICLES = True
    want_PCA = False

    filepattern = 'tom_articles_*.json'
    if len(sys.argv) > 1:
        article = getArticleById(int(sys.argv[1]), filepattern)
        pprint.pprint(article)
        sys.exit(0)
    
    # Build the corpus dictionary and the cooccurrence matrix.
    print('Pre-processing corpus')    
    word_corpus = MyCorpus(filepattern)
    
    print('Training the Word2Vec model')
    model = Word2Vec(word_corpus, min_count=10, window=5, workers=4, size=50, iter=100)


    if want_GRAPH:
        connections_done = set()
        wv = model.wv.index2word

        with open('tom_graph_word2vec.sif', 'w') as result_file:
            for word in wv:
                if word == "":
                    continue
                if isNumber(word):
                    continue

                similar_words = []

                for similar_word_data in model.similar_by_word(word, topn=10):
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
        
        

    # pprint.pprint(model.similar_by_word('muscat', topn=10))                    
    # pprint.pprint(model.similar_by_word('singleuse', topn=10))

    if want_ARTICLES:
        print('Starting article analysis...')
        filenames = glob(filepattern)
        article_results = []
        article_ids = []
        article_keywords = []

        for filename in filenames:
            print(f'Reading data from {filename}')
            tom_data = json.load(open(filename, 'r'))
            for article in tom_data:
                artID = article['id']
                artKeywords = article['keywords']
                print(f"Article ID: {artID}")
                processed_article = sentence_preprocess(article['article_body'])
                vectors = []
                for word in processed_article:
                    if word in model.wv.index2word:
                        vector = model.wv[word]
                        vectors.append(vector)
                average_vector = np.mean(vectors, axis=0)
                article_results.append(average_vector)
                article_ids.append(artID)
                article_keywords.append(artKeywords)

        article_results = np.array(article_results)
        print("Shape of article vectors: ", article_results.shape)
        print("Starting dimensionality reduction")
        if want_PCA:
            pca = PCA(n_components=2)
            dim_reduction_result = pca.fit_transform(article_results)
        else:
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            dim_reduction_result = tsne.fit_transform(article_results)

        df = pd.DataFrame()
        df['id'] = article_ids
        df['keywords'] = article_keywords
        df['dim-one'] = dim_reduction_result[:,0]
        df['dim-two'] = dim_reduction_result[:,1] 
        

        print("Generating plot")
        source = ColumnDataSource(
            data=dict(
                x=df['dim-one'],
                y=df['dim-two'],
                label=df['id'],
                keywords=df['keywords']
            )
        )
        TOOLS="crosshair,pan,wheel_zoom,box_zoom,reset,hover,previewsave"
        p = figure(title="ToM Articles Analysis", tools=TOOLS, plot_height=1000, plot_width=1200)

        # add a circle renderer with a size, color, and alpha
        p.circle('x', 'y', size=20, color="navy", alpha=0.5, source=source)

        hover =p.select(dict(type=HoverTool))
        hover.tooltips = OrderedDict([
            ("label", "$index"),
            ("keywords", "@keywords"),
        ])

        # show the results
        show(p)