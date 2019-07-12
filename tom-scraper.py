import requests
from html.parser import HTMLParser
import json
import pickle

class TomHTMLParser(HTMLParser):

    shouldParse = False
    results = False

    def handle_starttag(self, tag, attrs):
        if tag == 'script':
            if len(attrs) > 0:
                for attr in attrs:
                    if attr[0]=='type' and attr[1]=='application/ld+json':
                        self.shouldParse = True

    def handle_data(self, data):
        if self.shouldParse == True:
            if data.__contains__('"@type":"NewsArticle"'):
                self.results = json.loads(data)
            self.shouldParse = False

    def get_results(self):
        return self.results

class TomArticleHTMLParser(HTMLParser):

    shouldParse = False
    results = False
    
    def handle_starttag(self, tag, attrs):
        if tag == 'script':
            if len(attrs) > 0:
                for attr in attrs:
                    if attr[0]=='id' and attr[1]=='article-ld':
                        self.shouldParse = True

    def handle_data(self, data):
        if self.shouldParse == True:
            if data.__contains__('"@type":"NewsArticle"'):
                self.results = json.loads(data)
            self.shouldParse = False

    def get_results(self):
        return self.results

    def get_article_data(self):
        if self.results:
            article = False
            if '@graph' in self.results:
                article = {}
                article_data = self.results['@graph'][0]
                article['keywords'] = article_data['keywords']
                article['author'] = article_data['author']['name']
                article['article_body'] = article_data['articleBody']
                try:
                    article['comment_count'] = article_data['commentCount']
                except:
                    article['comment_count'] = 0
                

            return article


all_articles = []
article_counter = 0
for page_num in range(1, 211):

    print(f"Parsing page: {page_num}")
    parser = TomHTMLParser()
    response = requests.get(f'https://timesofmalta.com/articles/listing/national/page:{page_num}')
    parser.feed(response.text)
    result = parser.get_results()
    if result:
        for article_link in result['@graph']:
            article_response = requests.get(article_link['@id'])
            article_parser = TomArticleHTMLParser()
            article_parser.feed(article_response.text)
            article = article_parser.get_article_data()
            article['id'] = article_counter
            if article:
                article_counter += 1
                print(f'Parsed article: {article_counter}')
                all_articles.append(article)
    else:
        print("No results found!")

    # dump every 100 articles
    if len(all_articles) >= 100:
        filename = f'tom_articles_{article_counter}.json'
        print(f'Dumping articles to disk: {filename}')
        json.dump(all_articles, open(filename, 'w'))
        all_articles = []

filename = f'tom_articles_{article_counter}.json'
print(f'Dumping articles to disk: {filename}')
json.dump(all_articles, open(filename, 'w'))