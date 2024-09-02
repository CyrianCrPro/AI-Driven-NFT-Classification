# Automated creation of categories

import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

data_json = '''
{"data":{"generativeTokens":[
    {"tags":["AI","AIartwork","AIart","AI art","2d","Paint","artificial intelligence","AI_TezoArt","Abstract","art","tezosart","tezosnft","tezos","tezosnfts","nft"],"author":{"name":"AITezoArt"}},
    {"tags":["Generative Art","Digital Art","Child","Childhood","Charity","Game","Toy","Colorful","Marbles","Art","Contemporary","RecollectionArts","Nostalgic","Fine Arts","Interior Design"],"author":{"name":null}},
    {"tags":["generative","animation","png","sliced","time"],"author":{"name":null}},
    {"tags":["generative","genart","abstract","p5js","noise","perlin","randomness","colors","contours","uniray"],"author":{"name":"Uniray"}},
    {"tags":["explosion","pop art","hongkongers","halftone","eruption","upheaval","tranquility","forthcoming","diaspora","immigrate","BNO"],"author":{"name":null}},
    {"tags":["interactive","abstract","cubes","infinite"],"author":{"name":null}},
    {"tags":["onchainsummer2024","webgl","dreams","polygons","triangles"],"author":{"name":null}},
    {"tags":["code","creative","generative","waves","bw","ferdoropeza"],"author":{"name":null}},
    {"tags":["deterioration art","architecture","nature","animation","japan","p5js"],"author":{"name":"Asahamiz"}},
    {"tags":[],"author":{"name":null}},
    {"tags":["colors","art","picture","nft","GT","unsleeping"],"author":{"name":null}},
    {"tags":["geometric","abstract","art","shapes","colors"],"author":{"name":null}},
    {"tags":["art","generative","fxhash","tezos","random","pattern","geometry","pixel","color"],"author":{"name":null}},
    {"tags":["creative coding","abstract","p5js","tezos","xtz","data"],"author":{"name":"aliasrubytuesday"}},
    {"tags":["geometric","abstract","art","shapes","colors"],"author":{"name":"jrcart.tez"}},
    {"tags":["lines","javascript","genart","colors"],"author":{"name":"RosbelDev"}},
    {"tags":[],"author":{"name":null}},
    {"tags":[],"author":{"name":null}},
    {"tags":["generative art","layered","ai","animation","video","symbol","layers","frostxhash","hobo","homeless","disturbed","florida man","lost","poetry","rambler","madman","crazy","amsterdam","utrecht","den haag","haarlem","arnhem","netherlands","city","neighborhood","clairvoyant","prescient","supernatural","prophet"],"author":{"name":"Plastic Tolstoy"}},
    {"tags":["creative coding","abstract","p5js","tezos","xtz","data"],"author":{"name":"aliasrubytuesday"}}
]}}
'''

data = json.loads(data_json)
tags_list = [tag for item in data['data']['generativeTokens'] for tag in item['tags']]
unique_tags = list(set(tags.lower() for tags in tags_list))

tokenized_tags = [word_tokenize(tag) for tag in unique_tags]

# Model training
model_w2v = Word2Vec(tokenized_tags, vector_size=100, window=5, min_count=1, workers=4)

# Creation of tag vectors using the mean of word vectors
def tag_vector(tag):
    words = word_tokenize(tag.lower())
    return np.mean([model_w2v.wv[word] for word in words if word in model_w2v.wv], axis=0)

tag_vectors = np.array([tag_vector(tag) for tag in unique_tags if tag_vector(tag) is not None])

# Clustering using K-means
kmeans = KMeans(n_clusters=8, random_state=42)
kmeans.fit(tag_vectors)
labels = kmeans.labels_

tag_clusters = {tag: label for tag, label in zip(unique_tags, labels)}

for tag, cluster in tag_clusters.items():
    print(f"Tag: {tag} -> Cluster: {cluster}")