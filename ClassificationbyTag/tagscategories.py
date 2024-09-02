# Manually creating categories

import spacy
import json

# Loading spaCy 
nlp = spacy.load('en_core_web_md')

# Create categories and their description
categories = {
    "Abstract": "abstract forms and shapes",
    "Nature": "natural elements like trees and animals",
    "Digital": "digital art including pixels and 3D animations",
    "Architecture": "buildings and infrastructure related to architecture",
    "Space": "space related event such as blackholes, planets or physics and wave"
}

# Categories to vectors
category_vectors = {cat: nlp(desc).vector for cat, desc in categories.items()}

# Exemple of json
data_json = '''
{"data":{"generativeTokens":[
    {"tags":["abstract", "colorful", "geometric shapes"]},
    {"tags":["forest", "trees", "wildlife"]},
    {"tags":["3D rendering", "digital painting"]},
    {"tags":["modern architecture", "skyscrapers"]}
]}}
'''

# Loading data
data = json.loads(data_json)
tags_list = [item['tags'] for item in data['data']['generativeTokens']]

def find_best_category(tag, category_vectors):
    tag_doc = nlp(tag)
    best_category, highest_similarity = None, -1
    for category, vector in category_vectors.items():
        similarity = tag_doc.similarity(nlp(category))
        if similarity > highest_similarity:
            best_category, highest_similarity = category, similarity
    return best_category

tag_to_category = {}
for tags in tags_list:
    for tag in tags:
        normalized_tag = tag.lower()
        best_category = find_best_category(normalized_tag, category_vectors)
        tag_to_category[normalized_tag] = best_category

print(json.dumps(tag_to_category, indent=4))
