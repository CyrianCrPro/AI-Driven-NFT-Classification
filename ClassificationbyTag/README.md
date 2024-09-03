# Classification by Tag

Categorize tags based on their semantic meanings.
Both codes focus on grouping or categorizing tags using vector representations to identify semantic relationships, though one does so through automated clustering and the other through predefined category matching.


## -- tagsCategories.ipynb --

Manually assigns categories to tags using predefined category descriptions and semantic similarity.

Loads a pre-trained spaCy model for natural language processing.
Creates a set of predefined categories with their descriptions, such as "Abstract," "Nature," "Digital," "Architecture," and "Space."
Each category description is converted into a vector using the model.
Loads JSON data (Obtained from the api) containing tags.
Calculates the similarity between a tag and each category vector using cosine similarity. The tag is assigned to the category with the highest similarity score.
Outputs a JSON mapping of each tag to its most similar category.


## -- tagsAuto.ipynb --

Automatically creates categories for a set of tags using word embeddings and clustering techniques.

Loads JSON data (Obtained from the api) and extracts all tags into a list. It ensures that all tags are unique and in lowercase.
Each unique tag is tokenized into words. A Word2Vec model is trained on these tokenized tags to create vector representations of the words.
For each tag, the code calculates a vector representation by averaging the vectors of the words in the tag using the trained model.
Applies K-Means clustering to the tag vectors, grouping the tags into 8 clusters based on their vector similarities.
Outputs the tags along with their corresponding cluster numbers, effectively categorizing the tags based on their semantic similarity.