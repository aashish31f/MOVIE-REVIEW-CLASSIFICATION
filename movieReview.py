
import nltk
import random
from nltk.corpus import movie_reviews

#creating a list of tuples
documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids():
        documents.append((list(movie_reviews.words(fileid)),category))

#shuffle the documents
random.shuffle(documents)

print(documents[1])

#normalizing the dataset
all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())
    
#finding the frequency distribution of words
all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))
print(all_words["love"])

#limit the words
word_features = list(all_words.keys())[:3000]

#find features
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features :
        features[w] = w in words 
    return features


featureSets = [(find_features(rev),category)
                    for (rev,category) in documents]


#split the data into training and test sets
training_set = featureSets[:1900]
testing_set = featureSets[1900:]

#training the classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)

#test the accuracy
print("accuracy : ",(nltk.classify.accuracy(classifier , training_set))*100 )


classifier.show_most_informative_features(15)








