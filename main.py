import joblib
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)
import razdel
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import naive_bayes
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import time

stop_words = nltk.corpus.stopwords.words('russian')
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
stemmer = nltk.stem.snowball.RussianStemmer(ignore_stopwords=True)


def preprocess_text_lem(text):
    doc = Doc(text)
    doc.segment(segmenter) # токенизация
    doc.tag_morph(morph_tagger) # морфологические метки
    for token in doc.tokens:
        if token.text not in stop_words and token.text.isalpha():
            token.lemmatize(morph_vocab) # лемматизация
    return ' '.join([token.lemma for token in doc.tokens if token.lemma is not None])


def preprocess_text_stem(text):
    tokens = list(razdel.tokenize(text.lower()))  # токенизация и приведение к нижнему регистру
    tokens = [token.text for token in tokens if token.text.isalpha() and token.text not in stop_words]  # удаление стоп-слов и символов
    tokens = [stemmer.stem(token) for token in tokens]  # стемминг
    return ' '.join(tokens)


def vectorization(texts, labels):
    result_vectors = []
    for text in texts:
        word_vectors = []
        for word in segmenter.tokenize(text):
            if word.text in emb:
                word_vectors.append(emb[word.text])
        if len(word_vectors) == 0:
            labels = labels.drop(texts[texts == text].index.values[0])
            texts = texts.drop(texts[texts == text].index.values[0])
            continue
        result_vectors.append(np.mean(word_vectors, axis=0))

    return result_vectors, labels


def main():
    # считываем csv файл с данными
    data = pd.read_csv(r'./train_data/updated_file.csv', header=0, encoding='cp1251', sep=';')

    # предобработка и удаление дупликатов
    data['text'] = data['text'].apply(preprocess_text_lem) # Лемматизация обязательна для семантической векторизации
    #data['text'] = data['text'].apply(preprocess_text_stem) # стемминг
    #data = data.drop_duplicates()

    # векторизация
    #X, y = vectorization(data['text'], data['label']) # семантическая векторизация
    #vectorizer = TfidfVectorizer(use_idf=False) # относительная частота
    vectorizer = TfidfVectorizer() # tf-idf
    vectorizer.fit(data['text']) # составление словаря уникальных слов для векторизатора
    X, y = vectorizer.transform(data['text']), data['label']

    acc = [0, 0, 0]
    fit_time = [0, 0, 0]
    clfs = [None, None, None]
    iter = 200

    for i in range(iter):
        # разбиваем на обучающую и контрольную выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        # объекты модели и их обучение с замерением
        clfs = [svm.LinearSVC(C=0.5),
                svm.SVC(C=5, gamma=0.15),
                naive_bayes.MultinomialNB(force_alpha=True, fit_prior=True, alpha=0.3)] # Если мешок слов или tf-idf, то вместо Gaussian надо использовать Multinomial
        for i in range(len(clfs)):
            start_time = time.time()
            clfs[i].fit(X_train, y_train)
            fit_time[i] += time.time() - start_time
            acc[i] += clfs[i].score(X_test, y_test)

    for i in range(len(clfs)):
        print(f"{i} {acc[i] / iter} {fit_time[i] / iter}")



if __name__ == "__main__":
    main()