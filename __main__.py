import os
import gdown
import streamlit as st
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from typing import List


def drop_punctuation(s: str) -> str:
    return "".join(filter(str.isalpha, s))


def lower_tr(s: str) -> str:
    return s.replace("I", "ı").lower()


def is_nonstop_word(stop_words: set) -> bool:
    def is_nonstop(word: str) -> bool:
        return word not in stop_words

    return is_nonstop


def is_word_in_model(word_vectors: KeyedVectors) -> bool:
    def contains(s: str) -> bool:
        return s in word_vectors.key_to_index

    return contains


def vectorize_sentence(sentence: str, word_vectors: KeyedVectors, stop_words: set):
    words = map(drop_punctuation, sentence.split())
    nonstop_words = filter(is_nonstop_word(stop_words), words)
    lowercase_words = map(lower_tr, nonstop_words)
    found_words = list(filter(is_word_in_model(word_vectors), lowercase_words))
    if not found_words:
        return None
    vectors = word_vectors[found_words]
    average = np.mean(vectors, axis=0)
    return average


def is_sentence_understood(sentence_vector) -> bool:
    return sentence_vector is not None


def fetch_stop_words() -> set:
    url = "https://raw.githubusercontent.com/sgsinclair/trombone/master/src/main/resources/org/voyanttools/trombone/keywords/stop.tr.turkish-lucene.txt"
    data = pd.read_csv(url)
    stop_words = set(data[2:].iloc[:, 0].values)
    return stop_words


def download_model():
    output = "trmodel"
    if not os.path.isfile(output):
        gdown.download(id="1q1o2sGByIaUHd7vi5IX8KJEcJw329hgY", output=output)


def load_model() -> KeyedVectors:
    return KeyedVectors.load_word2vec_format("trmodel", binary=True)


def max_indexes(arr: np.array, count: int) -> np.array:
    return np.argpartition(arr, -count)[-count:]


def summarize_sentences(sentences: list, length: int,
                        word_vectors: KeyedVectors, stop_words: set) -> list:
    sentence_vectors = [vectorize_sentence(sentence, word_vectors, stop_words)
                        for sentence in sentences]
    sentence_vectors = list(filter(is_sentence_understood, sentence_vectors))
    similarity_matrix = cosine_similarity(sentence_vectors)
    score_vector = np.sum(similarity_matrix, axis=1)
    summary_len = min(length, len(sentences))
    summary_indexes = max_indexes(score_vector, summary_len)
    summary_indexes = np.sort(summary_indexes)
    summary = list(map(sentences.__getitem__, summary_indexes))
    return summary


def summarize_text(text: str, length: int,
                   word_vectors: KeyedVectors, stop_words: set) -> str:
    spaced_text = " ".join(text.split())
    sentence_sep = ". "
    sentences = spaced_text.split(sentence_sep)
    summary_sentences = summarize_sentences(
        sentences, length, word_vectors, stop_words)
    summary = sentence_sep.join(summary_sentences)
    return summary


def main():
    download_model()
    word_vectors = load_model()
    stop_words = fetch_stop_words()

    st.header("Turkish Summary Generator")
    text = st.text_area(label="Long boring text")
    length = st.number_input(label="Summary sentence count", value=2)

    try:
        summary = summarize_text(text, length, word_vectors, stop_words)
    except Exception as e:
        summary = ""
    
    st.text_area(label="Short summary text", value=summary, disabled=True)


if __name__ == "__main__":
    main()
