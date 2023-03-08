import string
import numpy as np
import umap
import torch
from transformers import AutoModel, AutoTokenizer

class WordEmbedding(object):
    def __init__(self, 
            model_name='bert-large-cased', 
            seed=42, 
        ) -> None:
        self.chars = list(string.printable)
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_chars = self.embed_chars(self.chars)
        # print(self.chars)
        # print(len(self.chars))
        # print(self.embedding_chars.shape)

        # # Reduce the dimensionality of the embeddings using UMAP
        # self.reducer = umap.UMAP(
        #     n_components=ndim,
        #     n_neighbors=5, 
        #     min_dist=0.3, 
        #     metric='cosine', 
        #     random_state=seed
        # )
    def embed_texts(self, texts):
        # Tokenize the words
        tokens = self.tokenizer(texts, padding=True,
                                truncation=True, return_tensors='pt')

        # Get the embeddings for the tokens
        with torch.no_grad():
            outputs = self.model(**tokens)
            word_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return word_embeddings
    
    def embed_chars(self, chars):
        return self.embed_texts(chars)
    
    def embed_words(self, words):
        return self.embed_texts(words)

        # # Tokenize the words
        # tokens = self.tokenizer(
        #     words, 
        #     padding=True, 
        #     truncation=True, 
        #     return_tensors='pt'
        # )

        # # Get the embeddings for the tokens
        # with torch.no_grad():
        #     outputs = self.model(**tokens)
        #     word_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        # return word_embeddings
            
    def reduced_emb(self, embeddings):
        pass

    
def main():
    we = WordEmbedding()

if __name__=="__main__":
    main()