# Student name: Eric Li
# Student number: 1007654307
# UTORid: ID

'''
This code is provided solely for the personal and private use of students
taking the CSC485H/2501H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Samarendra Dash, Zixin Zhao, Jinman Zhao, Jingcheng Niu, Zhewei Sun

All of the files in this directory and all subdirectories are:
Copyright (c) 2024 University of Toronto
'''

"""Statistical modelling/parsing classes"""

import sys
from itertools import islice
from pathlib import Path
from sys import stdout

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
import einops
from tqdm import tqdm

from data import score_arcs
from q1_parse import minibatchParse

class ParserModel(nn.Module):
    """
    Implements a feedforward neural network with an embedding layer and single
    hidden layer. This network will predict which transition should be applied
    to a given partial parse state.
    """
    def createEmbeddings(self, wordEmbeddings: torch.Tensor) -> None:
        """Create embeddings that map word, tag, and deprels to vectors

        Args:
            wordEmbeddings:
                torch.Tensor of shape (n_word_ids, embed_size) representing
                matrix of pre-trained word embeddings

        Embedding layers convert sparse ID representations to dense vector
        representations.
         - Create 3 embedding layers using nn.Embedding, one for each of
           the input types:
           - The word embedding layer must be initialized with the value of the
             argument wordEmbeddings, so you will want to create it using
             nn.Embedding.from_pretrained(...). Make sure not to freeze the
             embeddings!
           - You don't need to do anything special for initializing the other
             two embedding layers, so use nn.Embedding(...) for them.
         - The relevant values for the number of embeddings for each type can
           be found in {n_word_ids, n_tag_ids, n_deprel_ids}.
         - Assign the layers to self as attributes:
               self.wordEmbedding
               self.tagEmbedding
               self.dependencyRelationEmbedding
           (Don't use different variable names!)
        """
        print('Creating embeddings11111...')
        self.wordEmbedding = nn.Embedding.from_pretrained(wordEmbeddings, freeze=False)
        print('Creating embeddings22222...')
        self.tagEmbedding = nn.Embedding(self.config.n_tag_ids, self.config.embed_size)
        print('Creating embeddings33333...')
        self.dependencyRelationEmbedding = nn.Embedding(self.config.n_deprel_ids, self.config.embed_size)
        print('Creating embeddings44444...')
        

    def createNetLayers(self) -> None:
        """Create layer weights and biases for this neural network

        Our neural network computes predictions from the embedded input
        using a single hidden layer as well as an output layer. This method
        creates the hidden and output layers, including their weights and
        biases (but PyTorch will manage the weights and biases; you will not
        need to access them yourself). Note that the layers will only compute
        the result of the multiplication and addition (i.e., no activation
        function is applied, so the hidden layer will not apply the ReLu
        function).

         - Create the two layers mentioned above using nn.Linear. You will need
           to fill in the correct sizes for the nn.Linear(...) calls. Keep in mind
           the layer sizes:
               input layer (x): N * embed_size
               hidden layer (h): hidden_size
               output layer (pred): n_classes
           where N = n_word_features + n_tag_features + n_deprel_features
         - Assign the two layers to self as attributes:
               self.hiddenLayer
               self.outputLayer
           (Don't use different variable names!)

        nn.Linear will take care of randomly initializing the weight and bias
        tensors automatically, so that's all that is to be done here.
        """
        N = (self.config.n_word_features + 
             self.config.n_tag_features + 
             self.config.n_deprel_features)
        
        x = N * self.config.embed_size
        h = self.config.hidden_size
        c = self.config.n_classes

        self.hiddenLayer = nn.Linear(x, h)
        self.outputLayer = nn.Linear(h, c)
        

    def reshapeEmbedded(self, inputBatch: torch.Tensor) -> torch.Tensor:
        """Reshape an embedded input to combine the various embedded features

        Remember that we use various features based on the parser's state for
        our classifier, such as word on the top of the stack, next word in the
        buffer, etc. Each feature (such as a word) has its own embedding. But
        we will not want to keep the features separate for the classifier, so
        we must merge them all together. This method takes a tensor with
        separated embeddings for each feature and reshapes it accordingly.

        Args:
            inputBatch:
                torch.Tensor of dtype float and shape (B, N, embed_size)
                where B is the batch_size and N is one of {n_word_features,
                n_tag_features, n_deprel_features}.
        Returns:
            reshapedBatch:
                torch.Tensor of dtype float and shape (B, N * embed_size).

         - Reshape the embedded batch tensor into the specified shape using
           torch.reshape. You may find the value of -1 handy for one of the
           shape dimensions; see the docs for torch.reshape for what it does.
           You may alternatively use the inputBatch.view(...) or
           inputBatch.reshape(...) methods if you prefer.
        """
        # *** ENTER YOUR CODE BELOW *** #
        B = inputBatch.shape[0]
        reshapedBatch = inputBatch.reshape(B, -1)
        return reshapedBatch

    def concatEmbeddings(self, wordIdBatch: torch.Tensor,
                              tagIdBatch: torch.Tensor,
                              dependencyRelationIdBatch: torch.Tensor) -> torch.Tensor:
        """Get, reshape, and concatenate word, tag, and deprel embeddings

        Recall that in our neural network, we concatenate the word, tag, and
        deprel embeddings to use as input for our hidden layer. This method
        retrieves all word, tag, and deprel embeddings and concatenates them
        together.

        Args:
            wordIdBatch:
                torch.Tensor of dtype int64 and shape (B, n_word_features)
            tagIdBatch:
                torch.Tensor of dtype int64 and shape (B, n_tag_features)
            dependencyRelationIdBatch:
                torch.Tensor of dtype int64 and shape (B, n_deprel_features)
            where B is the batch size
        Returns:
            reshaped:
                torch.Tensor of dtype float and shape (B, N * embed_size) where
                N = n_word_features + n_tag_features + n_deprel_features

         - Look up the embeddings for the IDs represented by the wordIdBatch,
           tagIdBatch, and dependencyRelationIdBatch tensors using the embedding layers
           you defined in self.create_embeddings. (You do not need to call that
           method from this one; that is done automatically for you elsewhere.)
         - Use the self.reshape_embedded method you implemented on each of the
           resulting embedded batch tensors from the previous step.
         - Concatenate the reshaped embedded inputs together using torch.cat to
           get the necessary shape specified above and return the result.
        """
        word_embedded = self.wordEmbedding(wordIdBatch) 
        tag_embedded = self.tagEmbedding(tagIdBatch)
        deprel_embedded = self.dependencyRelationEmbedding(dependencyRelationIdBatch)

        # Reshape embeddings to (B, N * embed_size) each
        word_flat = self.reshapeEmbedded(word_embedded)
        tag_flat = self.reshapeEmbedded(tag_embedded)
        deprel_flat = self.reshapeEmbedded(deprel_embedded)

        # Concatenate along the feature dimension
        reshaped = torch.cat([word_flat, tag_flat, deprel_flat], dim=1)

        return reshaped

    def forward(self,
                wordIdBatch: np.array,
                tagIdBatch: np.array,
                dependencyRelationIdBatch: np.array) -> torch.Tensor:
        """Compute the forward pass of the single-layer neural network

        In our single-hidden-layer neural network, our predictions are computed
        as follows from the concatenated embedded input x:
          1. x is passed through the linear hidden layer to produce h.
          2. Dropout is applied to h to produce h_drop.
          3. h_drop is passed through the output layer to produce pred.
        This method computes pred from the x with the help of the setup done by
        the other methods in this class. Note that, compared to the assignment
        handout, we've added dropout to the hidden layer and we will not be
        applying the softmax activation at all in this model code. See the
        crossEntropyLoss method if you are curious as to why.

        Args:
            wordIdBatch:
                np.array of dtype int64 and shape (B, n_word_features)
            tagIdBatch:
                np.array of dtype int64 and shape (B, n_tag_features)
            dependencyRelationIdBatch:
                np.array of dtype int64 and shape (B, n_deprel_features)
        Returns:
            pred: torch.Tensor of shape (B, n_classes)

        - Use self.hiddenLayer that you defined in self.create_net_layers to
          compute the pre-activation hidden layer values.
        - Use the torch.relu function to activate the result of
          the previous step and then use the torch.dropout
          function to apply dropout with the appropriate dropout rate. You will use
          these function calls: torch.relu(...) and torch.dropout(...).
          - Remember that dropout behaves differently when training vs. when
          evaluating. The torch.dropout function reflects this via its arguments.
          You can use self.training to indicate whether or not the model is
          currently being trained.
        - Finally, use self.outputLayer to compute the model outputs from the
          result of the previous step.
        """
        x = self.concatEmbeddings(torch.tensor(np.array(wordIdBatch)),
                                       torch.tensor(np.array(tagIdBatch)),
                                       torch.tensor(np.array(dependencyRelationIdBatch)))

        h = self.hiddenLayer(x)
        h_act = torch.relu(h)
        h_drop = torch.dropout(h_act, p=self.config.dropout, train=self.training)
        pred = self.outputLayer(h_drop)

        return pred

    def crossEntropyLoss(self, predictionBatch: torch.Tensor,
                 labelBatch: torch.Tensor) -> torch.Tensor:
        """Calculate the value of the loss function

        In this case we are using cross entropy loss. The loss will be averaged
        over all examples in the current minibatch. This file already imports
        the function cross_entropy for you (line 14), so you can directly use
        `cross_entropy` to compute the loss. Note that we are not applying softmax
        to predictionBatch, since cross_entropy handles that in a more efficient way.
        Excluding the softmax in predictions won't change the expected transition.
        (Convince yourself of this.)

        Args:
            predictionBatch:
                A torch.Tensor of shape (batch_size, n_classes) and dtype float
                containing the logits of the neural network, i.e., the output
                predictions of the neural network without the softmax
                activation.
            labelBatch:
                A torch.Tensor of shape (batch_size,) and dtype int64
                containing the ground truth class labels.
        Returns:
            loss: A 0d tensor (scalar) of dtype float
        """
        # *** ENTER YOUR CODE BELOW *** #
        loss = cross_entropy(predictionBatch, labelBatch)
        return loss

    def add_optimizer(self):
        """Sets up the optimizer.

        Creates an instance of the Adam optimizer and sets it as an attribute
        for this class.
        """
        self.optimizer = torch.optim.Adam(self.parameters(), self.config.lr)

    def _fit_batch(self, wordIdBatch, tagIdBatch, dependencyRelationIdBatch,
                   labelBatch):
        self.optimizer.zero_grad()
        predictionBatch = self(wordIdBatch, tagIdBatch, dependencyRelationIdBatch)
        loss = self.crossEntropyLoss(predictionBatch, torch.tensor(labelBatch).argmax(-1))
        loss.backward()

        self.optimizer.step()

        return loss

    def fit_epoch(self, train_data, epoch, trn_progbar, batch_size=None):
        """Fit on training data for an epoch"""
        self.train()
        desc = 'Epoch %d/%d' % (epoch + 1, self.config.n_epochs)
        total = len(train_data) * batch_size if batch_size else len(train_data)
        bar_fmt = '{l_bar}{bar}| [{elapsed}<{remaining}{postfix}]'
        with tqdm(desc=desc, total=total, leave=False, miniters=1, unit='ex',
                  unit_scale=True, bar_format=bar_fmt, position=1) as progbar:
            trn_loss = 0
            trn_done = 0
            for ((wordIdBatch, tagIdBatch, dependencyRelationIdBatch),
                 labelBatch) in train_data:

                loss = self._fit_batch(wordIdBatch, tagIdBatch,
                                       dependencyRelationIdBatch, labelBatch)
                trn_loss += loss.item() * wordIdBatch.shape[0]
                trn_done += wordIdBatch.shape[0]
                progbar.set_postfix({'loss': '%.3g' % (trn_loss / trn_done)})
                progbar.update(wordIdBatch.shape[0])
                trn_progbar.update(wordIdBatch.shape[0] / total)
        return trn_loss / trn_done

    def predict(self, partial_parses):
        """Use this model to predict the next transitions/deprels of pps"""
        self.eval()
        feats = self.transducer.pps2feats(partial_parses)
        td_vecs = self(*feats).cpu().detach().numpy()
        preds = [
            self.transducer.td_vec2trans_deprel(td_vec) for td_vec in td_vecs]
        return preds

    def evaluate(self, sentences, ex_arcs):
        """LAS on either training or test sets"""
        act_arcs = minibatchParse(sentences, self, self.config.batch_size)
        ex_arcs = tuple([(a[0], a[1],
                          self.transducer.id2deprel[a[2]]) for a in pp]
                        for pp in ex_arcs)
        stdout.flush()
        return score_arcs(act_arcs, ex_arcs)

    def __init__(self, transducer, config, wordEmbeddings):
        self.transducer = transducer
        self.config = config

        super().__init__()

        self.createEmbeddings(torch.from_numpy(wordEmbeddings))
        self.createNetLayers()

        self.add_optimizer()