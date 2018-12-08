from __future__ import print_function
from __future__ import division

import re
import time
import itertools
import numpy as np

# For pretty-printing
import pandas as pd
from IPython.display import display, HTML

from . import constants

# Word processing functions
def canonicalize_digits(word):
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "") # remove thousands separator
    return word

def canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if (wordset != None) and (word in wordset): return word
        word = canonicalize_digits(word) # try to canonicalize numbers
    if (wordset == None) or (word in wordset):
        return word
    else:
        return constants.UNK_TOKEN

def canonicalize_words(words, **kw):
    return [canonicalize_word(word, **kw) for word in words]

def build_vocab(corpus, V=10000, **kw):
    from . import vocabulary
    if isinstance(corpus, list):
        token_feed = (canonicalize_word(w) for w in corpus)
        vocab = vocabulary.Vocabulary(token_feed, size=V, **kw)
    else:
        token_feed = (canonicalize_word(w) for w in corpus.words())
        vocab = vocabulary.Vocabulary(token_feed, size=V, **kw)

    print("Vocabulary: {:,} types".format(vocab.size))
    return vocab

def preprocess_sentences(sentences, vocab, use_eos=False, emit_ids=True,
                         progressbar=lambda l:l):
    """Preprocess sentences by canonicalizing and mapping to ids.

    Args:
      sentences ( list(list(string)) ): input sentences
      vocab: Vocabulary object, already initialized
      use_eos: if true, will add </s> token to end of sentence.
      emit_ids: if true, will emit as ids. Otherwise, will be preprocessed
          tokens.
      progressbar: (optional) progress bar to wrap iterator.

    Returns:
      ids ( array(int) ): flattened array of sentences, including boundary <s>
      tokens.
    """
    # Add sentence boundaries, canonicalize, and handle unknowns
    word_preproc = lambda w: canonicalize_word(w, wordset=vocab.word_to_id)
    ret = []
    for s in progressbar(sentences):
        canonical_words = vocab.pad_sentence(list(map(word_preproc, s)),
                                             use_eos=use_eos)
        ret.extend(vocab.words_to_ids(canonical_words) if emit_ids else
                   canonical_words)
    if not use_eos:  # add additional <s> to end if needed
        ret.append(vocab.START_ID if emit_ids else vocab.START_TOKEN)
    return ret


def rnnlm_batch_generator(ids, batch_size, max_time,labels):
    """Convert ids to data-matrix form for RNN language modeling."""
    # Clip to multiple of max_time for convenience
   # clip_len = ((len(ids)-1) // batch_size) * batch_size
    input_w = ids     # current word
    target_y = labels # 1 positive or 0 negative
    # Reshape so we can select columns
    input_w = input_w.reshape([-1,max_time])
    target_y = target_y
    # Yield batches
    for i in range(0, input_w.shape[0],batch_size):
        yield input_w[i:i+batch_size], target_y[i:i+batch_size]


def batch_generator(data, batch_size):
    """Generate minibatches from data.

    Args:
      data: array-like, supporting slicing along first dimension
      batch_size: int, batch size

    Yields:
      minibatches of maximum size batch_size
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]
