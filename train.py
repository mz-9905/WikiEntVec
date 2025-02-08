"""
2025-02-09 update
    - update for gensim 4.x
    - add save_model option to save model in .model format
"""

import re
import argparse
from pathlib import Path

import logzero
from logzero import logger
from gensim.models.word2vec import LineSentence, Word2Vec


logger_word2vec = logzero.setup_logger(name='gensim.models.word2vec')
logger_base_any2vec = logzero.setup_logger(name='gensim.models.base_any2vec')

regex_entity = re.compile(r'##[^#]+?##')


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    word_vectors_file = output_dir / 'word_vectors.txt'
    entity_vectors_file = output_dir / 'entity_vectors.txt'
    all_vectors_file = output_dir / 'all_vectors.txt'
    model_file = output_dir / 'word2vec.model'

    logger.info('training the model')
    model = Word2Vec(sentences=LineSentence(args.corpus_file),
                     vector_size=args.embed_size,
                     window=args.window_size,
                     negative=args.sample_size,
                     min_count=args.min_count,
                     workers=args.workers,
                     sg=1,
                     hs=0,
                     epochs=args.epoch)

    word_vocab_size = 0
    entity_vocab_size = 0

    for token in model.wv.index_to_key:
        if regex_entity.match(token):
            entity_vocab_size += 1
        else:
            word_vocab_size += 1

    total_vocab_size = word_vocab_size + entity_vocab_size
    logger.info(f'word vocabulary size: {word_vocab_size}')
    logger.info(f'entity vocabulary size: {entity_vocab_size}')
    logger.info(f'total vocabulary size: {total_vocab_size}')

    logger.info('writing word/entity vectors to files')
    with open(word_vectors_file, 'w') as fo_word, \
         open(entity_vectors_file, 'w') as fo_entity, \
         open(all_vectors_file, 'w') as fo_all:

        # write word2vec headers to each file
        print(word_vocab_size, args.embed_size, file=fo_word)
        print(entity_vocab_size, args.embed_size, file=fo_entity)
        print(total_vocab_size, args.embed_size, file=fo_all)

        # write tokens and vectors
        for token in sorted(model.wv.index_to_key, key=lambda t: -model.wv.get_vecattr(t, 'count')):
            vector = model.wv[token]

            if regex_entity.match(token):
                print(token[2:-2], *vector, file=fo_entity)
            else:
                print(token, *vector, file=fo_word)

            print(token, *vector, file=fo_all)

    # save the model
    if args.save_model:
        logger.info(f'saving model to {model_file}')
        model.save(model_file.as_posix())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_file', type=str, required=True,
        help='Corpus file (.txt)')
    parser.add_argument('--output_dir', type=str, required=True,
        help='Output directory to save embedding files')
    parser.add_argument('--embed_size', type=int, default=100,
        help='Dimensionality of the word/entity vectors [100]')
    parser.add_argument('--window_size', type=int, default=5,
        help='Maximum distance between the current and '
             'predicted word within a sentence [5]')
    parser.add_argument('--sample_size', type=int, default=5,
        help='Number of negative samples [5]')
    parser.add_argument('--min_count', type=int, default=5,
        help='Ignores all words/entities with total frequency lower than this [5]')
    parser.add_argument('--epoch', type=int, default=5,
        help='number of training epochs [5]')
    parser.add_argument('--workers', type=int, default=2,
        help='Use these many worker threads to train the model [2]')
    parser.add_argument('--save_model', action='store_true',
        help='whether to save the trained model in .model format')
    args = parser.parse_args()
    main(args)
