from __future__ import unicode_literals, print_function, division
import argparse
from data import readFile, prepareData
from utils import writeToFile, compute_bleu
from seq2seq import EncoderDecoder
import time
import random
import os


parser = argparse.ArgumentParser(description='Neural Machine Translation')

# train, test, and output location arguments
parser.add_argument('--train-file', type=str, default="eng-fra.train.small.txt",
                    help='input file for training (default: eng-fra.train.small.txt)')
parser.add_argument('--test-file', type=str, default="eng-fra.test.small.txt",
                    help='input file for evaluation (default: eng-fra.test.small.txt)')
parser.add_argument('--output-dir', type=str, default="results/",
                    help='output directory to save the model(default: results/)')

# hyperparameters
parser.add_argument('--max-length', type=int, default=60,
                    help='Maximum sequence length (default: 60)')
parser.add_argument('--tfr', type=float, default=0.5,
                    help='Teacher Forcing Ratio (default: 0.5)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning Rate (default: 0.01)')
parser.add_argument('--drop', type=float, default=0.1,
                    help='Dropout Probability (default: 0.1)')
parser.add_argument('--hidden-size', type=int, default=128,
                    help='Size of hidden layer (default: 128)')
parser.add_argument('--n-iters', type=int, default=10000,
                    help='Number of Iterations (default: 10000)')

parser.add_argument('--plot-every', type=int, default=100,
                    help='Plot after (default: 100)')
parser.add_argument('--print-every', type=int, default=1000,
                    help='Print after(default: 1000)')
parser.add_argument('--eval', default=False, action='store_true',
                    help='Run the model on test file')

# model architectures
parser.add_argument('--simple', default=False, action='store_true',
                    help='Run the simple decoder')
parser.add_argument('--bidirectional', default=False, action='store_true',
                    help='Run the bidirectional encoder')
parser.add_argument('--dot', default=False, action='store_true',
                    help='Run the Attention decoder with dot type')
"""
Models for part 3

To run multi-layer optimal model:       python3 main.py --multi --num-layers=<n>
"""
parser.add_argument('--char', default=False, action='store_true',
                    help='Run the character-based model')
parser.add_argument('--char-bleu', default=False, action='store_true',
                    help='Use a character-based BLEU metric')
parser.add_argument('--multi', default=False, action='store_true',
                    help='Run the Multi-layered Bidirectional Encoder with DotAttention Decoder')
parser.add_argument('--num-layers', type=int, default=8,
                    help='Number of layers in multi-layer model')
# note: Google NMT used 8 layers

def main():

    global args, max_length
    args = parser.parse_args()

    if args.eval:

        if not os.path.exists(args.output_dir):
            print("Output directory do not exists")
            exit(0)
        try:
            model = EncoderDecoder().load(args.output_dir)
            print("Model loaded successfully")
        except:
            print("The trained model could not be loaded...")
            exit()

        test_pairs = readFile(args.test_file)

        outputs = model.evaluatePairs(test_pairs,  rand=False, char=args.char)
        writeToFile(outputs, os.path.join(args.output_dir, "output.pkl"))
        reference = []
        hypothesis = []

        for (hyp, ref) in outputs:
            if args.char or args.char_bleu:
                reference.append([list(ref)])
                hypothesis.append(list(hyp))
            else:
                reference.append([ref.split(" ")])
                hypothesis.append(hyp.split(" "))

        bleu_score = compute_bleu(reference, hypothesis)
        print("Bleu Score: " + str(bleu_score))

        print(model.evaluateAndShowAttention(
            "L'anglais n'est pas facile pour nous.", char=args.char))
        print(model.evaluateAndShowAttention(
            "J'ai dit que l'anglais est facile.", char=args.char))
        print(model.evaluateAndShowAttention(
            "Je n'ai pas dit que l'anglais est une langue facile.", char=args.char))
        print(model.evaluateAndShowAttention(
            "Je fais un blocage sur l'anglais.", char=args.char))

    else:
        input_lang, output_lang, pairs = prepareData(args.train_file)

        print(random.choice(pairs))

        if args.char:
            model = EncoderDecoder(args.hidden_size, input_lang.n_chars, output_lang.n_chars, args.drop,
                                   args.tfr, args.max_length, args.lr, args.simple, args.bidirectional, args.dot, False, 1)
        else:
            model = EncoderDecoder(args.hidden_size, input_lang.n_words, output_lang.n_words, args.drop,
                                   args.tfr, args.max_length, args.lr, args.simple, args.bidirectional, args.dot, args.multi, args.num_layers)

        model.trainIters(pairs, input_lang, output_lang, args.n_iters,
                         print_every=args.print_every, plot_every=args.plot_every, char=args.char)
        model.save(args.output_dir)
        model.evaluatePairs(pairs, char=args.char)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
