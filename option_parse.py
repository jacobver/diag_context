import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        add_help=False, description=' memory seq 2 seq ')

    # Data options

    parser.add_argument('-data',
                        help='Path to the *-train.pt file from preprocess.py')
    parser.add_argument('-save_model', default='models/',
                        help='Model filename (the model will be saved in models as [mem].[data].pt')
    parser.add_argument('-train_from_state_dict', default='', type=str,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model's state_dict.""")
    parser.add_argument('-train_from', default='', type=str,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model.""")
    parser.add_argument('-context_size', type=int, default=3,
                        help='Number of utterances in context')

    # Model options

    parser.add_argument('-layers', type=int, default=2,
                        help='Number of layers in the LSTM encoder/decoder')
    parser.add_argument('-rnn_size', type=int, default=300,
                        help='Size of LSTM hidden states')
    parser.add_argument('-word_vec_size', type=int, default=300,
                        help='Word embedding sizes')
    parser.add_argument('-input_feed', type=int, default=1,
                        help="""Feed the context vector at each time step as
                        additional input (via concatenation with the word
                        embeddings) to the decoder.""")
    parser.add_argument('-attn', type=int, default=1)

    # Optimization options
    parser.add_argument('-batch_size', type=int, default=32,
                        help='Maximum batch size')
    parser.add_argument('-max_generator_batches', type=int, default=32,
                        help="""Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster, but uses
                        more memory.""")
    parser.add_argument('-epochs', type=int, default=79,
                        help='Number of training epochs')
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')
    parser.add_argument('-param_init', type=float, default=0.1,
                        help="""Parameters are initialized over uniform distribution
                        with support (-param_init, param_init)""")
    parser.add_argument('-optim', default='adam',
                        help="Optimization method. [sgd|adagrad|adadelta|adam]")
    parser.add_argument('-max_grad_norm', type=float, default=5,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to max_grad_norm""")
    parser.add_argument('-dropout', type=float, default=0.4,
                        help='Dropout probability; applied between LSTM stacks.')
    parser.add_argument('-dropout_nse', type=float, default=0.2,
                        help='Dropout probability; applied between LSTM stacks.')
    parser.add_argument('-curriculum', action="store_true",
                        help="""For this many epochs, order the minibatches based
                        on source sequence length. Sometimes setting this to 1 will
                        increase convergence speed.""")
    parser.add_argument('-extra_shuffle', type=int, default=1,
                        help="""By default only shuffle mini-batch order; when true,
                        shuffle and re-assign mini-batches""")

    # learning rate
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Starting learning rate. If adagrad/adadelta/adam is
                        used, then this is the global learning rate. Recommended
                        settings: sgd = 1, adagrad = 0.1,
                        adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_decay', type=float, default=0.75,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past
                        start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=9,
                        help="""Start decaying every epoch after and including this
                        epoch""")

    # pretrained word vectors
    parser.add_argument('-pre_word_vecs',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the encoder side.
                        See README for specific formatting instructions.""")

    # GPU
    parser.add_argument('-gpus', type=int,
                        help="Use CUDA on the listed devices.")

    parser.add_argument('-log_interval', type=int, default=50,
                        help="Print stats at this interval.")

    parser.add_argument('-act_vec_size', type=int, default=25,
                        help='use context of dialogue acts')

    # Memory options
    parser.add_argument('-mem', default=None,
                        help='which type of memory to use, default: None')

    # DNC options
    parser.add_argument('-mem_slots', type=int, default=40,
                        help='in case of [dnc]: number of memory slots')
    parser.add_argument('-mem_size', type=int, default=100,
                        help='in case of [dnc]: size of memory slots')
    parser.add_argument('-read_heads', type=int, default=1,
                        help='in case of [dnc]: number of read heads')

    # hypertune options
    parser.add_argument('-prev_opts', default=None,
                        help='pkl file with previously tried options')

    # random_seed
    parser.add_argument('-seed', type=int, default=-1,
                        help="""Random seed used for the experiments
                        reproducibility.""")

    return parser
