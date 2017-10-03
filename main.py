import argparse
import object_net_trainer
import rnn_trainer
import tensorflow as tf
import tf_utils


def main():
    # Set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_net", type=bool, default=False)
    parser.add_argument("--rnn", type=bool, default=False)
    tf_utils.generic_runner.add_arguments(parser)
    tf_utils.data_holder.add_arguments(parser)
    args = parser.parse_args()

    layer_size = 128
    num_layers = 2

    # Set up data
    data = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]] * 100

    if args.object_net and args.rnn:
        print("Only one of object_net and rnn must be specified")
    elif args.object_net:
        with tf.variable_scope("object_net"):
            object_net_trainer.train(args, data, layer_size, num_layers)
    elif args.rnn:
        with tf.variable_scope("rnn"):
            rnn_trainer.train(args, data, layer_size, num_layers)
    else:
        print("One of object_net and rnn must be specified")


if __name__ == "__main__":
    main()
