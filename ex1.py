import argparse


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_train_samples', type=int, default=-1)
    parser.add_argument('--max_eval_samples', type=int, default=-1)
    parser.add_argument('--max_predict_samples', type=int, default=-1)
    parser.add_argument('--num_train_epochs', type=int, default=2)  # max 5
    parser.add_argument('--lr', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--do_train', type=bool, default=False)
    parser.add_argument('--do_predict', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='path/to/model')
    return parser


def main():
    # Command-line arguments
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    print(args)


if __name__ == '__main__':
    main()
