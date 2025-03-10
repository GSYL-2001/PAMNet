import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('-ep', '--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-device', '--device', type=str, default='cuda', help='running device')
    parser.add_argument('-noise_level', '--noise_level', type=float, default=0.05)
    parser.add_argument('-add_noise', action='store_true', help='add noise')
    parser.add_argument('-pe', '--position_encoding', type=str, default='None')
    parser.add_argument('-id', '--id', type=str, default=None)
    parser.add_argument('-dp', '--data_path', type=str, default="RW_for_invModel.csv")
    parser.add_argument('-m', "--model", type=str, default="pe_resnet", choices=['pe_resnet', 'pamnet', 'resnet'])
    return parser.parse_args()