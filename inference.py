import argparse
import torch
from tqdm import tqdm

from trainer import ImageLoader
from model.wide_res_net import WideResNet


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--fn', type=str, default='model.dat')
    
    args = parser.parse_args()
    return args


class Predictor(object):

    def __init__(self, batch_size, fn):
        self.model = None
        self.load(fn)
        dataset = ImageLoader(batch_size)
        self.test_loader = dataset.test

    def load(self, fn='model.dat'):
        cp = torch.load(fn, map_location=DEVICE)
        cfg = cp['cfg']
        self.model = WideResNet(**cfg).to(DEVICE)
        self.model.load_state_dict(cp['model_state_dict'])

    def inference(self):
        results = []
        for inputs, targets in tqdm(self.test_loader):
            outputs = self.model(inputs.to(DEVICE))
            results += torch.argmax(outputs.data, 1).cpu().tolist()
        return results



if __name__ == '__main__':
    args = get_args()
    predictor = Predictor(batch_size=args.batch_size, fn=args.fn)
    results = predictor.inference()
    print(results[:10])
