import torch
from IQAmodel import *
import os
import numpy as np
import random
from argparse import ArgumentParser
from torchvision.transforms.functional import resize, to_tensor, normalize
from PIL import Image
import h5py
import cv2

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model_Joint(return_feature=True).to(device) if args.save_heatmap else Model_Joint().to(device)

    checkpoint = torch.load(args.trained_model_file)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    k = checkpoint['k']
    b = checkpoint['b']

    im = Image.open((os.path.join(args.root_path, args.img_name))).convert('RGB')
    if args.resize:
        im = resize(im, (args.resize_size_h, args.resize_size_w))
    im = to_tensor(im).to(device)
    im = normalize(im, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if args.save_heatmap is None:
        q = model(im.unsqueeze(0))
        print('The image quality score is {}'.format(q[-1].item() * k[-1] + b[-1]))

    # save heatmaps for side network features, useful for analysis
    else:
        def extract(g):
            global features_grad
            features_grad = g

        f_dist, f_q, dist, q = model(im.unsqueeze(0))

        print('The image quality score is {}'.format(q[-1].item() * k[-1] + b[-1]))

        f_q.register_hook(extract)
        q.backward()

        grads = features_grad

        pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

        pooled_grads = pooled_grads[0]
        features = f_q[0]

        for i in range(256):
            features[i, ...] *= pooled_grads[i, ...]

        heatmap = features.detach().cpu().numpy()
        heatmap = np.mean(heatmap, axis=0)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 0.00000001)

        img = cv2.imread(os.path.join(args.root_path, args.img_name))
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img * 0.6
        cv2.imwrite(os.path.join(args.root_path, args.img_name[0:-4]+'_quality.jpg'), superimposed_img)

        # save heatmaps  for four defects
        d_names=['noise','blur','contrast','color']
        for d_num in range(4):
            f_dist, f_q, dist, q = model(im.unsqueeze(0))

            f_dist.register_hook(extract)
            dist[:, d_num].backward()

            grads = features_grad

            pooled_grads = (torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1)))

            pooled_grads = pooled_grads[0]
            features = f_dist[0]

            for i in range(256):
                features[i, ...] *= pooled_grads[i, ...]

            heatmap = features.detach().cpu().numpy()
            heatmap = np.mean(heatmap, axis=0)

            heatmap = np.maximum(heatmap, 0)
            heatmap /= (np.max(heatmap)+0.0001)

            img = cv2.imread(os.path.join(args.root_path, args.img_name))
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = heatmap * 0.4 + img * 0.6
            cv2.imwrite(os.path.join(args.root_path, args.img_name[0:-4]+'_dist_'+d_names[d_num]+'.jpg'), superimposed_img)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--root_path', default='./imgs/', type=str,
                        help='root path for test image')
    parser.add_argument("--img_name", default='test_img.jpg', type=str)
    parser.add_argument('--save_heatmap', action='store_true',
                        help='whether save heatmap?')

    parser.add_argument('--loss_type', default='norm-in-norm', type=str,
                        help='loss type (default: norm-in-norm)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('-bs', '--batch_size', type=int, default=8,
                        help='batch size for training (default: 8)')
    parser.add_argument('--ft_lr_ratio', type=float, default=0.1,
                        help='ft_lr_ratio (default: 0.1)')
    parser.add_argument('-e', '--epochs', type=int, default=25,
                        help='number of epochs to train (default: 25)')
    parser.add_argument('--p', type=float, default=1,
                        help='p (default: 1)')
    parser.add_argument('--q', type=float, default=2,
                        help='q (default: 2)')
    parser.add_argument('--alpha', nargs=2, type=float, default=[1, 0],
                        help='loss coefficient alpha in total loss (default: [1, 0])')
    parser.add_argument('--opt_level', default='O1', type=str,
                        help='opt_level for amp (default: O1)')
    parser.add_argument('--detach', action='store_true',
                        help='Detach in loss?')
    parser.add_argument('--dataset', default='KonIQ-10k', type=str,
                        help='dataset name (default: KonIQ-10k)')
    parser.add_argument('--augment', action='store_true',
                        help='Data augmentation?')
    parser.add_argument('--monotonicity_regularization', action='store_true',
                        help='use monotonicity_regularization?')

    parser.add_argument('--trained_model_file', default=None, type=str,
                        help='trained_model_file')

    parser.add_argument('--resize', action='store_true',
                        help='Resize?')
    parser.add_argument('-rs_h', '--resize_size_h', default=480, type=int,
                        help='resize_size_h (default: 480)')
    parser.add_argument('-rs_w', '--resize_size_w', default=640, type=int,
                        help='resize_size_w (default: 640)')

    args = parser.parse_args()

    args.format_str = 'model-loss={}-p={}-q={}-detach-{}-ft_lr_ratio={}-alpha={}-{}-res={}-{}x{}-aug={}-monotonicity={}-lr={}-bs={}-e={}-opt_level={}' \
        .format(args.loss_type, args.p, args.q, args.detach, args.ft_lr_ratio, args.alpha,
                args.dataset, args.resize, args.resize_size_h, args.resize_size_w, args.augment,
                args.monotonicity_regularization, args.lr, args.batch_size, args.epochs, args.opt_level)
    args.trained_model_file = './checkpoints/' + args.format_str

    run(args)
