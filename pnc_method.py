import argparse
import torch
from scipy.stats import t
from pnc import pnc
from dataset import generate_data
import numpy as np

def main():
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch pnc')

	parser.add_argument('--num-validation', type=int, default=4, metavar='N',
	                    help='number of validation sets used for ensemble')
	parser.add_argument('--repeats', type=int, default=10, metavar='N',
	                    help='number of experiment repeated') # 100
	parser.add_argument('--data-dim', type=int, default=16, metavar='N',
	                    help='number of validation sets used for ensemble')
	parser.add_argument('--data-length', type=int, default=1024, metavar='N',
	                    help='number of validation sets used for ensemble')
	parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
	                    help='input batch size for training')
	parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
	                    help='input batch size for testing')
	parser.add_argument('--epochs', type=int, default=350, metavar='N',
	                    help='number of epochs to train') # 50000
	parser.add_argument('--lr', type=float, default=0.6, metavar='LR', # 0.01, 100
	                    help='learning rate')
	parser.add_argument('--wd', type=float, default=1e-10, metavar='WD',
	                    help='weight decay')
	parser.add_argument('--cl', type=float, default=0.05, metavar='CL',
	                    help='confidence level')
	parser.add_argument('--no-cuda', action='store_true', default=False,
	                    help='disables CUDA training')
	parser.add_argument('--save-model', action='store_true', default=False,
	                    help='For Saving the current Model')

	args = parser.parse_args()

	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	train_kwargs = {'batch_size': args.batch_size}
	test_kwargs = {'batch_size': args.test_batch_size}
	if use_cuda:
	    cuda_kwargs = {'num_workers': 1,
	                   'pin_memory': True,
	                   'shuffle': True}
	    train_kwargs.update(cuda_kwargs)
	    test_kwargs.update(cuda_kwargs)


	T_POINT = torch.ones(1,args.data_dim) * 0.1
	T_y = torch.sin(T_POINT)
	T_y = torch.sum(T_y, -1)
	T_POINT = T_POINT

	base_mse_list = []
	pnc_mse_list = []
	X_test, y_test =  generate_data(data_dim = args.data_dim, data_length = 2048, gt_label = True)

	for _ in range(args.repeats):
	    base_mse, pnc_mse = pnc(X_test, y_test, args, T_POINT, T_y, device)
	    base_mse_list.append(base_mse)
	    pnc_mse_list.append(pnc_mse)

	print('base_mean: {:.6f}, pnc_mean: {:.6f}' .format(np.mean(base_mse_list), np.mean(pnc_mse_list)))
	print('base_std: {:.6f}, pnc_std: {:.6f}' .format(np.std(base_mse_list), np.std(pnc_mse_list)))


if __name__ == '__main__':
    main()