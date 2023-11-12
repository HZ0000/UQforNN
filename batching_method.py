import argparse
import torch
from scipy.stats import t
from batching import batching


def main():
    # Training settings
	parser = argparse.ArgumentParser(description='PyTorch batching and bootstrap')

	parser.add_argument('--num-validation', type=int, default=4, metavar='N',
	                    help='number of validation sets used for ensemble')
	parser.add_argument('--repeats', type=int, default=100, metavar='N',
	                    help='number of experiment repeated') # 100
	parser.add_argument('--data-dim', type=int, default=16, metavar='N',
	                    help='number of validation sets used for ensemble')
	parser.add_argument('--data-length', type=int, default=1024, metavar='N',
	                    help='number of validation sets used for ensemble')
	parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
	                    help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
	                    help='input batch size for testing')
	parser.add_argument('--epochs', type=int, default=81, metavar='N',
	                    help='number of epochs to train') # 50000
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR', # 0.01, 100
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
	#pdb.set_trace()
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

	c_batch1, c_batch2  = 0, 0
	width_batch1, width_batch2 = 0, 0
	pred_batch = 0
	N_batch = args.num_validation
	q1 = t.ppf(q = args.cl/2, df = N_batch-1, loc=0, scale=1)
	q2 = t.ppf(q = args.cl, df = N_batch-1, loc=0, scale=1)

	std_batch_list = []
	mean_batch_list = []

	for repeat in range(args.repeats):

	    mean_batch, std_batch = batching(args, T_POINT, T_y, device)
	    std_batch_list.append(std_batch)
	    mean_batch_list.append(mean_batch)
	    width_batch1 += - q1 * std_batch *2
	    width_batch2 += - q2 * std_batch *2
	    pred_batch += mean_batch
	    if T_y >= mean_batch + q1 * std_batch and T_y <= mean_batch - q1 * std_batch:
	        c_batch1 += 1
	    if T_y >= mean_batch + q2 * std_batch and T_y <= mean_batch - q2 * std_batch:
	        c_batch2 += 1
	    print('repeats {}, 95% c_values: {:.2f}, 90% c_values: {:.2f}' 
	    	.format(repeat, c_batch1/len(std_batch_list), c_batch2/len(std_batch_list)) )
	print('95% c_values: {}, width: {:.4f}, 90% c_values: {} width: {:.4f}, pred: {:.4f}' 
	      .format(c_batch1/args.repeats, width_batch1/args.repeats,
	              c_batch2/args.repeats, width_batch2/args.repeats, pred_batch/args.repeats))

if __name__ == '__main__':
    main()