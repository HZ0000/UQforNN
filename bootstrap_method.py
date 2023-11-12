import argparse
import torch
from scipy.stats import t
from bootstrap import bootstrap

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
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=80, metavar='N',
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

    N_boot = 4
    q1 = t.ppf(q = args.cl/2, df = N_boot, loc=0, scale=1)
    q2 = t.ppf(q = args.cl, df = N_boot, loc=0, scale=1)
    c_boot1, c_boot2 = 0, 0
    width_boot1, width_boot2 = 0, 0
    pred_boot = 0
    std_bootstrap_list = []
    mean_bootstrap_list = []

    for repeat in range(args.repeats):
        mean_boot, std_boot = bootstrap(args, T_POINT, T_y, N_boot, device)
        std_bootstrap_list.append(std_boot)
        mean_bootstrap_list.append(mean_boot)
        width_boot1 += - q1 * std_boot *2
        width_boot2 += - q2 * std_boot *2
        pred_boot += mean_boot
        if T_y >= mean_boot + std_boot * q1 and T_y <= mean_boot - std_boot * q1:
            c_boot1 += 1
        if T_y >= mean_boot + std_boot * q2 and T_y <= mean_boot - std_boot * q2:
            c_boot2 += 1

        print('repeats {}, 95% c_values: {:.2f}, 90% c_values: {:.2f}' 
            .format(repeat, c_boot1/len(std_bootstrap_list), c_boot2/len(std_bootstrap_list)))
        
    print('95% c_values: {}, width: {:.4f}, 90% c_values: {} width: {:.4f}, pred: {:.4f}' 
        .format(c_boot1/args.repeats, width_boot1/args.repeats, 
        c_boot2/args.repeats, width_boot2/args.repeats, pred_boot/args.repeats))

if __name__ == '__main__':
    main()