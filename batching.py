import torch
import numpy as np
import torch.optim as optim
from utils import *
from model import Net, weight_init
from dataset import ImdbData, generate_data


def batching(args, T_POINT, T_y, device):
    
    # generate the training set
    X_train, y_train = generate_data(data_dim = args.data_dim, data_length = args.data_length)
    dataset1 = ImdbData(X_train, y_train)
    
    # number of sampels in each steps
    step = len(dataset1)//args.num_validation
    train_split = list(range(0, len(dataset1), step))
    train_split.append(len(dataset1))

    # get the initial network
    model0 = Net(input_channel = X_train.shape[-1], hidden_layer=args.data_length*32).to(device)
    model0.apply(weight_init)
    optimizer = optim.SGD(model0.parameters(), lr=args.lr, weight_decay=args.wd)
    PATH_ini = 'model0_initial.pth'
    torch.save(model0.state_dict(),PATH_ini)

    # batching
    sub_net_pred = []
    for i in range(1, len(train_split)):
        # get the datasets
        print('Training on {} network' .format(i))
        A_start_index, A_end_index = train_split[i-1], train_split[i]
        B_start_index, B_end_index = train_split[i-1], train_split[i]
        A_set_index = list(range(A_start_index, A_end_index))
        B_set_index = list(range(B_start_index, B_end_index))
        A_data_sub = ImdbData(X_train[A_set_index], y_train[A_set_index])
        A_loader_sub = torch.utils.data.DataLoader(dataset=A_data_sub,
                                                   batch_size=args.batch_size,
                                                   shuffle=True)


        # train the initial model
        model0 = Net(input_channel = X_train.shape[-1], hidden_layer=args.data_length*32).to(device)
        model0.load_state_dict(torch.load(PATH_ini))
        optimizer = optim.SGD(model0.parameters(), lr=args.lr, weight_decay=args.wd)

        #get the psuedo labels
        B_pesudo, averaged_T = generate_psuedo_labels(args, X_train[B_set_index], T_POINT, device,
                                                      test_feature = None, N = 5000)
        B_data_sub = ImdbData(X_train[B_set_index], B_pesudo)
        B_loader_sub = torch.utils.data.DataLoader(dataset=B_data_sub,
                                                   batch_size=args.batch_size,
                                                   shuffle=True)

        # train model0
        optimizer = optim.SGD(model0.parameters(), lr=args.lr, weight_decay=args.wd)
        for epoch in range(1, args.epochs + 1):
            train(args, model0, device, A_loader_sub, optimizer, epoch)
        if args.save_model:
            PATH_final = 'model0_final.pth'
            torch.save(model0.state_dict(), PATH_final)

        #train model_new
        model_new = Net(input_channel = X_train.shape[-1], hidden_layer=args.data_length*32).to(device)
        model_new.load_state_dict(torch.load(PATH_ini))
        optimizer_new = optim.SGD(model_new.parameters(), lr=args.lr, weight_decay=args.wd)
        for epoch in range(1, args.epochs + 1):
            train(args, model_new, device, B_loader_sub, optimizer_new, epoch)

        if args.save_model:
            PATH = 'modelnew_final.pth'
            torch.save(model_new.state_dict(), PATH)

        # evaluate performance
        sub_pred = pnc_predictor(T_POINT, T_y, averaged_T, model0, model_new, device)
        sub_net_pred.append(sub_pred)

    sub_net_pred = np.array(sub_net_pred)
    std = np.var(sub_net_pred, ddof=1)/args.num_validation
    std = np.sqrt(std)
    mean_sub = np.mean(sub_net_pred)
    # calculate mean values
    print('Batch mean {:.4f}, std {:.4f}, T_y {:.4f}' .format(mean_sub, std, T_y.item()))
    return mean_sub, std