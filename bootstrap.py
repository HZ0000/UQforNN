import torch
import numpy as np
import torch.optim as optim
from utils import *
from model import Net, weight_init
from dataset import ImdbData, generate_data

def bootstrap(args, T_POINT, T_y, N_boot, device):

    # generate the dataset
    # split the dataset into two part, A and B.
    X_train, y_train = generate_data(data_dim = args.data_dim, data_length = args.data_length)

    # generate dataset A
    A_start_index, A_end_index = 0, args.data_length
    A_set_index = list(range(A_start_index, A_end_index))
    X_Aset, Y_Aset = X_train[A_set_index], y_train[A_set_index]

    # generate dataset B
    B_start_index, B_end_index = 0, args.data_length
    B_set_index = list(range(B_start_index, B_end_index))

    # get the psuedo labels
    B_pesudo, averaged_T = generate_psuedo_labels(args, X_train[B_set_index], T_POINT, device,
                                                  test_feature = None, N = 5000)

    X_Bset, Y_Bset = X_train[B_set_index], B_pesudo

    # get the initial network
    model0 = Net(input_channel = X_train.shape[-1], hidden_layer=args.data_length*32).to(device)
    model0.apply(weight_init)
    optimizer = optim.SGD(model0.parameters(), lr=args.lr, weight_decay=args.wd)
    PATH_ini = 'model0_initial.pth'
    torch.save(model0.state_dict(),PATH_ini)

    # train base models
    print('Training the base model')
    A_data_base = ImdbData(X_Aset, Y_Aset)
    A_loader_base = torch.utils.data.DataLoader(dataset=A_data_base,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    for epoch in range(1, args.epochs + 1):
        train(args, model0, device, A_loader_base, optimizer, epoch)
    if args.save_model:
        PATH_final = 'model0_final.pth'
        torch.save(model0.state_dict(), PATH_final)


    B_data_base = ImdbData(X_Bset, Y_Bset)
    B_loader_base = torch.utils.data.DataLoader(dataset=B_data_base,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    model_new = Net(input_channel = X_train.shape[-1], hidden_layer=args.data_length*32).to(device)
    model_new.load_state_dict(torch.load(PATH_ini))
    optimizer_new = optim.SGD(model_new.parameters(), lr=args.lr, weight_decay=args.wd)
    for epoch in range(1, args.epochs + 1):
        train(args, model_new, device, B_loader_base, optimizer_new, epoch)

    if args.save_model:
        PATH = 'modelnew_final.pth'
        torch.save(model_new.state_dict(), PATH)

    base_val = pnc_predictor(T_POINT, T_y, averaged_T, model0, model_new, device)

    # train model0 and modelnew
    sub_net_pred = []
    for i in range(N_boot):
        # sample selection
        print('Training model {} with A dataset' .format(i))
        selected_samples = np.random.randint(args.data_length, size= args.data_length)
        A_data_sub = ImdbData(X_Aset[selected_samples], Y_Aset[selected_samples])
        A_loader_sub = torch.utils.data.DataLoader(dataset=A_data_sub,
                                                   batch_size=args.batch_size,
                                                   shuffle=True)

        model0 = Net(input_channel = X_train.shape[-1], hidden_layer=args.data_length*32).to(device)
        model0.load_state_dict(torch.load(PATH_ini))
        optimizer = optim.SGD(model0.parameters(), lr=args.lr, weight_decay=args.wd)

        for epoch in range(1, args.epochs + 1):
            train(args, model0, device, A_loader_sub, optimizer, epoch)
        if args.save_model:
            PATH_final = 'model0_final.pth'
            torch.save(model0.state_dict(), PATH_final)

        print('Training model {} with B dataset' .format(i))
        selected_samples = np.random.randint(args.data_length, size= args.data_length)
        B_data_sub = ImdbData(X_Bset[selected_samples], Y_Bset[selected_samples])
        B_loader_sub = torch.utils.data.DataLoader(dataset=B_data_sub,
                                                   batch_size=args.batch_size,
                                                   shuffle=True)
        model_new = Net(input_channel = X_train.shape[-1], hidden_layer=args.data_length*32).to(device)
        model_new.load_state_dict(torch.load(PATH_ini))
        optimizer_new = optim.SGD(model_new.parameters(), lr=args.lr, weight_decay=args.wd)
        for epoch in range(1, args.epochs + 1):
            train(args, model_new, device, B_loader_sub, optimizer_new, epoch)

        if args.save_model:
            PATH = 'modelnew_final.pth'
            torch.save(model_new.state_dict(), PATH)

        sub_pred = pnc_predictor(T_POINT, T_y, averaged_T, model0, model_new, device)
        sub_net_pred.append(sub_pred)
    std = np.mean((np.array(sub_net_pred) - base_val)**2)
    std = np.sqrt(std)

    print('base_val, {:.4f}; std, {:.4f}; T_y, {:.4f}' .format(base_val, std, T_y.item()))
    return base_val, std