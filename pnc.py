import torch
import numpy as np
import torch.optim as optim
from utils import *
from model import Net, weight_init
from dataset import ImdbData, generate_data


def pnc(X_test, y_test, args, T_POINT, T_y, device):
    # generate the dataset
    X_train, y_train = generate_data(data_dim = args.data_dim, data_length = args.data_length)
    train_data = ImdbData(X_train, y_train)

    # get the initial network
    model0 = Net(input_channel = args.data_dim, hidden_layer=args.data_length*32).to(device)
    model0.apply(weight_init)
    optimizer = optim.SGD(model0.parameters(), lr=args.lr, weight_decay=args.wd)
    PATH_ini = 'model0_initial.pth'
    torch.save(model0.state_dict(),PATH_ini)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True)


    # train the initial model
    model0 = Net(input_channel = args.data_dim, hidden_layer=args.data_length*32).to(device)
    model0.load_state_dict(torch.load(PATH_ini))
    optimizer = optim.SGD(model0.parameters(), lr=args.lr, weight_decay=args.wd)
    for epoch in range(1, args.epochs + 1):
        train(args, model0, device, train_loader, optimizer, epoch)
    if args.save_model:
        PATH_final = 'model0_final.pth'
        torch.save(model0.state_dict(), PATH_final)

    
    B_pesudo, averaged_T, test_psuedo = generate_psuedo_labels(args, X_train, T_POINT, device,
                                                  test_feature = X_test, N = 0)

    B_data = ImdbData(X_train, B_pesudo)
    B_loader = torch.utils.data.DataLoader(dataset=B_data,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    #train model_new
    model_new = Net(input_channel = args.data_dim, hidden_layer=args.data_length*32).to(device)
    model_new.load_state_dict(torch.load(PATH_ini))
    optimizer_new = optim.SGD(model_new.parameters(), lr=args.lr, weight_decay=args.wd)
    for epoch in range(1, args.epochs + 1):
        train(args, model_new, device, B_loader, optimizer_new, epoch)

    if args.save_model:
        PATH = 'modelnew_final.pth'
        torch.save(model_new.state_dict(), PATH)


    # evaluate performance
    base_mse, pnc_mse = evaluate_pnc_mse(X_test, y_test, test_psuedo, model0, model_new, device)

    return base_mse, pnc_mse
    