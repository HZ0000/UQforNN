import torch
from model import Net, weight_init
from sklearn.metrics import mean_squared_error

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_func = torch.nn.MSELoss()
        output = torch.reshape(output, target.shape)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

    if epoch%500==0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(
            epoch, loss.item()))

def pnc_predictor(X, Y, averaged_X, model0, model_new, device):
    Y = Y.cpu().numpy()
    with torch.no_grad():
        X = X.to(device)
        base_pred = torch.squeeze(model0(X)).cpu().numpy()
        new_pred =torch.squeeze(model_new(X)).cpu().numpy()
        pnc_pred = base_pred - new_pred + averaged_X.cpu().numpy()
        return pnc_pred

def evaluate_pnc_mse(X, Y, averaged_X, model0, model_new, device):
    Y = Y.cpu().numpy()
    with torch.no_grad():
        X = X.to(device)
        base_pred = torch.squeeze(model0(X)).cpu().numpy()
        new_pred =torch.squeeze(model_new(X)).cpu().numpy()
        pnc_pred = base_pred - new_pred + averaged_X.cpu().numpy()

        base_mse = mean_squared_error(base_pred, Y)
        pnc_mse = mean_squared_error(pnc_pred, Y)
        return base_mse, pnc_mse

def generate_psuedo_labels(args, data, T_POINT, device, test_feature = None, N = 5000):
    if N == 0: # use zero-constant network, do not simulate
        averaged_ouput = torch.zeros(len(data))
        averaged_T = torch.tensor(0)
        if test_feature is not None:
            averaged_test = torch.zeros(len(test_feature))
            return averaged_ouput, averaged_T, averaged_test
        else:
            return averaged_ouput, averaged_T
    
    for i in range(N):
        model = Net(input_channel = data.shape[-1], hidden_layer=args.data_length*32).to(device)
        model.apply(weight_init)
        with torch.no_grad():
            T_POINT = T_POINT.to(device)
            output = model(data.to(device))
            output_T = model(T_POINT)
            if test_feature is not None:
                output_test = model(test_feature.to(device))
            if i == 0:
                output_sum = torch.squeeze(output)
                output_T_sum = torch.squeeze(output_T)
                if test_feature is not None:
                    output_test_sum = torch.squeeze(output_test)
            else:
                output_sum += torch.squeeze(output)
                output_T_sum += torch.squeeze(output_T)
                if test_feature is not None:
                    output_test_sum += torch.squeeze(output_test)


    averaged_ouput = output_sum/N # get the averaged values
    averaged_ouput = averaged_ouput.cpu().numpy()
    averaged_ouput  = torch.tensor(averaged_ouput)

    averaged_T = torch.tensor(output_T_sum.cpu().numpy())/N

    if test_feature is not None:
        averaged_test = torch.tensor(output_test_sum.cpu().numpy())/N
        return averaged_ouput, averaged_T, averaged_test
    else:
        return averaged_ouput, averaged_T




