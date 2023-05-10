# -*- coding: utf-8 -*-
"""
Created on Fri May 13 22:00:35 2022

@author: de'l'l
"""
import copy
import pandas as pd
import numpy as np
import torch
import csv
from torch import nn
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import mynets
import myfunctions



def do_all(dataset, park, predict_time):
    use_cuda = True
    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")

    # super mark
    LOOK_BACK = 12
    INPUT_FEATURES_NUM = 1
    HIDDEN_SIZE = 1
    OUTPUT_FEATURES_NUM = 1
    NUM_LAYERS = 1
    max_epochs = 3000 # 3000
    LEARNING_RATE = 0.01
    LOSS_BOUND = 0.00001

    #------------input data---------

    rate = dataset['RATE'].values.astype('float32')
    fouriers = torch.FloatTensor(np.array(range(len(dataset)))).unsqueeze(1)

    #division
    train_size = int(len(rate)*0.8)
    train_rate = rate[:train_size]
    test_rate = rate[train_size:]
    test_size = len(test_rate)

    #-----------fourier series-----------------
    def linear_reg(input):
        return input @ w.t() + b
    def gd(params):
        for param in params:
            param.data -= fouriers_lr * param.grad

    N = 30
    period = 2016
    fouriers_loss_function = torch.nn.MSELoss()
    fouriers_lr = 0.02
    num_epochs = 1000
    model = linear_reg

    #-----------fourier tranform
    fourier_series = fouriers
    for i in range(N):
        n = i + 1
        cos_1 = np.cos(n * 2 * np.pi * fouriers / period)
        sin_1 = np.sin(n * 2 * np.pi * fouriers / period)
        fourier_series = np.concatenate((fourier_series,cos_1,sin_1),axis=1)
    fourier_series = fourier_series[:,1:]
    fourier_series = torch.FloatTensor(fourier_series)
    fourier_features = fourier_series.shape[1]
    train_fouriers = fourier_series[:train_size]
    test_fouriers = fourier_series[train_size:]

    #define Weights and biases
    w = torch.randn(1,fourier_features,requires_grad=True)  # 1 * input_feature
    b = torch.zeros(1, requires_grad=True)

    fourier_train_rate = torch.reshape(torch.FloatTensor(train_rate),[len(train_rate)])
    #train
    for epoch in range(num_epochs):
        preds = torch.reshape(model(train_fouriers),[len(train_fouriers)])
        loss = fouriers_loss_function(preds,fourier_train_rate)
        loss.backward()
        gd([w, b])
        w.grad.zero_()
        b.grad.zero_()

    #prediction
    period_factors = model(fourier_series)
    period_factors = period_factors.detach().cpu().numpy()
    train_four = period_factors[:train_size]
    test_four = period_factors[train_size:]

    #------------effect term
    train_rate = train_rate.reshape(-1,1)
    test_rate = test_rate.reshape(-1,1)
    train_rate = np.array(train_rate)
    test_rate = np.array(test_rate)
    train_eff = (train_rate - train_four)/train_rate
    test_eff = (test_rate - test_four)/test_rate

    #-------------make data------------
    #-------train data
    train_trend, train_y = myfunctions.creat_interval_dataset(train_rate, LOOK_BACK, predict_time)
    test_trend, test_y = myfunctions.creat_interval_dataset(test_rate, LOOK_BACK, predict_time)
    print(train_trend.shape)
    train_cycle, _ = myfunctions.creat_interval_dataset(train_four, LOOK_BACK, predict_time)
    train_effect, _ = myfunctions.creat_interval_dataset(train_eff, LOOK_BACK, predict_time)
    test_cycle, _ = myfunctions.creat_interval_dataset(test_four, LOOK_BACK, predict_time)
    test_effect, _ = myfunctions.creat_interval_dataset(test_eff, LOOK_BACK, predict_time)

    train_x = np.concatenate((train_trend, train_cycle, train_effect), axis=2)
    test_x = np.concatenate((test_trend, test_cycle, test_effect), axis=2)

    train_x_tensor = torch.from_numpy(train_x).to(device)
    print(train_x_tensor.shape)
    train_y_tensor = torch.from_numpy(train_y).to(device)

    #-------test data
    test_x_tensor = torch.from_numpy(test_x).to(device)
    test_y_tensor = torch.from_numpy(test_y).to(device)


    #----------------trainning
    trainning_loss = []
    model = mynets.TpaModel(3, 3, 6, 1).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(max_epochs):
        output = model(train_x_tensor)
        loss = loss_function(output, train_y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        trainning_loss.append([loss.item()])

        if loss.item() < LOSS_BOUND:
            print('Epoch [{}/{}], loss:{:.5f}'.format(epoch+1, max_epochs, loss.item()))
            print('the loss bound is reached')
            break
        elif (epoch+1)%100 == 0:
            print('Epoch [{}/{}], Loss:{}'.format(epoch+1, max_epochs, loss.item()))

    out_loss_df = pd.DataFrame(columns=['train_loss'],data=trainning_loss)
    out_loss_df.to_csv('./result_'+str(predict_time)+'/'+park[:-4]+'_out_loss_'+str(predict_time)+'.csv',encoding='gbk')
    torch.save(model.state_dict(), 'models/model_%s.pt' % park[:-4])

    #---------------testing--------------------

    test_model = mynets.TpaModel(3, 3, 6, 1)
    test_model.load_state_dict(torch.load('models/model_%s.pt' % park[:-4]))

    y_pre = model(test_x_tensor).to(device)
    pre_loss = loss_function(y_pre, test_y_tensor)
    y_pre = y_pre.detach().cpu().numpy()
    pre_loss = pre_loss.detach().cpu().numpy()

    test_y_numpy = test_y_tensor.detach().cpu().numpy()
    test_y_numpy = test_y_numpy.reshape(-1,1)
    y_pre = y_pre.reshape(-1,1)
    # print('test_y_numpy:',test_y_numpy.shape)
    # print('y_pre:',y_pre.shape)
    MAPE = np.mean(abs((y_pre - test_y_numpy)/test_y_numpy))
    MAE=mean_absolute_error(test_y_numpy,y_pre)
    MSE=mean_squared_error(test_y_numpy,y_pre)
    RAE=np.sum(abs(y_pre - test_y_numpy)) / np.sum(abs(np.mean(test_y_numpy) - test_y_numpy))
    RMSE= np.sqrt(MSE)
    R2=r2_score(test_y_numpy,y_pre)

    print('MAPE15: {}, Prediction Loss: {:.6f}'.format(MAPE, pre_loss.item()))
    print('MAE:{}'.format(MAE))
    print('MSE:{}'.format(MSE))
    print('RMSE:{}'.format(RMSE))
    print('R2:{}'.format(R2))
    print('MAE:{}'.format(RAE))
    # print('output data:',y_pre.shape)

    fg = np.asarray([1,2,3,4,5])
    output_list = [park[:-4], MSE, RMSE, MAPE, RAE, MAE, R2]
    print(output_list)

    return output_list





