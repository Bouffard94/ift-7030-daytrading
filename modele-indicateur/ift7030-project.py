#!/home/sdamours/anaconda3/bin/python

import numpy
import time
import pandas
pandas.set_option('display.max_colwidth', 0)
import collections
import itertools
from numpy import genfromtxt
from datetime import datetime as Datetime


import torch
import matplotlib.pyplot as plt


import pandas as pd
import matplotlib.pyplot as plt
import torch
import sys


import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale

import torch.nn.functional as F



from IPython import display

from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import minmax_scale
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot, ticker
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
import matplotlib

from sklearn.feature_selection import SelectKBest, RFE, chi2, mutual_info_classif
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Nous ne voulons pas être signalé par ce type d'avertissement, non pertinent pour le devoir
# We don't want to be signaled of this warning, irrelevant for the homework
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore", category=FutureWarning)


import csv
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset




user_device = ""
user_tag = ""
print( f"Arguments count: {len(sys.argv)}" )
for i, arg in enumerate(sys.argv):
  print(f"Argument {i:>6}: {arg}")
  if( i == 0 ):
    continue
  elif( i == 1 ):
    user_file = arg
  elif( i == 2 ):
    user_device = arg
  else:
    user_tag = arg



# if( device != "" ):
#     pass
# else:
#     gpu_flag = 1
#     if( gpu_flag ):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     else:
#         device = "cpu"

device = "cpu"
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################




# vol_1min_var_threshold = 1.3
# vol_5min_var_threshold = 1.15
# vol_10min_var_threshold = 1.05

# price_1min_var_threshold = 1.1
# price_5min_var_threshold = 1
# price_10min_var_threshold = 0.9

# buy_threshold = 1.02



###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################



def prepend( tensor, size, value, device ):
    padding = torch.ones(size, device=device).float()
    padding *= value
    return torch.cat((padding, tensor), dim=0)

def append( tensor, size, value, device ):
    padding = torch.ones(size, device=device).float()
    padding *= value
    return torch.cat((tensor.to(device), padding.to(device)), dim=0)



def generate_spectrogram_global( tensor, filename, axs, i ):
    n_fft = 256
    hop_length = n_fft // 64
    window = torch.hann_window(n_fft)
    specgram = torch.stft(tensor, n_fft=n_fft, hop_length=hop_length, window=window)

    magnitude_specgram = torch.abs(specgram)

    im = axs[i].imshow(magnitude_specgram.log2().numpy(), aspect='auto', origin='lower')
    axs[i].set_title(f'Spectrogram {i+1}')
    axs[i].grid(False)
    axs[i].set_yticks([]) 








def generate_spectrogram( tensor, filename ):
    n_fft = 256
    hop_length = n_fft // 64
    window = torch.hann_window(n_fft)
    specgram = torch.stft(tensor, n_fft=n_fft, hop_length=hop_length, window=window)

    magnitude_specgram = torch.abs(specgram)

    plt.figure(figsize=(8, 4))
    plt.imshow(magnitude_specgram[0].log2().numpy(), aspect='auto', origin='lower')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.xlim(-0.5,0)
    plt.title('Spectrogram')
    plt.colorbar(label='Magnitude (log scale)')

    plt.savefig( filename )













import torch

def calculate_rsi(prices, window_length=14):

    print( "calculating RSI...")
    deltas = prices[1:] - prices[:-1]
    gains = torch.where(deltas > 0, deltas, torch.zeros_like(deltas))
    losses = torch.where(deltas < 0, -deltas, torch.zeros_like(deltas))

    rsi = torch.zeros_like(prices)
    for i in range(1, len(prices)):
        avg_gain = torch.mean(gains[max(0, i - window_length):i])
        avg_loss = torch.mean(losses[max(0, i - window_length):i])

        if avg_loss != 0:
            rs = avg_gain / avg_loss
        else:
            rs = torch.ones_like(avg_gain)
        rsi[i] = 100 - (100 / (1 + rs))

    print( "Done" )
    return rsi





def ema(prices, window):
    alpha = 2 / (window + 1)
    ema_values = torch.zeros_like(prices)
    ema_values[0] = prices[0]
    
    for i in range(1, len(prices)):
        ema_values[i] = alpha * prices[i] + (1 - alpha) * ema_values[i - 1]
    
    return ema_values


def calculate_macd(prices, short_window=12, long_window=26, signal_window=9):
    
    print( "calculating MACD...")
    exp_short = ema(prices, short_window)
    exp_long = ema(prices, long_window)
    macd_line = exp_short - exp_long
    signal_line = ema(macd_line, signal_window)
    histogram = macd_line - signal_line
    print( "Done" )
    return macd_line, signal_line, histogram








def calculate_rolling_statistics(prices, window):
    means = prices.unfold(0, window, 1).mean(dim=1)
    std_devs = prices.unfold(0, window, 1).std(dim=1)
    return means, std_devs

def calculate_bollinger_bands(prices, window=20, num_std=2):

    print( "calculating bollinger bands...")

    middle_band, std_dev = calculate_rolling_statistics(prices, window)
    upper_band = middle_band + (num_std * std_dev)
    lower_band = middle_band - (num_std * std_dev)

    print( "Done" )
    return middle_band, upper_band, lower_band




















def calculate_obv(prices, volumes, date_list):
    print( "calculating OBV...")
    obv = torch.zeros(len(prices))
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif prices[i] < prices[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]

        if( date_list[i] != date_list[i-1] ):
            obv[i] = 0
    print( "Done" )
    return obv














# # Compute FFT
# fft_result = torch.fft.fft(close_price_tensor)

# # Calculate the magnitude of the FFT result
# fft_magnitude = torch.abs(fft_result)

# # Plotting the FFT magnitude
# plt.figure(figsize=(10, 6))
# plt.plot(np.fft.fftfreq(len( close_price )), fft_magnitude.numpy(), label='FFT Magnitude')
# # plt.plot( fft_result, label='fft_result', color="blue")
# plt.plot( close_price, label='close_price', color="red")
# plt.title('FFT')
# plt.xlabel('Frequency')
# plt.ylabel('Magnitude')
# plt.ylim(0, 2 )  # Set y-axis range
# plt.legend()
# plt.show()





def generate_fft( data ):

    fft_result = torch.fft.fft(data)

    # Calculate the magnitude of the FFT result
    fft_magnitude = torch.abs(fft_result)

    freqs = torch.fft.fftfreq(len(data), d=1)  # Frequency values for FFT
    auc = np.trapz(fft_magnitude, x=freqs.numpy())

    # print( auc )

    # # Plotting the FFT magnitude
    # plt.figure(figsize=(10, 6))
    # plt.plot(np.fft.fftfreq(len( data )), fft_magnitude.numpy(), label='FFT Magnitude')

    # plt.text(0.5, 0.9, f'AUC: {auc:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    # plt.title('FFT Magnitude of Financial Data')
    # plt.ylim(0, 2 )  # Set y-axis range
    # plt.xlabel('Frequency')
    # plt.ylabel('Magnitude')
    # plt.legend()
    # plt.show()

    return auc













def save_confusion_matrix( confusion_matrix, filename ):


    # Plotting the confusion matrix as a heatmap
    class_names = ['Negative', 'Positive']

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix.numpy(), annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - {}'.format( user_tag ) )


    plt.savefig( filename )








def compute_metrics( predictions, labels ):

    predictions = predictions.cpu()
    labels = labels.cpu()
    non_zeros_labels = labels[labels != 0]
    non_zeros_predictions = predictions[labels != 0]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', warn_for=tuple())
    nz_precision, nz_recall, nz_f1, _ = precision_recall_fscore_support(non_zeros_labels, non_zeros_predictions, average='weighted', warn_for=tuple())
    acc = accuracy_score(labels, predictions)
    nz_acc = accuracy_score(non_zeros_labels, non_zeros_predictions)
    
    return { 'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall, "nonzeros_pred": np.count_nonzero(predictions) / len(predictions), "nonzeros_labels": np.count_nonzero(labels) / len(labels),
            'nz_precision': nz_precision, 'nz_recall': nz_recall, 'nz_f1': nz_f1, 'nz_accuracy': nz_acc }


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################






import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import math


class CNN1D(nn.Module):
    def __init__(self, input_channels):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(128 * 5, 64)
        self.fc2 = nn.Linear(64, 1)  # Single output neuron for binary classification
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 128 * 5)  # Flatten before FC layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # Apply sigmoid activation for binary classification
        return x





class NeuralNetworkSimple(nn.Module):
    def __init__(self):
        super(NeuralNetworkSimple, self).__init__()
        self.fc1 = nn.Linear( 13, 9192 )
        self.fc2 = nn.Linear( 9192, 2048 )
        self.fc3 = nn.Linear( 2048, 9192 )
        self.fc4 = nn.Linear( 9192, 1 )
        self.sigmoid = nn.Sigmoid()  
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x




class NeuralNetworkHuge(nn.Module):
    def __init__(self):
        super(NeuralNetworkHuge, self).__init__()
        width_pre = 18
        width_post = 8192
        self.fc1 = nn.Linear(width_pre, width_post)
        self.bn1 = nn.BatchNorm1d(width_post)
        self.dropout1 = nn.Dropout(0.5)

        width_pre = width_post
        width_post = 4096
        self.fc2 = nn.Linear(width_pre, width_post)
        self.bn2 = nn.BatchNorm1d(width_post)
        self.dropout2 = nn.Dropout(0.5)

        width_pre = width_post
        width_post = 8192
        self.fc3 = nn.Linear(width_pre, width_post)
        self.bn3 = nn.BatchNorm1d(width_post)
        self.dropout3 = nn.Dropout(0.5)

        width_pre = width_post
        self.fc4 = nn.Linear(width_pre, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc4(x))
        return x


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        width_pre = 18
        width_post = 2028
        self.fc1 = nn.Linear(width_pre, width_post)
        self.bn1 = nn.BatchNorm1d(width_post)
        self.dropout1 = nn.Dropout(0.5)

        width_pre = width_post
        width_post = 512
        self.fc2 = nn.Linear(width_pre, width_post)
        self.bn2 = nn.BatchNorm1d(width_post)
        self.dropout2 = nn.Dropout(0.5)

        width_pre = width_post
        width_post = 2028
        self.fc3 = nn.Linear(width_pre, width_post)
        self.bn3 = nn.BatchNorm1d(width_post)
        self.dropout3 = nn.Dropout(0.5)

        width_pre = width_post
        self.fc4 = nn.Linear(width_pre, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc4(x))
        return x




















###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################








import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset









vol_kernel_size_1min = 6*1
vol_kernel_size_5min = 6*5
vol_kernel_size_10min = 6*10






def load_file( filename ):



    print( "-------------------------- Reading data file: %s ---------------------------" % ( filename ) )
    raw_data = np.genfromtxt(filename, delimiter=',')
    print( "Done" )



    hour = []
    weekday = []
    month = []
    date_list = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            datetime = Datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S")

            hour.append( datetime.hour )
            weekday.append( datetime.weekday() )
            month.append( datetime.month )
            date_list.append( datetime.date().strftime('%Y-%m-%d') )



    close_price = raw_data[:,6]
    close_price_unscaled = raw_data[:,6]
    # close_price = close_price / close_price[0]


    close_price_tensor_unscaled = torch.tensor(close_price_unscaled).view(-1, 1).squeeze().float().to(device)

    volume = raw_data[:,8]

    volume_tensor = torch.tensor(volume).view(-1, 1).squeeze().float().to(device)
    hour_tensor = torch.tensor(hour).view(-1, 1).squeeze().float().to(device)
    weekday_tensor = torch.tensor(weekday).view(-1, 1).squeeze().float().to(device)
    month_tensor = torch.tensor(month).view(-1, 1).squeeze().float().to(device)





    factor = close_price[0]
    for i in range( len( close_price ) ):
        if( i > 0 and weekday_tensor[i] != weekday_tensor[i-1] ):
            factor = close_price[i]
        
        close_price[i] = close_price[i] / factor








    close_price_tensor = torch.tensor(close_price).view(-1, 1).squeeze().float().to(device)

    print ( close_price )


    np.savetxt('caca.csv', close_price)

    # generate_fft( close_price_tensor )



    print( "genrating FFT")
    auc_list = [ 0 ]
    day_index = 0
    if( 1 ):
        # Compute FFT for each point considering all prior prices
        for i in range(1, len(close_price_tensor) ):
            if( date_list[i] != date_list[i-1] ):
                # print( date_list[i-1], ": ", auc )
                day_index = i

            auc = 0
            if( day_index != i ):
                subset = close_price_tensor[day_index:i]  # Consider all prices up to the current point
                auc = generate_fft( subset )

            auc_list.append( auc )

    auc_tensor = torch.tensor( auc_list ).unsqueeze(1)

    print( auc_tensor.shape )
    print( "done")

    # # Plotting the FFT magnitude
    # plt.figure(figsize=(10, 6))
    # plt.plot(auc_list, label='FFT Magnitude')

    # plt.title( "AUC")
    # plt.ylim(0, 2 )  # Set y-axis range
    # plt.xlabel('Frequency')
    # plt.ylabel('Magnitude')
    # plt.legend()
    # plt.show()



















    print( "$$$$$$$$$$$$$$$$$$$$$$$$$$")
    # print(unique_strings)




    day_offset_index = 15 * 6
    kernel = torch.ones(day_offset_index, device=device).float() / day_offset_index
    vol_moving_averages = F.conv1d(volume_tensor.view(1, 1, -1), weight=kernel.view(1, 1, -1)).view(-1, 1).squeeze()
    vol_moving_averages = prepend( vol_moving_averages, day_offset_index, 1, device )[:-1]

    print( vol_moving_averages )




    kernel = torch.ones(vol_kernel_size_1min, device=device).float() / vol_kernel_size_1min
    vol_var_1min = F.conv1d(volume_tensor.view(1, 1, -1), weight=kernel.view(1, 1, -1)).view(-1, 1).squeeze()
    vol_var_1min = prepend( vol_var_1min, vol_kernel_size_1min, 0, device )[:-1]
    # vol_var_1min = torch.cat((kernel, vol_var_1min), dim=0)
    # vol_var_1min = ( volume_tensor / vol_var_1min ) > vol_1min_var_threshold

    kernel = torch.ones(vol_kernel_size_5min, device=device).float() / vol_kernel_size_5min
    vol_var_5min = F.conv1d(volume_tensor.view(1, 1, -1), weight=kernel.view(1, 1, -1)).view(-1, 1).squeeze()
    vol_var_5min = prepend( vol_var_5min, vol_kernel_size_5min, 0, device )[:-1]
    # vol_var_5min = torch.cat((kernel, vol_var_5min), dim=0)
    # vol_var_5min = ( volume_tensor / vol_var_5min ) > vol_5min_var_threshold


    kernel = torch.ones(vol_kernel_size_10min, device=device).float() / vol_kernel_size_10min
    vol_var_10min = F.conv1d(volume_tensor.view(1, 1, -1), weight=kernel.view(1, 1, -1)).view(-1, 1).squeeze()
    vol_var_10min = prepend( vol_var_10min, vol_kernel_size_10min, 0, device )[:-1]
    # vol_var_10min = torch.cat((kernel, vol_var_10min), dim=0)
    # vol_var_10min = ( volume_tensor / vol_var_10min ) > vol_10min_var_threshold




    vol_var_1min = vol_var_1min / vol_moving_averages
    vol_var_5min = vol_var_5min / vol_moving_averages
    vol_var_10min = vol_var_10min / vol_moving_averages















    if( 0 ):
        print( "+++++++++++++++++++++++++++++++")
        print( close_price_tensor.shape )
        print( vol_moving_averages.shape )
        print( vol_var_1min.shape )
        print( vol_var_5min.shape )
        print( vol_var_10min.shape )




        print( "+++++++++++++++++++++++++++++++")
        print( close_price_tensor.shape )

        print( vol_moving_averages.shape )

        print( vol_var_1min.shape )
        print( vol_var_5min.shape )
        print( vol_var_10min.shape )


        print( "***********************")

        print( vol_moving_averages )



    hour_tensor = hour_tensor / 24
    weekday_tensor = weekday_tensor / 7
    month_tensor = month_tensor / 12









    obv_values = calculate_obv(close_price_tensor, volume_tensor, date_list)

    kernel_size_5min = 6*5
    kernel = torch.ones(kernel_size_5min, device=device).float() / kernel_size_5min
    obv_avg_5min = torch.abs( F.conv1d(obv_values.view(1, 1, -1), weight=kernel.view(1, 1, -1)).view(-1, 1).squeeze() )
    obv_avg_5min = prepend( obv_avg_5min, kernel_size_5min, 0, device )[:-1]


    kernel_size_10min = 6*10
    kernel = torch.ones(kernel_size_10min, device=device).float() / kernel_size_10min
    obv_avg_10min = torch.abs( F.conv1d(obv_values.view(1, 1, -1), weight=kernel.view(1, 1, -1)).view(-1, 1).squeeze() )
    obv_avg_10min = prepend( obv_avg_10min, kernel_size_10min, 0, device )[:-1]


    obv_values_5min = obv_values / obv_avg_5min
    obv_values_10min = obv_values / obv_avg_10min










    print( close_price_tensor.shape )
    rsi_values = calculate_rsi(close_price_tensor).float()
    macd_line, signal_line, histogram = calculate_macd(close_price_tensor)
    # bb_middle_band, bb_upper_band, bb_lower_band = calculate_bollinger_bands(close_price_tensor)

    # padding = len(close_price_tensor) - len(bb_middle_band)
    # if padding > 0:
    #     bb_middle_band = torch.cat([torch.ones(padding), bb_middle_band])
    #     bb_upper_band = torch.cat([torch.ones(padding), bb_upper_band])
    #     bb_lower_band = torch.cat([torch.ones(padding), bb_lower_band])






    volume_tensor = volume_tensor.unsqueeze(1)
    hour_tensor = hour_tensor.unsqueeze(1)
    weekday_tensor = weekday_tensor.unsqueeze(1)
    month_tensor = month_tensor.unsqueeze(1)



    close_price_tensor_unscaled = close_price_tensor_unscaled.unsqueeze(1)









    if( 0 ):
        print( "---------------------------")

        print( vol_var_1min )
        print( "---------------------------")

        print( vol_var_5min )
        print( "---------------------------")

        print( vol_var_10min )

        print( close_price_tensor.shape )
        print( volume_tensor.shape )
        print( hour_tensor.shape )
        print( weekday_tensor.shape )
        print( month_tensor.shape )
        print( vol_moving_averages.shape )
        print( vol_var_1min_thresholded.shape )
        print( vol_var_5min_thresholded.shape )
        print( vol_var_10min_thresholded.shape )

        print( price_var_1min.shape )
        print( price_var_5min.shape )
        print( price_var_10min.shape )



    # if( 1 ):



    return ( close_price_tensor_unscaled, volume_tensor, hour_tensor, weekday_tensor, month_tensor,
            close_price_tensor, auc_tensor, vol_var_1min, vol_var_5min, vol_var_10min, obv_values_5min, obv_values_10min,
            rsi_values, macd_line, signal_line, histogram, date_list )



####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################







# reference_file = "../common_10s_20231112213000-etsy.csv"
# user_file = "../common_10s_20231112213000-rig.csv"
# reference_file = "../common_10s_20231112213000-mara.csv"
# user_file = "../common_10s_20231112213000-tsla.csv"
reference_file = "../common_10s_20231112213000-upst.csv"
# user_file = "../common_10s_20231112213000-pton.csv"




# reference_file = "../common_10s_20231112213000.csv"




training_file = reference_file
( close_price_tensor_unscaled, volume_tensor, hour_tensor, weekday_tensor, month_tensor, close_price_tensor, auc_tensor,
vol_var_1min, vol_var_5min, vol_var_10min, obv_values_5min, obv_values_10min, rsi_values, macd_line, signal_line, histogram, date_list ) = load_file( training_file )





xtesting_file = user_file
( close_price_tensor_unscaled2, volume_tensor2, hour_tensor2, weekday_tensor2, month_tensor2, close_price_tensor2, auc_tensor2,
vol_var_1min2, vol_var_5min2, vol_var_10min2, obv_values_5min2, obv_values_10min2, rsi_values2, macd_line2, signal_line2, histogram2, date_list2 ) = load_file( xtesting_file )




















import torch
import matplotlib.pyplot as plt
from scipy.signal import spectrogram




####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################



if( 1 ):
    print( "generating spectrogram <---------------------------")

    from collections import defaultdict
    from matplotlib import gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable



    unique_strings = list(set(date_list))

    indices_grouped = {val: [] for val in unique_strings}

    for i, val in enumerate(date_list):
        indices_grouped[val].append(i)






    # i = 0
    # for value in indices_grouped: 
    #     # print( "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     # print( "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     # print( "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     # print( "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     # print( "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     # print( "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     # print( "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     # print( "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     # print( "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     # print( "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     # print(close_price_tensor[indices_grouped[value] ])

    #     generate_spectrogram_global( close_price_tensor[indices_grouped[value]], "spectrogram-" + value + ".png", axs, i ) 
    #     i += 1



    num_spectrograms = len( indices_grouped )


    num_columns = 10 
    num_rows = -(-num_spectrograms // num_columns) 
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(20, 4*num_rows))
    axs = axs.ravel()












    i = 0
    for value in sorted( indices_grouped ):  

        if( 1 ):
            n_fft = 512
            hop_length = n_fft // 32
            window_size = 128

            frequencies, times, specgram = spectrogram(close_price_tensor[indices_grouped[value]], nperseg=window_size, noverlap=window_size-hop_length)
            im = axs[i].pcolormesh(times, frequencies, 10 * np.log10(specgram), shading='auto')


        else:


            n_fft = 256
            hop_length = n_fft // 4
            window = torch.hann_window(n_fft)
            specgram = torch.stft(close_price_tensor[indices_grouped[value]], n_fft=n_fft, hop_length=hop_length, window=window)



            magnitude_specgram = torch.abs(specgram)

            im = axs[i].imshow(magnitude_specgram[0].log2().numpy(), aspect='auto', origin='lower')
            axs[i].set_title(value)
            # axs[i].grid(False)
            # axs[i].set_yticks([]) 
            # axs[i].set_xlim(-0.5,0)
        i += 1

    for j in range(num_spectrograms, num_columns * num_rows):
        axs[j].axis('off')

    divider = make_axes_locatable(axs[-1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('Magnitude (log scale)')


    plt.tight_layout()
    plt.savefig( "spectrogram.png" )




    print( "Done!")










####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################


if( 1 ):
    print( "generating spectral contrast <---------------------------")


    import torch
    import torchaudio
    import matplotlib.pyplot as plt






    n_fft = 2048
    hop_length = 64
    win_length = 400
    n_mels = 32

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=44100,
        n_mfcc=n_mels,
        melkwargs={'n_fft': n_fft, 'hop_length': hop_length, 'win_length': win_length}
    )







    num_spectrograms = len( indices_grouped )
    num_columns = 10
    num_rows = -(-num_spectrograms // num_columns)
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(20, 4*num_rows))
    axs = axs.ravel()

    i = 0
    for value in sorted( indices_grouped ):  

        mfcc = mfcc_transform(close_price_tensor[indices_grouped[value]])
        spectral_contrast = mfcc[:, 1:] - mfcc[:, :-1]
        im = axs[i].imshow(spectral_contrast.squeeze().numpy(), aspect='auto', origin='lower')
        axs[i].set_title(value)
        i += 1

    for j in range(num_spectrograms, num_columns * num_rows):
        axs[j].axis('off')



    divider = make_axes_locatable(axs[-1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('Magnitude (log scale)')


    plt.tight_layout()
    plt.savefig( "spectral-contrast.png" )







print( "Done!")








#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################










def prepare_data( close_price_tensor_unscaled, volume_tensor, hour_tensor, weekday_tensor, month_tensor, close_price_tensor, auc_tensor,
                vol_var_1min, vol_var_5min, vol_var_10min, obv_values_5min, obv_values_10min, rsi_values, macd_line, signal_line, histogram, 
                vol_1min_var_threshold, vol_5min_var_threshold, vol_10min_var_threshold, price_1min_var_threshold, price_5min_var_threshold, price_10min_var_threshold,
                rsi_threshold, macd_threshold, signal_threshold, ovb_avg_5min_threshold, ovb_avg_10min_threshold, auc_threshold, buy_threshold ):


    price_var_1min = torch.zeros(vol_var_10min.shape[0], device=device).float()
    price_var_5min = torch.zeros(vol_var_10min.shape[0], device=device).float()
    price_var_10min = torch.zeros(vol_var_10min.shape[0], device=device).float()


    price_var_1min[vol_kernel_size_1min:] = close_price_tensor[vol_kernel_size_1min:] / close_price_tensor[:-vol_kernel_size_1min] > price_1min_var_threshold
    price_var_5min[vol_kernel_size_5min:] = close_price_tensor[vol_kernel_size_5min:] / close_price_tensor[:-vol_kernel_size_5min] > price_5min_var_threshold
    price_var_10min[vol_kernel_size_10min:] = close_price_tensor[vol_kernel_size_10min:] / close_price_tensor[:-vol_kernel_size_10min] > price_10min_var_threshold

    vol_var_1min_thresholded = vol_var_1min > vol_1min_var_threshold
    vol_var_5min_thresholded = vol_var_5min > vol_5min_var_threshold
    vol_var_10min_thresholded = vol_var_10min > vol_10min_var_threshold



    obv_5min_low_thresholded = obv_values_5min < -ovb_avg_5min_threshold
    obv_5min_high_thresholded = obv_values_5min > ovb_avg_5min_threshold
    obv_10min_low_thresholded = obv_values_10min < -ovb_avg_10min_threshold
    obv_10min_high_thresholded = obv_values_10min > ovb_avg_10min_threshold



    close_price_tensor_unsqueezed = close_price_tensor.unsqueeze(1)


    auc_tensor_thresholded = auc_tensor > auc_threshold



    max_values_after_each = torch.zeros( vol_var_10min.shape[0] )
    max_values_after_each[-1] = 0

    max_value = 0
    for i in range(len(close_price_tensor_unsqueezed) - 2, -1, -1):
        # print( weekday_tensor[i] ) 
        if( weekday_tensor[i,0] != weekday_tensor[i+1,0] ):
            max_value = 0
        if( close_price_tensor_unsqueezed[i+1] > max_value ):
            max_value = close_price_tensor_unsqueezed[i+1]
        
        max_values_after_each[i] = max_value


    max_values_after_each = max_values_after_each.unsqueeze(1)


    buy_flag = max_values_after_each > buy_threshold




    if( 0 ):
        with numpy.printoptions(threshold=numpy.inf):
            print( max_values_after_each )
            print( buy_flag )









    # print( close_price_tensor )
    # print( volume_tensor )
    # print( hour_tensor )
    # print( weekday_tensor )
    # print( month_tensor )
    # print( vol_moving_averages )
    # print( vol_var_1min )
    # print( vol_var_5min )
    # print( vol_var_10min )
    # print( buy_flag )


    # print( "+++++++++++++++++++++++++++" )
    # print( torch.sum( buy_flag ) )



    # close_price_tensor_unscaled = close_price_tensor_unscaled.unsqueeze(1)
    # vol_moving_averages = vol_moving_averages.unsqueeze(1)

    # vol_var_1min_thresholded = vol_var_1min_thresholded.unsqueeze(1)
    # vol_var_5min_thresholded = vol_var_5min_thresholded.unsqueeze(1)
    # vol_var_10min_thresholded = vol_var_10min_thresholded.unsqueeze(1)

    # price_var_1min = price_var_1min.unsqueeze(1)
    # price_var_5min = price_var_5min.unsqueeze(1)
    # price_var_10min = price_var_10min.unsqueeze(1)


    close_price_tensor_unsqueezed = close_price_tensor_unsqueezed.cpu()
    price_var_1min = price_var_1min.cpu()
    price_var_5min = price_var_5min.cpu()
    price_var_10min = price_var_10min.cpu()



    print( torch.sum( buy_flag ) )
    print( buy_flag.shape )








    # rsi_values = rsi_values.float()

    # temp = torch.where(rsi_values > 70, 1, 0.5).float()
    # thresholded_rsi = torch.where(rsi_values < 30, 0, temp)
    if( rsi_threshold ):
        thresholded_rsi = torch.where(rsi_values < 30.0, torch.tensor(0.0), torch.where(rsi_values > 70.0, torch.tensor(1.0), torch.tensor(0.5)))
    else:
        thresholded_rsi = torch.where(rsi_values < 20.0, torch.tensor(0.0), torch.where(rsi_values > 80.0, torch.tensor(1.0), torch.tensor(0.5)))

    thresholded_macd_line = torch.where(macd_line < -macd_threshold, torch.tensor(0.0), torch.where(macd_line > macd_threshold, torch.tensor(1.0), torch.tensor(0.5)))
    thresholded_signal_line = torch.where(signal_line < -signal_threshold, torch.tensor(0.0), torch.where(signal_line > signal_threshold, torch.tensor(1.0), torch.tensor(0.5)))
    # thresholded_histogram = torch.where(histogram < 30.0, torch.tensor(0.0), torch.where(histogram > 70.0, torch.tensor(1.0), torch.tensor(0.5)))





    thresholded_rsi = thresholded_rsi.unsqueeze(1)
    thresholded_macd_line = thresholded_macd_line.unsqueeze(1)
    thresholded_signal_line = thresholded_signal_line.unsqueeze(1)
    # thresholded_histogram = thresholded_histogram.unsqueeze(1)



    obv_5min_low_thresholded = obv_5min_low_thresholded.unsqueeze(1)
    obv_5min_high_thresholded = obv_5min_high_thresholded.unsqueeze(1)
    obv_10min_low_thresholded = obv_10min_low_thresholded.unsqueeze(1)
    obv_10min_high_thresholded = obv_10min_high_thresholded.unsqueeze(1)






    print( thresholded_rsi.shape )
    print( thresholded_macd_line.shape )
    print( thresholded_signal_line.shape )
    # print( thresholded_histogram.shape )



    if( 0 ):
        x = range(len(close_price_tensor))

        # Plotting
        plt.figure(figsize=(12, 8))

        # Plot Prices
        plt.subplot(3, 1, 1)
        plt.plot(x, close_price_tensor, label='Prices')
        plt.title('Price and Indicators')
        plt.legend()

        # Plot RSI
        plt.subplot(3, 1, 2)
        plt.plot(x, rsi_values, label='RSI')
        plt.axhline(y=70, color='r', linestyle='--')  # Overbought line
        plt.axhline(y=30, color='g', linestyle='--')  # Oversold line
        plt.title('RSI')
        plt.legend()

        # Plot MACD and Signal Line
        plt.subplot(3, 1, 3)
        plt.plot(x, macd_line, label='MACD Line', color='blue')
        plt.plot(x, signal_line, label='Signal Line', color='orange')
        plt.bar(x, histogram, label='Histogram', color='gray')
        plt.title('MACD')
        plt.legend()

        plt.tight_layout()
        plt.savefig( "caca.png")




    # stacked_tensors = torch.cat((close_price_tensor, hour_tensor, weekday_tensor, month_tensor, vol_var_1min, vol_var_5min, vol_var_10min, price_var_1min, buy_flag), dim=1).to( device)
    # stacked_tensors = torch.cat((close_price_tensor, volume_tensor, hour_tensor, weekday_tensor, month_tensor, vol_moving_averages, vol_var_1min, vol_var_5min, vol_var_10min, buy_flag), dim=1).to( device)
    # stacked_tensors = torch.cat((close_price_tensor_unscaled, close_price_tensor, hour_tensor, weekday_tensor, month_tensor, vol_var_1min, vol_var_5min, vol_var_10min, price_var_1min, price_var_5min, price_var_10min, buy_flag), dim=1).to( device)
    stacked_tensors = torch.cat((close_price_tensor_unsqueezed, hour_tensor, weekday_tensor, month_tensor,
                                vol_var_1min_thresholded.unsqueeze(1), vol_var_5min_thresholded.unsqueeze(1), vol_var_10min_thresholded.unsqueeze(1), 
                                price_var_1min.unsqueeze(1), price_var_5min.unsqueeze(1), price_var_10min.unsqueeze(1), thresholded_rsi, thresholded_macd_line,
                                thresholded_signal_line, obv_5min_low_thresholded, obv_5min_high_thresholded, obv_10min_low_thresholded, obv_10min_high_thresholded, auc_tensor_thresholded, buy_flag), dim=1)



    return stacked_tensors
















#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################

from itertools import product
import random


vol_1min_var_vector = [ 0.9, 1.1 ]
vol_5min_var_vector = [ 0.95, 1.05 ]
vol_10min_var_vector = [ 0.95, 1.05 ]

price_1min_var_vector = [ 0.9, 1.1 ]
price_5min_var_vector = [ 0.9, 1.1 ]
price_10min_var_vector = [ 0.9, 1.1 ]

rsi_vector = [ 0, 1 ]
macd_vector = [ 5, 10, 20 ]
signal_vector = [ 5, 10, 20 ]

ovb_avg_5min_vector = [ 1, 5 ]
ovb_avg_10min_vector = [ 1, 3 ]

auc_vector = [ 0.5, 0.6 ]

buy_vector = [ 1.02, 1.05 ]









all_combinations = list(product(vol_1min_var_vector, vol_5min_var_vector, vol_10min_var_vector, price_1min_var_vector, price_5min_var_vector, price_10min_var_vector, rsi_vector, macd_vector, signal_vector, ovb_avg_5min_vector, ovb_avg_10min_vector, auc_vector, buy_vector ))

combinations = random.sample(all_combinations, 200)

best_precision = 0
best_config = ""
best_true_positive = 0
best_confusion_matrix = torch.zeros(2, 2)
best_metrics = ""

best_precision2 = 0
best_config2 = ""
best_true_positive2 = 0
best_confusion_matrix2 = torch.zeros(2, 2)
best_metrics2 = ""

for ( vol_1min_var_threshold, vol_5min_var_threshold, vol_10min_var_threshold, price_1min_var_threshold, price_5min_var_threshold, price_10min_var_threshold, rsi_threshold, macd_threshold, signal_threshold, ovb_avg_5min_threshold, ovb_avg_10min_threshold, auc_threshold, buy_threshold ) in combinations:


    config = "{:.2f}-{:.2f}-{:.2f}-{:.2f}-{:.2f}-{:.2f}-{:.2f}-{:.2f}-{:.2f}-{:.2f}-{:.2f}-{:.2f}-{:.2f}".format( vol_1min_var_threshold, vol_5min_var_threshold, vol_10min_var_threshold, price_1min_var_threshold, price_5min_var_threshold,
                                                                                                            price_10min_var_threshold, rsi_threshold, macd_threshold, signal_threshold, ovb_avg_5min_threshold, ovb_avg_10min_threshold, auc_threshold, buy_threshold )

    print( "current config : ", config )



    stacked_tensors = prepare_data( close_price_tensor_unscaled, volume_tensor, hour_tensor, weekday_tensor, month_tensor, close_price_tensor, auc_tensor,
                                    vol_var_1min, vol_var_5min, vol_var_10min, obv_values_5min, obv_values_10min, rsi_values, macd_line, signal_line, histogram,
                                    vol_1min_var_threshold, vol_5min_var_threshold, vol_10min_var_threshold,
                                    price_1min_var_threshold, price_5min_var_threshold, price_10min_var_threshold,
                                    rsi_threshold, macd_threshold, signal_threshold, ovb_avg_5min_threshold, ovb_avg_10min_threshold, auc_threshold, buy_threshold )


    stacked_tensors2 = prepare_data( close_price_tensor_unscaled2, volume_tensor2, hour_tensor2, weekday_tensor2, month_tensor2, close_price_tensor2, auc_tensor2,
                                    vol_var_1min2, vol_var_5min2, vol_var_10min2, obv_values_5min2, obv_values_10min2, rsi_values2, macd_line2, signal_line2, histogram2,
                                    vol_1min_var_threshold, vol_5min_var_threshold, vol_10min_var_threshold,
                                    price_1min_var_threshold, price_5min_var_threshold, price_10min_var_threshold,
                                    rsi_threshold, macd_threshold, signal_threshold, ovb_avg_5min_threshold, ovb_avg_10min_threshold, auc_threshold, buy_threshold )



    np.savetxt('stacked_tensors2.csv', stacked_tensors2.numpy())






    dataset_size = len(stacked_tensors)
    indices = torch.randperm(dataset_size)

    train_size = int(0.8 * dataset_size) 
    test_size = dataset_size - train_size 

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_set = stacked_tensors[train_indices]
    test_set = stacked_tensors[test_indices]

    train_data = torch.tensor( train_set[:,:-1] )
    train_label = torch.tensor( train_set[:,-1].unsqueeze(1) )
    
    test_data = torch.tensor( test_set[:,:-1] )
    test_label = torch.tensor( test_set[:,-1].unsqueeze(1) )
    




    dataset_size = len(stacked_tensors2)
    indices = torch.randperm(dataset_size)
    test_size = dataset_size - train_size 
    test_indices = indices[train_size:]
    test_set2 = stacked_tensors2[test_indices]
    test_data2 = torch.tensor( test_set2[:,:-1] )
    test_label2 = torch.tensor( test_set2[:,-1].unsqueeze(1) )
    















    if( user_device != "" ):
        device = user_device
    else:
        gpu_flag = 1
        if( gpu_flag ):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = "cpu"





    # print( "------------------- device -----------------------")
    # print( device )




    # model = LSTMModel( 7, 4096, 10, 1 ).to( device )
    # model = NeuralNetworkSimple().to( device )
    # model = NeuralNetworkHuge().to( device )
    model = NeuralNetwork().to( device )
    # model = CNN1D( 10 ).to( device )
    


    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)



    print( "####################### starting to train #######################")

    batch_size = 400000 
    batch_size = 40000
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


    test_dataset2 = torch.utils.data.TensorDataset(test_data2, test_label2)
    test_loader2 = torch.utils.data.DataLoader(test_dataset2, batch_size=batch_size)







    #####################################################################################
    #####################################################################################
    #####################################################################################
    #####################################################################################
    #####################################################################################
    #
    # TRAINING

    num_epochs = 50
    loss_values = []
    accuracy_values = []

    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to( device )
            labels = labels.to( device )

            # print( "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # print( inputs.shape )
            # print( labels.shape )


            np.savetxt('train-label.csv', labels.cpu().numpy() )

            # print( inputs )
            # print( labels )

            optimizer.zero_grad()  
            outputs = model(inputs) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  
            running_loss += loss.item()

            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        accuracy_values.append(epoch_accuracy)  # Store accuracy for this epoch
        loss_values.append(epoch_loss)  # Store loss for this epoch

        # scheduler.step(epoch_loss)

        print( epoch, end=' ', flush=True )
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}, Accuracy: {epoch_accuracy}")


        if( epoch_accuracy >= 0.99 ):
            break


    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(loss_values) + 1), loss_values, label='Loss', color='red')
    plt.plot(range(1, len(accuracy_values) + 1), accuracy_values, label='Accuracy', color='blue')
    plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch - ()'.format( user_tag ) )
    plt.legend()
    plt.grid(True)

    plt.savefig( "training-{}-{}.png".format( config, user_tag ) )





    #####################################################################################
    #####################################################################################
    #####################################################################################
    #####################################################################################
    #####################################################################################
    #
    # TESTING

    print( "" )
    print( "####################### starting to test #######################")
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix



    conf_matrix = torch.zeros(2, 2)

    model.eval() 
    total_test_loss = 0.0
    correct = 0
    total = 0
    # total_sum = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data

            inputs = inputs.to( device )
            labels = labels.to( device )

            test_outputs = model(inputs)
            test_loss = criterion(test_outputs, labels)
            total_test_loss += test_loss.item()

            # print( test_outputs )

            predicted = (test_outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            conf_matrix += confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), labels=[0, 1])

        confusion_matrix_sum = torch.sum( conf_matrix )
        precision = ( conf_matrix[0, 0] + conf_matrix[1, 1] ) / confusion_matrix_sum
        true_positive = conf_matrix[1, 1] / confusion_matrix_sum
        if( ( precision + 1.5 * true_positive ) > ( best_precision + 1.5 * best_true_positive ) and true_positive > 0.15 ):
            print( "NEW BEST CONFIG: ", config )
            best_precision = precision
            best_config = config
            best_true_positive = true_positive
            best_confusion_matrix = conf_matrix


            save_confusion_matrix( best_confusion_matrix / torch.sum( best_confusion_matrix ) * 100, "best-testing-confusion-matrix-{}-{}.png".format( config,user_tag ) )
            best_metrics = compute_metrics( predicted, labels )




        print(f"precision {precision}, true_positive: {true_positive}")
        print( best_confusion_matrix )
        print( best_confusion_matrix / torch.sum( best_confusion_matrix ) )

            # total_sum += torch.sum( test_outputs )

    print( "#### current config: ", user_tag, ",  ", config, "  precision: ", precision, "  true_positive: ", true_positive ,  "    #####" )

    print( "---------------------> best config: ", user_tag, ",  ", best_config, "  best_precision: ", best_precision, "  best_true_positive: ", best_true_positive ,  "    <--------------------------------------------------------" )
    print(f"Test Loss: {total_test_loss / len(test_loader)}")
    print( "best metrics: ", best_metrics )


    print( conf_matrix )
    # print( total_sum )












    #####################################################################################
    #####################################################################################
    #####################################################################################
    #####################################################################################
    #####################################################################################
    #
    # CROSS TRAINING

    print( "###################################################################################")
    print( "###################################################################################")
    print( "###")
    print( "X testing")







    conf_matrix = torch.zeros(2, 2)
    test_data = torch.tensor( test_set[:,:-1] )

    model.eval() 
    total_test_loss = 0.0
    correct = 0
    total = 0
    # total_sum = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader2, 0):
            inputs, labels = data

            inputs = inputs.to( device )
            labels = labels.to( device )

            test_outputs = model(inputs)
            test_loss = criterion(test_outputs, labels)
            total_test_loss += test_loss.item()

            # print( test_outputs )

            predicted = (test_outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            conf_matrix += confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), labels=[0, 1])

        confusion_matrix_sum = torch.sum( conf_matrix )
        precision = ( conf_matrix[0, 0] + conf_matrix[1, 1] ) / confusion_matrix_sum
        true_positive = conf_matrix[1, 1] / confusion_matrix_sum


        print(f"precision {precision}, true_positive: {true_positive}")
        print( conf_matrix[0, 0] )
        print( conf_matrix[0, 1] )
        print( conf_matrix[1, 0] )
        print( conf_matrix[1, 1] )


        if( ( precision + 1.5 * true_positive ) > ( best_precision2 + 1.5 * best_true_positive2 ) and true_positive > 0.15 ):
            print( "NEW BEST CONFIG: ", config )
            best_precision2 = precision
            best_config2 = config
            best_true_positive2 = true_positive
            best_confusion_matrix2 = conf_matrix

            save_confusion_matrix( best_confusion_matrix2 / torch.sum( best_confusion_matrix2 ) * 100, "best-xtesting-confusion-matrix-{}-{}.png".format( config,user_tag ) )
            best_metrics2 = compute_metrics( predicted, labels )
            


        print( "$$$$$$$$$$$$$$$$$$$$$$$ BEST CONFIG: ", best_config2, " best_precision2: ", best_precision2, " best_true_positive2: ", best_true_positive2 )
        print( best_confusion_matrix2 )
        print( best_confusion_matrix2 / torch.sum( best_confusion_matrix2 ) )
        print( "training_file: ", training_file )
        print( "xtesting_file: ", xtesting_file )
        print( "xtesting best confusion matrix: ", best_confusion_matrix2 )
        print( "xtesting best metrics ", best_metrics2 )


    print(f"Test Loss: {total_test_loss / len(test_loader)}")
    print( conf_matrix )
    # print( total_sum )


    print( best_metrics )
    print( "###")
    print( "###################################################################################")
    print( "###################################################################################")






















