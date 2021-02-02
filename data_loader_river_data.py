import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import pytz
import torch

#Adjust this function to the relevant dataset
def read_river_data( data_folder, file_name):
    dataset = pd.read_csv(data_folder + file_name + '.csv', header=0, low_memory=False,
                             infer_datetime_format=True, parse_dates={'datetime': [0]}, index_col=['datetime'], usecols=[0,2])
    dataset = dataset[(dataset.index >= '2012-12-19')  & (dataset.index <= '2017-11-28') ]  #rain data only until end 2017
    return dataset

def read_nerf_data(data_folder, file_name):
    dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y')
    dataset = pd.read_csv(data_folder + file_name + '.csv', header=0, low_memory=False,
                             usecols=[0, 1])
    dataset.index=  pd.to_datetime(dataset.date, format='%d/%m/%Y')
    dataset= dataset.drop(columns=['date'])

    dataset = dataset[(dataset.index > '1982')& (dataset.index <'2018') ]
    print(dataset.isna().sum(), 'isna')
    dataset = dataset.dropna(axis=0)
    dataset = dataset.sort_index()

    return dataset


def combine_data(main_df, cond_df):
    cols = list(main_df) + list(cond_df)
    dataset = main_df.merge(cond_df,  left_index=True, right_index=True)
    dataset.columns = cols
    return dataset



class DataLoader(object):
    def __init__(self, pred_var, sliding_window, predict_size, input_size, base_path, data_folder, sub_folder, cond_vars):
        """
        :param xs:
        :param ys:
        :param batch_size:
        """

        self.tw= sliding_window
        self.predict_size= predict_size
        self.input_size= input_size
        self.pred_var= pred_var
        self.sliding_window=sliding_window

        self.base_path = base_path
        self.data_folder = data_folder
        self.sub_folder = sub_folder

        if self.sub_folder == 'darlaston':
            self.flow_data = '28083_gdf_clean'
            self.rain_data = '28083_cdr_clean'
            self.stage_data = 'river-trent-stone-rural-darlaston'

        self.cond_vars = {k: v for k, v in cond_vars.items() if v is not False}

        #Specify which variables should have a scaler (apart from prediction variable which has a scaler)
        self.scaler=MinMaxScaler(feature_range=(0, 1))
        scaler_vars=['stage' , 'flow', 'rain']
        self.scaler_stage = MinMaxScaler(feature_range=(0, 1))
        self.scaler_flow = MinMaxScaler(feature_range=(0, 1))
        self.scaler_rain = MinMaxScaler(feature_range=(0, 1))
        self.scalers = list()
        self.scaler_index =dict()
        self.scaler_index['self.scaler']=0
        self.scaler_name=['self.scaler']

        for key in self.cond_vars:
            if key in scaler_vars:
                self.scaler_name.append('self.scaler_'+ key)
                self.scaler_index['self.scaler_'+ key] = list(self.cond_vars).index(key) + 1

        super(DataLoader, self).__init__()

    def load_data(self):
        # load the dataset
        if self.sub_folder in ['darlaston' , 'north_muskham']:
            if 'stage' == self.pred_var:
                dataset = read_river_data(self.data_folder, self.stage_data)
            elif 'flow' == self.pred_var:
                dataset = read_nerf_data(self.data_folder, self.flow_data )
            else:
                print('Enter valid prediction variable')

            if 'stage' in self.cond_vars.keys():
                print('stage is true')
                add_data= read_river_data(self.data_folder, self.stage_data)
                dataset=combine_data(dataset, add_data)
            if 'flow' in self.cond_vars.keys():
                add_data = read_nerf_data(self.data_folder, self.flow_data )
                dataset=combine_data(dataset, add_data)
            if 'rain' in self.cond_vars.keys():
                add_data = read_gold_data(self.data_folder, self.rain_data )
                dataset=combine_data(dataset, add_data)
        else:
            print('Enter valid prediction time series')


        if 'month_sine' in self.cond_vars.keys() and 'month_cosine' in self.cond_vars.keys():
            #get values or one hot encoded months
            month_dummies= pd.DataFrame(dataset.index.month, index=dataset.index)
            month_dummies.columns=['month']
            dataset= dataset.join(month_dummies, how='left')
            #get sine and cosine of years
            dataset['sin_time'] = np.sin(2 * np.pi * dataset.month / 12)
            dataset['cos_time'] = np.cos(2 * np.pi * dataset.month  / 12)
            dataset=dataset.drop(columns=['month'])

        if 'year' in self.cond_vars.keys():
            #get year dummies or values
            year_dummies= pd.DataFrame(dataset.index.year)
            #get years since present
            end= dataset.index[-1]
            dates= dataset.index
            time_since =  end - dates  # calculate timedelta
            years_since = time_since.map(lambda x: round(x.days/365)) # get no of days
            dataset['years']= years_since.tolist()

        dataset = pd.DataFrame(dataset)
        dataset = dataset.astype('float64')
        return dataset

    def split_data(self, dataset):
        # split data
        train_data_end = round(len(dataset) * 0.6)
        val_data_end = round(len(dataset) * 0.8)
        train_data = dataset[:train_data_end]
        val_data = dataset[train_data_end:val_data_end]
        test_data_end= len(dataset)
        test_data = dataset[val_data_end:test_data_end]

        return train_data, val_data, test_data

    def scale_data(self, data):
        #print(data.shape, 'inout shape') #expects S, F  # should expect B,S,F
        scaler_no=0
        for i in range(data.shape[2]):
            if i in self.scaler_index.values():
                scaler_i=self.scaler_index[self.scaler_name[scaler_no]]
                data_scaled =(eval(self.scaler_name[scaler_no]).transform(data[:,:, scaler_i]))
                scaler_no+=1
                if i==0:
                    output =  np.expand_dims(data_scaled, axis=2)
                else:
                    output = np.dstack((output, data_scaled))
            else:
                data_scaled=data[:,:, i]  #.reshape(-1, 1)
                output = np.dstack((output, data_scaled))  # hstack means stack along second axis
        output= torch.FloatTensor(output).view(-1)
        return output

    def create_inout_sequences(self, input_data, tw, predict_size):
        inout_seq = []
        L = len(input_data)
        x = []
        y = []

        for i in range(L - tw):
            train_seq = input_data[i:i + tw]
            train_label = input_data[
                          i + tw:i + tw + predict_size]
            # for conditional model only keep the target variable values
            if input_data.ndim >1:
                train_label = train_label[:, 0]
            inout_seq.append((train_seq, train_label))  # tuple containing the 7 days values and the next value
            if len(train_label) == predict_size:
                x.append(train_seq)
                y.append(train_label)

        return inout_seq  # x, y

    def split_scale_transform(self, dataset):

        train_data, val_data, test_data = self.split_data(dataset)

        train_data = train_data.to_numpy()
        val_data = val_data.to_numpy()
        test_data = test_data.to_numpy()
        train_data = np.expand_dims(train_data, axis=1)
        val_data = np.expand_dims(val_data, axis=1)
        test_data = np.expand_dims(test_data, axis=1)

        #fit scalers
        scaler_no = 0
        for i in range(train_data.shape[2]):
            if i in self.scaler_index.values():
                scaler_i=self.scaler_index[self.scaler_name[scaler_no]]
                eval(self.scaler_name[scaler_no]).fit(train_data[:, :, scaler_i ])
                scaler_no += 1
            else:
                continue

        #scale data with fitted scalers
        train_data_normalized = self.scale_data(train_data)  #shape [samples, timesteps , features]
        val_data_normalized = self.scale_data(val_data)
        test_data_normalized = self.scale_data(test_data)

        if self.sliding_window is not False: # reshape into X=t->t+tw and Y=t+tw+predict_size

            print('sliding window method used', self.tw)
            train_inout = self.create_inout_sequences(train_data_normalized, self.tw, self.predict_size)
            val_inout= self.create_inout_sequences(val_data_normalized, self.tw, self.predict_size)
            test_inout = self.create_inout_sequences(test_data_normalized, self.tw, self.predict_size)


        return  train_inout, val_inout, test_inout

    def scale_back(self, data):
        # this expects a 2 dim input B,S,F
        scaler_no=0
        for i in range(data.shape[2]):
            if i in self.scaler_index.values() :
                if i in self.scaler_index.values():
                    scaler_i = self.scaler_index[self.scaler_name[scaler_no]]
                    data_scaled = eval(self.scaler_name[scaler_no]).inverse_transform(data[:, :, scaler_i])
                    scaler_no += 1
                    if i == 0:
                        output = np.expand_dims(data_scaled, axis=2)
                    else:
                        output = np.dstack((output, data_scaled))
            else:
                data_scaled = data[:, :, i]
                output = np.dstack((output, data_scaled))  # hstack means stack along second axis, dstack along third axis
        return output



