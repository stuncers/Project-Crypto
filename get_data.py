
from binance.client import Client
import sys
import os
import pandas
import numpy

base_dir = 'C:\\Users\\suat.tuncer\\OneDrive - ICRYPEX BILISIM A.S\\Documents\\GitHub\creds\\'

sys.path.append(os.path.abspath(base_dir))

from bnb_creds import api_key, api_secret



#print ("Input examples: pair= 'BTCUSDT' interval = Client.KLINE_INTERVAL_1MINUTE OR DAY OR HOUR start = '4 Aug, 2022' finish = '5 Aug, 2022'","get_hist_data(pair, interval,start,finish)")
def get_hist_data(pair,interval, start,finish):
  import pandas as pd
  api_key = api_key
  api_secret = api_secret
  client = Client(api_key,api_secret)
  klines = client.get_historical_klines(str(pair), interval, str(start), str(finish))
  df = pd.DataFrame(klines, columns =['Open time','Open','High','Low','Close','Volume','Close time','Quote asset volume','Number of trades','Taker buy base asset volume','Taker buy quote asset volume','Ignore'], dtype = float) 
  return df



#This part must be revised
import pandas as pd
def multivar_data(curr_list):
    pair= 'BTCUSDT' 
    interval = Client.KLINE_INTERVAL_1HOUR
    start = '1 Jan, 2020' 
    finish = '1 Mar, 2023'
    df_BTCUSDT=get_hist_data(pair, interval,start,finish)
    df_BTCUSDT['sec_0'] = pd.to_datetime((df_BTCUSDT['Close time']/1000).astype(int), unit='s')
    df_BTCUSDT['BTCUSDT'+'_Close'] = df_BTCUSDT['Close']
    df_merged= df_BTCUSDT[['BTCUSDT'+'_Close','sec_0']]
    for i in curr_list:
        pair= str(i) 
        interval = Client.KLINE_INTERVAL_1HOUR
        start = '1 Jan, 2020' 
        finish = '1 Mar, 2023'
        var_name = "df_" + str(i)
        var_name = get_hist_data(pair, interval,start,finish)
        var_name['sec_0'] = pd.to_datetime(((var_name)['Close time']/1000).astype(int), unit='s')
        var_name[str(i)+'_Close'] = var_name['Close']
        var_name= var_name[['sec_0',str(i)+'_Close']]
        df_merged = pd.merge(var_name, df_merged, how='inner', on=['sec_0'])
    return df_merged










# have to get curr_list over here
  #curr_list = ['BUSDUSDT','ETHUSDT','BNBUSDT' ,'XRPUSDT' ,
  #           'ADAUSDT','DOGEUSDT','MATICUSDT' ,'SOLUSDT','DOTUSDT','LTCUSDT','AVAXUSDT']
merged = multivar_data(curr_list)
#merged.head()
#sec_0	AVAXUSDT_Close	LTCUSDT_Close	DOTUSDT_Close	SOLUSDT_Close	MATICUSDT_Close	DOGEUSDT_Close	ADAUSDT_Close	XRPUSDT_Close	BNBUSDT_Close	ETHUSDT_Close	BUSDUSDT_Close	BTCUSDT_Close
#0	2020-09-22 06:59:59	4.8811	43.35	3.9847	2.8000	0.01906	0.002598	0.08082	0.23230	23.3510	342.05	1.0001	10423.27
#1	2020-09-22 07:59:59	4.9096	43.07	3.9164	2.6728	0.01863	0.002592	0.07962	0.22978	22.9993	336.12	1.0002	10372.90
#2	2020-09-22 08:59:59	6.8219	43.58	4.0194	2.7602	0.01905	0.002616	0.08143	0.23129	23.4484	342.11	1.0003	10463.18
#3	2020-09-22 09:59:59	6.2108	43.57	4.0207	2.7857	0.01923	0.002616	0.08153	0.23144	23.5538	340.09	1.0002	10452.03
#4	2020-09-22 10:59:59	5.4901	43.75	4.0494	2.7847	0.01932	0.002634	0.08198	0.23197	23.8488	341.00	1.0002	10458.40
len(merged)
21332














# Set up the webdriver
import pandas as pd
from selenium.webdriver.common.by import By
from selenium import webdriver
driver = webdriver.Chrome()
driver.get('https://coinmarketcap.com')
page_height = driver.execute_script('return document.body.scrollHeight')
driver.execute_script('window.scrollBy(0, '+str(page_height)+')')
# sleep for 7 seconds for getting data
import time
time.sleep(7)

# Find the table and extract its content
table = driver.find_element(By.XPATH,'//*[@id="__next"]/div/div[1]/div[2]/div/div[1]/div[4]/table')
html = table.get_attribute('outerHTML')
df = pd.read_html(html)[0]
driver.quit()
import pandas as pd
import re

# Sample DataFrame with a "Name" column


# Define a function to extract the cryptocurrency symbol from a string
def get_symbol(s):
    symbol = re.search(r'([^\d]+)$', s).group(1)
    return symbol

# Apply the function to the "Name" column and store the result in a new "Symbol" column

df["Symbol"] = df["Name"].apply(get_symbol)

# Print the resulting DataFrame
df= df.drop(['Unnamed: 0','#','Name','Price','1h %', '24h %', '7d %', 'Market Cap',
       'Volume(24h)', 'Circulating Supply', 'Last 7 Days', 'Unnamed: 11'],axis=1)
# Returning first 100 highest market cap. Market Cap = Current Price x Circulating Supply.
# For more info check for following link: 
#https://support.coinmarketcap.com/hc/en-us/articles/360043836811-Market-Capitalization-Cryptoasset-Aggregate-
print(df.head())
  Symbol
0    BTC
1    ETH
2   USDT
3    BNB
4   USDC
df.head(20)
Symbol
0	BTC
1	ETH
2	USDT
3	BNB
4	USDC
5	XRP
6	ADA
7	DOGE
8	MATIC
9	BUSD
10	SolanaSOL
11	PolkadotDOT
12	LitecoinLTC
13	Shiba InuSHIB
14	TRONTRX
15	AvalancheAVAX
16	DaiDAI
17	UniswapUNI
18	ChainlinkLINK
19	Wrapped BitcoinWBTC
 
 
 
 
 
merged.dtypes
sec_0              datetime64[ns]
AVAXUSDT_Close            float64
LTCUSDT_Close             float64
DOTUSDT_Close             float64
SOLUSDT_Close             float64
MATICUSDT_Close           float64
DOGEUSDT_Close            float64
ADAUSDT_Close             float64
XRPUSDT_Close             float64
BNBUSDT_Close             float64
ETHUSDT_Close             float64
BUSDUSDT_Close            float64
BTCUSDT_Close             float64
dtype: object
col_to_move = 'BTCUSDT_Close'

# remove the column and insert it at the first position
merged.insert(0, col_to_move, merged.pop(col_to_move))

print(merged.head())
   BTCUSDT_Close               sec_0  AVAXUSDT_Close  LTCUSDT_Close  \
0       10423.27 2020-09-22 06:59:59          4.8811          43.35   
1       10372.90 2020-09-22 07:59:59          4.9096          43.07   
2       10463.18 2020-09-22 08:59:59          6.8219          43.58   
3       10452.03 2020-09-22 09:59:59          6.2108          43.57   
4       10458.40 2020-09-22 10:59:59          5.4901          43.75   

   DOTUSDT_Close  SOLUSDT_Close  MATICUSDT_Close  DOGEUSDT_Close  \
0         3.9847         2.8000          0.01906        0.002598   
1         3.9164         2.6728          0.01863        0.002592   
2         4.0194         2.7602          0.01905        0.002616   
3         4.0207         2.7857          0.01923        0.002616   
4         4.0494         2.7847          0.01932        0.002634   

   ADAUSDT_Close  XRPUSDT_Close  BNBUSDT_Close  ETHUSDT_Close  BUSDUSDT_Close  
0        0.08082        0.23230        23.3510         342.05          1.0001  
1        0.07962        0.22978        22.9993         336.12          1.0002  
2        0.08143        0.23129        23.4484         342.11          1.0003  
3        0.08153        0.23144        23.5538         340.09          1.0002  
4        0.08198        0.23197        23.8488         341.00          1.0002  
df_prep = merged.drop('sec_0', axis=1)
import pandas as pd
df_prep = pd.read_csv("multivar.csv").drop('Unnamed: 0',axis=1)
df_prep.head()
BTCUSDT_Close	AVAXUSDT_Close	LTCUSDT_Close	DOTUSDT_Close	SOLUSDT_Close	MATICUSDT_Close	DOGEUSDT_Close	ADAUSDT_Close	XRPUSDT_Close	BNBUSDT_Close	ETHUSDT_Close	BUSDUSDT_Close
0	10423.27	4.8811	43.35	3.9847	2.8000	0.01906	0.002598	0.08082	0.23230	23.3510	342.05	1.0001
1	10372.90	4.9096	43.07	3.9164	2.6728	0.01863	0.002592	0.07962	0.22978	22.9993	336.12	1.0002
2	10463.18	6.8219	43.58	4.0194	2.7602	0.01905	0.002616	0.08143	0.23129	23.4484	342.11	1.0003
3	10452.03	6.2108	43.57	4.0207	2.7857	0.01923	0.002616	0.08153	0.23144	23.5538	340.09	1.0002
4	10458.40	5.4901	43.75	4.0494	2.7847	0.01932	0.002634	0.08198	0.23197	23.8488	341.00	1.0002
import numpy as np
def split_data(series, train_fraq, test_len=120):
    """Splits input series into train, val and test.
    
        Default to 1 year of test data.
    """
    #slice the last year of data for testing 1 year has 8760 hours
    test_slice = len(series)-test_len

    test_data = series[test_slice:]
    train_val_data = series[:test_slice]

    #make train and validation from the remaining
    train_size = int(len(train_val_data) * train_fraq)
    
    train_data = train_val_data[:train_size]
    val_data = train_val_data[train_size:]
    
    return train_data, val_data, test_data




#add hour and month features

train_multi, val_multi, test_multi = split_data(df_prep, train_fraq=0.65, test_len=15)
print("Multivarate Datasets")
print(f"Train Data Shape: {train_multi.shape}")
print(f"Val Data Shape: {val_multi.shape}")
print(f"Test Data Shape: {test_multi.shape}")
print(f"Nulls In Train {np.any(np.isnan(train_multi))}")
print(f"Nulls In Validation {np.any(np.isnan(val_multi))}")
print(f"Nulls In Test {np.any(np.isnan(test_multi))}")
Multivarate Datasets
Train Data Shape: (13856, 12)
Val Data Shape: (7461, 12)
Test Data Shape: (15, 12)
Nulls In Train False
Nulls In Validation False
Nulls In Test False
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
def window_dataset(data, n_steps, n_horizon, batch_size, shuffle_buffer, multi_var=False, expand_dims=False):
    """ Create a windowed tensorflow dataset
    
    """

    #create a window with n steps back plus the size of the prediction length
    window = n_steps + n_horizon
    
    #expand dimensions to 3D to fit with LSTM inputs
    #creat the inital tensor dataset
    if expand_dims:
        ds = tf.expand_dims(data, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(ds)
    else:
        ds = tf.data.Dataset.from_tensor_slices(data)
    
    #create the window function shifting the data by the prediction length
    ds = ds.window(window, shift=n_horizon, drop_remainder=True)
    
    #flatten the dataset and batch into the window size
    ds = ds.flat_map(lambda x : x.batch(window))
    ds = ds.shuffle(shuffle_buffer)    
    
    #create the supervised learning problem x and y and batch
    if multi_var:
        ds = ds.map(lambda x : (x[:-n_horizon], x[-n_horizon:, :1]))
    else:
        ds = ds.map(lambda x : (x[:-n_horizon], x[-n_horizon:]))
    
    ds = ds.batch(batch_size).prefetch(1)
    
    return ds

tf.random.set_seed(42)

n_steps = 115
n_horizon = 1
batch_size =128
shuffle_buffer = 100


ds = window_dataset(train_multi, n_steps, n_horizon, batch_size, shuffle_buffer, multi_var=True)

print('Example sample shapes')
for idx,(x,y) in enumerate(ds):
    print("x = ", x.numpy().shape)
    print("y = ", y.numpy().shape)
    break
Example sample shapes
x =  (128, 115, 12)
y =  (128, 1, 1)
def build_dataset(train_fraq=0.65, 
                  n_steps=115, 
                  n_horizon=1, 
                  batch_size=128, 
                  shuffle_buffer=100, 
                  expand_dims=False, 
                  multi_var=False):
    """If multi variate then first column is always the column from which the target is contstructed.
    """
    
    tf.random.set_seed(23)
    data = df_prep
    mm = MinMaxScaler()
    data = mm.fit_transform(data)

    
    train_data, val_data, test_data = split_data(data, train_fraq=train_fraq, test_len=125)
    
    train_ds = window_dataset(train_data, n_steps, n_horizon, batch_size, shuffle_buffer, multi_var=multi_var, expand_dims=expand_dims)
    val_ds = window_dataset(val_data, n_steps, n_horizon, batch_size, shuffle_buffer, multi_var=multi_var, expand_dims=expand_dims)
    test_ds = window_dataset(test_data, n_steps, n_horizon, batch_size, shuffle_buffer, multi_var=multi_var, expand_dims=expand_dims)
    
    
    print(f"Prediction lookback (n_steps): {n_steps}")
    print(f"Prediction horizon (n_horizon): {n_horizon}")
    print(f"Batch Size: {batch_size}")
    print("Datasets:")
    print(train_ds.element_spec)
    
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = build_dataset(multi_var=True)
Prediction lookback (n_steps): 115
Prediction horizon (n_horizon): 1
Batch Size: 128
Datasets:
(TensorSpec(shape=(None, None, 12), dtype=tf.float64, name=None), TensorSpec(shape=(None, None, 1), dtype=tf.float64, name=None))
def get_params(multivar=True):
    lr = 0.00001
    n_steps=115
    n_horizon=1
    if multivar:
        n_features=12
    else:
        n_features=1
        
    return n_steps, n_horizon, n_features, lr

model_configs = dict()

def cfg_model_run(model, history, test_ds):
    return {"model": model, "history" : history, "test_ds": test_ds}


from tensorflow.keras.callbacks import EarlyStopping
def run_model(model_name, model_func, model_configs, epochs):
    
    n_steps, n_horizon, n_features, lr = get_params(multivar=True)
    train_ds, val_ds, test_ds = build_dataset(n_steps=n_steps, n_horizon=n_horizon, multi_var=True)

    model = model_func(n_steps, n_horizon, n_features, lr=lr)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, restore_best_weights=True)

    model_hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stopping])

    model_configs[model_name] = cfg_model_run(model, model_hist, test_ds)
    return test_ds
def dnn_model(n_steps, n_horizon, n_features, lr):
    tf.keras.backend.clear_session()
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(n_steps, n_features)),
        tf.keras.layers.Dense(512, activation='tanh'),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(512, activation='tanh'),
        tf.keras.layers.Dropout(0.01),
        tf.keras.layers.Dense(n_horizon)
    ], name='dnn')
    
    loss=tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    model.compile(loss=loss, optimizer='adam', metrics=['mse'])
    
    return model


dnn = dnn_model(*get_params(multivar=True))
dnn.summary()
Model: "dnn"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten (Flatten)           (None, 1380)              0         
                                                                 
 dense (Dense)               (None, 512)               707072    
                                                                 
 dropout (Dropout)           (None, 512)               0         
                                                                 
 dense_1 (Dense)             (None, 512)               262656    
                                                                 
 dropout_1 (Dropout)         (None, 512)               0         
                                                                 
 dense_2 (Dense)             (None, 1)                 513       
                                                                 
=================================================================
Total params: 970,241
Trainable params: 970,241
Non-trainable params: 0
_________________________________________________________________
def lstm_model(n_steps, n_horizon, n_features, lr):
    
    tf.keras.backend.clear_session()
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, activation='tanh', input_shape=(n_steps, n_features), return_sequences=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation='tanh', return_sequences=True)),
        tf.keras.layers.LSTM(128, activation='tanh', return_sequences=True),
        tf.keras.layers.LSTM(128, activation='tanh', return_sequences=True),
        tf.keras.layers.LSTM(128, activation='tanh', return_sequences=False),
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dropout(0.05),
       
        tf.keras.layers.Dense(128, activation='tanh'),
        #tf.keras.layers.Dropout(0.01),
        tf.keras.layers.Dense(n_horizon)
    ], name='lstm')
    
    loss = tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    model.compile(loss=loss, optimizer='adam', metrics=['mse'])
    
    return model

lstm = lstm_model(*get_params(multivar=True))
lstm.summary()
Model: "lstm"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 115, 128)          72192     
                                                                 
 bidirectional (Bidirectiona  (None, 115, 256)         263168    
 l)                                                              
                                                                 
 lstm_2 (LSTM)               (None, 115, 128)          197120    
                                                                 
 lstm_3 (LSTM)               (None, 115, 128)          131584    
                                                                 
 lstm_4 (LSTM)               (None, 128)               131584    
                                                                 
 flatten (Flatten)           (None, 128)               0         
                                                                 
 dense (Dense)               (None, 128)               16512     
                                                                 
 dense_1 (Dense)             (None, 1)                 129       
                                                                 
=================================================================
Total params: 812,289
Trainable params: 812,289
Non-trainable params: 0
_________________________________________________________________
def gru_model(n_steps, n_horizon, n_features, lr):
    
    tf.keras.backend.clear_session()
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.GRU(128, activation='tanh', input_shape=(n_steps, n_features), return_sequences=True),
        tf.keras.layers.GRU(128, activation='tanh', return_sequences=True),
        tf.keras.layers.GRU(128, activation='tanh', return_sequences=True),
        tf.keras.layers.GRU(128, activation='tanh', return_sequences=False),
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dropout(0.05),
        
        tf.keras.layers.Dense(128, activation='tanh'),
        #tf.keras.layers.Dropout(0.01),
        tf.keras.layers.Dense(n_horizon)
    ], name='lstm')
    
    loss = tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    model.compile(loss=loss, optimizer='adam', metrics=['mse'])
    
    return model

gru = gru_model(*get_params(multivar=True))
gru.summary()
Model: "lstm"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 gru (GRU)                   (None, 115, 128)          54528     
                                                                 
 gru_1 (GRU)                 (None, 115, 128)          99072     
                                                                 
 gru_2 (GRU)                 (None, 115, 128)          99072     
                                                                 
 gru_3 (GRU)                 (None, 128)               99072     
                                                                 
 flatten (Flatten)           (None, 128)               0         
                                                                 
 dense (Dense)               (None, 128)               16512     
                                                                 
 dense_1 (Dense)             (None, 1)                 129       
                                                                 
=================================================================
Total params: 368,385
Trainable params: 368,385
Non-trainable params: 0
_________________________________________________________________
model_configs=dict()
#run_model("dnn", dnn_model, model_configs, epochs=50)
run_model("lstm", lstm_model, model_configs, epochs=200)
run_model("gru", gru_model, model_configs, epochs=200)
Prediction lookback (n_steps): 115
Prediction horizon (n_horizon): 1
Batch Size: 128
Datasets:
(TensorSpec(shape=(None, None, 12), dtype=tf.float64, name=None), TensorSpec(shape=(None, None, 1), dtype=tf.float64, name=None))
Epoch 1/200
107/107 [==============================] - 11s 61ms/step - loss: 0.0024 - mse: 0.0049 - val_loss: 0.0358 - val_mse: 0.0716
Epoch 2/200
107/107 [==============================] - 6s 51ms/step - loss: 0.0089 - mse: 0.0179 - val_loss: 0.0548 - val_mse: 0.1095
Epoch 3/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0109 - mse: 0.0218 - val_loss: 0.0556 - val_mse: 0.1113
Epoch 4/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0101 - mse: 0.0202 - val_loss: 0.0444 - val_mse: 0.0889
Epoch 5/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0095 - mse: 0.0191 - val_loss: 0.0387 - val_mse: 0.0774
Epoch 6/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0091 - mse: 0.0181 - val_loss: 0.0407 - val_mse: 0.0814
Epoch 7/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0075 - mse: 0.0150 - val_loss: 0.0317 - val_mse: 0.0633
Epoch 8/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0076 - mse: 0.0153 - val_loss: 0.0083 - val_mse: 0.0167
Epoch 9/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0106 - mse: 0.0212 - val_loss: 0.0274 - val_mse: 0.0548
Epoch 10/200
107/107 [==============================] - 6s 51ms/step - loss: 0.0125 - mse: 0.0250 - val_loss: 0.0286 - val_mse: 0.0571
Epoch 11/200
107/107 [==============================] - 6s 51ms/step - loss: 0.0137 - mse: 0.0274 - val_loss: 0.0330 - val_mse: 0.0659
Epoch 12/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0141 - mse: 0.0282 - val_loss: 0.0382 - val_mse: 0.0764
Epoch 13/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0148 - mse: 0.0296 - val_loss: 0.0362 - val_mse: 0.0725
Epoch 14/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0115 - mse: 0.0230 - val_loss: 0.0232 - val_mse: 0.0464
Epoch 15/200
107/107 [==============================] - 6s 51ms/step - loss: 0.0096 - mse: 0.0191 - val_loss: 0.0286 - val_mse: 0.0572
Epoch 16/200
107/107 [==============================] - 6s 53ms/step - loss: 0.0087 - mse: 0.0174 - val_loss: 0.0083 - val_mse: 0.0167
Epoch 17/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0058 - mse: 0.0117 - val_loss: 0.0108 - val_mse: 0.0216
Epoch 18/200
107/107 [==============================] - 6s 53ms/step - loss: 0.0034 - mse: 0.0069 - val_loss: 0.0012 - val_mse: 0.0023
Epoch 19/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0028 - mse: 0.0056 - val_loss: 0.0013 - val_mse: 0.0026
Epoch 20/200
107/107 [==============================] - 6s 51ms/step - loss: 0.0041 - mse: 0.0082 - val_loss: 0.0027 - val_mse: 0.0055
Epoch 21/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0034 - mse: 0.0069 - val_loss: 5.8396e-04 - val_mse: 0.0012
Epoch 22/200
107/107 [==============================] - 6s 54ms/step - loss: 0.0042 - mse: 0.0083 - val_loss: 0.0034 - val_mse: 0.0068
Epoch 23/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0053 - mse: 0.0106 - val_loss: 0.0056 - val_mse: 0.0112
Epoch 24/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0057 - mse: 0.0114 - val_loss: 0.0051 - val_mse: 0.0103
Epoch 25/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0047 - mse: 0.0095 - val_loss: 0.0065 - val_mse: 0.0130
Epoch 26/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0031 - mse: 0.0063 - val_loss: 0.0063 - val_mse: 0.0125
Epoch 27/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0026 - mse: 0.0053 - val_loss: 0.0062 - val_mse: 0.0124
Epoch 28/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0020 - mse: 0.0040 - val_loss: 4.5638e-04 - val_mse: 9.1275e-04
Epoch 29/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0013 - mse: 0.0025 - val_loss: 0.0026 - val_mse: 0.0051
Epoch 30/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0011 - mse: 0.0023 - val_loss: 0.0081 - val_mse: 0.0162
Epoch 31/200
107/107 [==============================] - 6s 53ms/step - loss: 0.0012 - mse: 0.0024 - val_loss: 4.5658e-04 - val_mse: 9.1316e-04
Epoch 32/200
107/107 [==============================] - 6s 51ms/step - loss: 0.0012 - mse: 0.0023 - val_loss: 0.0068 - val_mse: 0.0135
Epoch 33/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0016 - mse: 0.0032 - val_loss: 9.0505e-04 - val_mse: 0.0018
Epoch 34/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0014 - mse: 0.0029 - val_loss: 0.0118 - val_mse: 0.0237
Epoch 35/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0016 - mse: 0.0032 - val_loss: 9.8883e-04 - val_mse: 0.0020
Epoch 36/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0011 - mse: 0.0022 - val_loss: 0.0069 - val_mse: 0.0138
Epoch 37/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0012 - mse: 0.0024 - val_loss: 3.3758e-04 - val_mse: 6.7515e-04
Epoch 38/200
107/107 [==============================] - 6s 51ms/step - loss: 0.0010 - mse: 0.0020 - val_loss: 0.0035 - val_mse: 0.0070
Epoch 39/200
107/107 [==============================] - 6s 53ms/step - loss: 0.0012 - mse: 0.0024 - val_loss: 0.0017 - val_mse: 0.0035
Epoch 40/200
107/107 [==============================] - 6s 52ms/step - loss: 8.9755e-04 - mse: 0.0018 - val_loss: 1.5885e-04 - val_mse: 3.1770e-04
Epoch 41/200
107/107 [==============================] - 5s 51ms/step - loss: 9.6065e-04 - mse: 0.0019 - val_loss: 0.0018 - val_mse: 0.0035
Epoch 42/200
107/107 [==============================] - 5s 50ms/step - loss: 8.0787e-04 - mse: 0.0016 - val_loss: 2.1116e-04 - val_mse: 4.2231e-04
Epoch 43/200
107/107 [==============================] - 6s 52ms/step - loss: 7.5521e-04 - mse: 0.0015 - val_loss: 1.9819e-04 - val_mse: 3.9639e-04
Epoch 44/200
107/107 [==============================] - 6s 52ms/step - loss: 7.5570e-04 - mse: 0.0015 - val_loss: 3.4388e-04 - val_mse: 6.8776e-04
Epoch 45/200
107/107 [==============================] - 6s 52ms/step - loss: 7.6977e-04 - mse: 0.0015 - val_loss: 3.6407e-04 - val_mse: 7.2815e-04
Epoch 46/200
107/107 [==============================] - 5s 51ms/step - loss: 6.4951e-04 - mse: 0.0013 - val_loss: 3.4341e-04 - val_mse: 6.8683e-04
Epoch 47/200
107/107 [==============================] - 6s 53ms/step - loss: 5.2828e-04 - mse: 0.0011 - val_loss: 1.2964e-04 - val_mse: 2.5928e-04
Epoch 48/200
107/107 [==============================] - 6s 52ms/step - loss: 4.7626e-04 - mse: 9.5251e-04 - val_loss: 0.0011 - val_mse: 0.0022
Epoch 49/200
107/107 [==============================] - 6s 53ms/step - loss: 4.8801e-04 - mse: 9.7602e-04 - val_loss: 0.0015 - val_mse: 0.0029
Epoch 50/200
107/107 [==============================] - 6s 52ms/step - loss: 6.6257e-04 - mse: 0.0013 - val_loss: 6.0362e-04 - val_mse: 0.0012
Epoch 51/200
107/107 [==============================] - 6s 52ms/step - loss: 7.5755e-04 - mse: 0.0015 - val_loss: 3.9445e-04 - val_mse: 7.8891e-04
Epoch 52/200
107/107 [==============================] - 6s 52ms/step - loss: 9.2640e-04 - mse: 0.0019 - val_loss: 0.0027 - val_mse: 0.0053
Epoch 53/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0011 - mse: 0.0022 - val_loss: 0.0010 - val_mse: 0.0021
Epoch 54/200
107/107 [==============================] - 6s 53ms/step - loss: 0.0011 - mse: 0.0022 - val_loss: 0.0013 - val_mse: 0.0025
Epoch 55/200
107/107 [==============================] - 6s 52ms/step - loss: 8.7737e-04 - mse: 0.0018 - val_loss: 5.9181e-04 - val_mse: 0.0012
Epoch 56/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0010 - mse: 0.0020 - val_loss: 0.0020 - val_mse: 0.0040
Epoch 57/200
107/107 [==============================] - 6s 52ms/step - loss: 9.6359e-04 - mse: 0.0019 - val_loss: 2.6639e-04 - val_mse: 5.3279e-04
Epoch 58/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0011 - mse: 0.0022 - val_loss: 6.2185e-04 - val_mse: 0.0012
Epoch 59/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0015 - mse: 0.0029 - val_loss: 9.1654e-04 - val_mse: 0.0018
Epoch 60/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0019 - mse: 0.0037 - val_loss: 2.3309e-04 - val_mse: 4.6619e-04
Epoch 61/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0017 - mse: 0.0033 - val_loss: 4.6411e-04 - val_mse: 9.2823e-04
Epoch 62/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0017 - mse: 0.0034 - val_loss: 6.4643e-04 - val_mse: 0.0013
Epoch 63/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0020 - mse: 0.0040 - val_loss: 0.0011 - val_mse: 0.0023
Epoch 64/200
107/107 [==============================] - 6s 53ms/step - loss: 0.0021 - mse: 0.0042 - val_loss: 5.3787e-04 - val_mse: 0.0011
Epoch 65/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0017 - mse: 0.0035 - val_loss: 0.0017 - val_mse: 0.0033
Epoch 66/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0018 - mse: 0.0036 - val_loss: 8.9423e-04 - val_mse: 0.0018
Epoch 67/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0011 - mse: 0.0022 - val_loss: 0.0011 - val_mse: 0.0021
Epoch 68/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0016 - mse: 0.0032 - val_loss: 0.0216 - val_mse: 0.0433
Epoch 69/200
107/107 [==============================] - 6s 53ms/step - loss: 0.0024 - mse: 0.0048 - val_loss: 0.0212 - val_mse: 0.0424
Epoch 70/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0016 - mse: 0.0033 - val_loss: 0.0065 - val_mse: 0.0131
Epoch 71/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0011 - mse: 0.0021 - val_loss: 6.9097e-04 - val_mse: 0.0014
Epoch 72/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0013 - mse: 0.0025 - val_loss: 0.0125 - val_mse: 0.0251
Epoch 73/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0020 - mse: 0.0039 - val_loss: 0.0111 - val_mse: 0.0223
Epoch 74/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0013 - mse: 0.0027 - val_loss: 0.0135 - val_mse: 0.0269
Epoch 75/200
107/107 [==============================] - 6s 51ms/step - loss: 0.0023 - mse: 0.0047 - val_loss: 0.0222 - val_mse: 0.0444
Epoch 76/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0031 - mse: 0.0061 - val_loss: 0.0193 - val_mse: 0.0386
Epoch 77/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0024 - mse: 0.0047 - val_loss: 0.0147 - val_mse: 0.0294
Epoch 78/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0020 - mse: 0.0041 - val_loss: 0.0059 - val_mse: 0.0118
Epoch 79/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0017 - mse: 0.0034 - val_loss: 0.0025 - val_mse: 0.0051
Epoch 80/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0020 - mse: 0.0040 - val_loss: 0.0081 - val_mse: 0.0161
Epoch 81/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0015 - mse: 0.0031 - val_loss: 0.0081 - val_mse: 0.0162
Epoch 82/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0016 - mse: 0.0031 - val_loss: 0.0109 - val_mse: 0.0217
Epoch 83/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0018 - mse: 0.0037 - val_loss: 0.0041 - val_mse: 0.0082
Epoch 84/200
107/107 [==============================] - 6s 52ms/step - loss: 7.1977e-04 - mse: 0.0014 - val_loss: 2.2361e-04 - val_mse: 4.4723e-04
Epoch 85/200
107/107 [==============================] - 5s 51ms/step - loss: 6.6500e-04 - mse: 0.0013 - val_loss: 0.0016 - val_mse: 0.0031
Epoch 86/200
107/107 [==============================] - 6s 53ms/step - loss: 0.0017 - mse: 0.0034 - val_loss: 0.0152 - val_mse: 0.0304
Epoch 87/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0022 - mse: 0.0044 - val_loss: 0.0082 - val_mse: 0.0164
Epoch 88/200
107/107 [==============================] - 6s 51ms/step - loss: 0.0016 - mse: 0.0031 - val_loss: 0.0013 - val_mse: 0.0027
Epoch 89/200
107/107 [==============================] - 5s 51ms/step - loss: 5.5955e-04 - mse: 0.0011 - val_loss: 0.0031 - val_mse: 0.0062
Epoch 90/200
107/107 [==============================] - 6s 52ms/step - loss: 9.8281e-04 - mse: 0.0020 - val_loss: 0.0043 - val_mse: 0.0085
Epoch 91/200
107/107 [==============================] - 6s 53ms/step - loss: 0.0015 - mse: 0.0030 - val_loss: 4.8737e-04 - val_mse: 9.7474e-04
Epoch 92/200
107/107 [==============================] - 6s 51ms/step - loss: 4.9657e-04 - mse: 9.9313e-04 - val_loss: 0.0020 - val_mse: 0.0040
Epoch 93/200
107/107 [==============================] - 6s 52ms/step - loss: 4.8256e-04 - mse: 9.6511e-04 - val_loss: 0.0014 - val_mse: 0.0028
Epoch 94/200
107/107 [==============================] - 5s 51ms/step - loss: 3.4326e-04 - mse: 6.8653e-04 - val_loss: 3.2219e-04 - val_mse: 6.4437e-04
Epoch 95/200
107/107 [==============================] - 6s 52ms/step - loss: 4.8344e-04 - mse: 9.6688e-04 - val_loss: 3.4676e-04 - val_mse: 6.9352e-04
Epoch 96/200
107/107 [==============================] - 5s 51ms/step - loss: 7.1226e-04 - mse: 0.0014 - val_loss: 2.5520e-04 - val_mse: 5.1039e-04
Epoch 97/200
106/107 [============================>.] - ETA: 0s - loss: 5.5416e-04 - mse: 0.0011Restoring model weights from the end of the best epoch: 47.
107/107 [==============================] - 6s 53ms/step - loss: 5.5162e-04 - mse: 0.0011 - val_loss: 0.0011 - val_mse: 0.0022
Epoch 97: early stopping
Prediction lookback (n_steps): 115
Prediction horizon (n_horizon): 1
Batch Size: 128
Datasets:
(TensorSpec(shape=(None, None, 12), dtype=tf.float64, name=None), TensorSpec(shape=(None, None, 1), dtype=tf.float64, name=None))
Epoch 1/200
107/107 [==============================] - 11s 58ms/step - loss: 9.8936e-04 - mse: 0.0020 - val_loss: 0.0111 - val_mse: 0.0222
Epoch 2/200
107/107 [==============================] - 5s 49ms/step - loss: 0.0029 - mse: 0.0058 - val_loss: 0.0497 - val_mse: 0.0993
Epoch 3/200
107/107 [==============================] - 5s 49ms/step - loss: 0.0080 - mse: 0.0160 - val_loss: 0.0613 - val_mse: 0.1226
Epoch 4/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0081 - mse: 0.0162 - val_loss: 0.0555 - val_mse: 0.1110
Epoch 5/200
107/107 [==============================] - 5s 49ms/step - loss: 0.0062 - mse: 0.0124 - val_loss: 0.0238 - val_mse: 0.0476
Epoch 6/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0073 - mse: 0.0145 - val_loss: 0.0553 - val_mse: 0.1106
Epoch 7/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0114 - mse: 0.0227 - val_loss: 0.0539 - val_mse: 0.1078
Epoch 8/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0112 - mse: 0.0223 - val_loss: 0.0633 - val_mse: 0.1266
Epoch 9/200
107/107 [==============================] - 5s 48ms/step - loss: 0.0115 - mse: 0.0230 - val_loss: 0.0487 - val_mse: 0.0975
Epoch 10/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0099 - mse: 0.0198 - val_loss: 0.0483 - val_mse: 0.0966
Epoch 11/200
107/107 [==============================] - 5s 49ms/step - loss: 0.0094 - mse: 0.0187 - val_loss: 0.0324 - val_mse: 0.0649
Epoch 12/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0066 - mse: 0.0132 - val_loss: 0.0131 - val_mse: 0.0263
Epoch 13/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0031 - mse: 0.0062 - val_loss: 7.7634e-04 - val_mse: 0.0016
Epoch 14/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0019 - mse: 0.0038 - val_loss: 4.1290e-04 - val_mse: 8.2581e-04
Epoch 15/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0017 - mse: 0.0034 - val_loss: 0.0012 - val_mse: 0.0024
Epoch 16/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0016 - mse: 0.0033 - val_loss: 0.0053 - val_mse: 0.0107
Epoch 17/200
107/107 [==============================] - 5s 49ms/step - loss: 0.0018 - mse: 0.0035 - val_loss: 0.0045 - val_mse: 0.0091
Epoch 18/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0025 - mse: 0.0050 - val_loss: 0.0024 - val_mse: 0.0047
Epoch 19/200
107/107 [==============================] - 5s 49ms/step - loss: 0.0026 - mse: 0.0051 - val_loss: 0.0030 - val_mse: 0.0061
Epoch 20/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0022 - mse: 0.0044 - val_loss: 0.0113 - val_mse: 0.0227
Epoch 21/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0030 - mse: 0.0060 - val_loss: 0.0014 - val_mse: 0.0029
Epoch 22/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0029 - mse: 0.0059 - val_loss: 0.0142 - val_mse: 0.0284
Epoch 23/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0036 - mse: 0.0072 - val_loss: 0.0036 - val_mse: 0.0073
Epoch 24/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0020 - mse: 0.0039 - val_loss: 0.0016 - val_mse: 0.0032
Epoch 25/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0022 - mse: 0.0045 - val_loss: 0.0030 - val_mse: 0.0061
Epoch 26/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0022 - mse: 0.0043 - val_loss: 0.0047 - val_mse: 0.0095
Epoch 27/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0025 - mse: 0.0050 - val_loss: 0.0115 - val_mse: 0.0231
Epoch 28/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0029 - mse: 0.0058 - val_loss: 0.0047 - val_mse: 0.0094
Epoch 29/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0025 - mse: 0.0050 - val_loss: 0.0045 - val_mse: 0.0089
Epoch 30/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0020 - mse: 0.0039 - val_loss: 0.0012 - val_mse: 0.0024
Epoch 31/200
107/107 [==============================] - 5s 49ms/step - loss: 0.0019 - mse: 0.0039 - val_loss: 0.0024 - val_mse: 0.0048
Epoch 32/200
107/107 [==============================] - 5s 49ms/step - loss: 0.0020 - mse: 0.0040 - val_loss: 0.0012 - val_mse: 0.0025
Epoch 33/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0015 - mse: 0.0029 - val_loss: 0.0011 - val_mse: 0.0021
Epoch 34/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0017 - mse: 0.0034 - val_loss: 0.0068 - val_mse: 0.0136
Epoch 35/200
107/107 [==============================] - 5s 49ms/step - loss: 0.0027 - mse: 0.0054 - val_loss: 0.0039 - val_mse: 0.0078
Epoch 36/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0020 - mse: 0.0040 - val_loss: 0.0022 - val_mse: 0.0044
Epoch 37/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0025 - mse: 0.0050 - val_loss: 0.0020 - val_mse: 0.0039
Epoch 38/200
107/107 [==============================] - 5s 49ms/step - loss: 0.0023 - mse: 0.0046 - val_loss: 0.0010 - val_mse: 0.0021
Epoch 39/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0023 - mse: 0.0046 - val_loss: 0.0010 - val_mse: 0.0021
Epoch 40/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0023 - mse: 0.0047 - val_loss: 5.9897e-04 - val_mse: 0.0012
Epoch 41/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0020 - mse: 0.0040 - val_loss: 0.0013 - val_mse: 0.0025
Epoch 42/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0026 - mse: 0.0052 - val_loss: 0.0017 - val_mse: 0.0033
Epoch 43/200
107/107 [==============================] - 6s 52ms/step - loss: 0.0031 - mse: 0.0063 - val_loss: 5.5586e-04 - val_mse: 0.0011
Epoch 44/200
107/107 [==============================] - 6s 51ms/step - loss: 0.0023 - mse: 0.0047 - val_loss: 6.6154e-04 - val_mse: 0.0013
Epoch 45/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0029 - mse: 0.0059 - val_loss: 9.0038e-04 - val_mse: 0.0018
Epoch 46/200
107/107 [==============================] - 5s 49ms/step - loss: 0.0027 - mse: 0.0053 - val_loss: 7.8888e-04 - val_mse: 0.0016
Epoch 47/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0025 - mse: 0.0051 - val_loss: 9.4456e-04 - val_mse: 0.0019
Epoch 48/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0031 - mse: 0.0061 - val_loss: 0.0014 - val_mse: 0.0028
Epoch 49/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0034 - mse: 0.0068 - val_loss: 0.0017 - val_mse: 0.0033
Epoch 50/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0028 - mse: 0.0057 - val_loss: 0.0048 - val_mse: 0.0096
Epoch 51/200
107/107 [==============================] - 6s 51ms/step - loss: 0.0026 - mse: 0.0052 - val_loss: 0.0034 - val_mse: 0.0068
Epoch 52/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0018 - mse: 0.0036 - val_loss: 0.0042 - val_mse: 0.0083
Epoch 53/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0015 - mse: 0.0031 - val_loss: 0.0050 - val_mse: 0.0101
Epoch 54/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0015 - mse: 0.0031 - val_loss: 0.0095 - val_mse: 0.0190
Epoch 55/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0015 - mse: 0.0031 - val_loss: 0.0023 - val_mse: 0.0046
Epoch 56/200
107/107 [==============================] - 5s 50ms/step - loss: 8.9882e-04 - mse: 0.0018 - val_loss: 0.0023 - val_mse: 0.0047
Epoch 57/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0010 - mse: 0.0021 - val_loss: 0.0021 - val_mse: 0.0042
Epoch 58/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0012 - mse: 0.0023 - val_loss: 0.0017 - val_mse: 0.0033
Epoch 59/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0011 - mse: 0.0022 - val_loss: 0.0011 - val_mse: 0.0022
Epoch 60/200
107/107 [==============================] - 5s 49ms/step - loss: 7.8185e-04 - mse: 0.0016 - val_loss: 0.0011 - val_mse: 0.0021
Epoch 61/200
107/107 [==============================] - 5s 49ms/step - loss: 6.7525e-04 - mse: 0.0014 - val_loss: 0.0029 - val_mse: 0.0058
Epoch 62/200
107/107 [==============================] - 5s 51ms/step - loss: 0.0015 - mse: 0.0029 - val_loss: 0.0091 - val_mse: 0.0183
Epoch 63/200
107/107 [==============================] - 5s 50ms/step - loss: 0.0015 - mse: 0.0030 - val_loss: 0.0109 - val_mse: 0.0217
Epoch 64/200
107/107 [==============================] - ETA: 0s - loss: 7.5374e-04 - mse: 0.0015Restoring model weights from the end of the best epoch: 14.
107/107 [==============================] - 5s 50ms/step - loss: 7.5374e-04 - mse: 0.0015 - val_loss: 7.5409e-04 - val_mse: 0.0015
Epoch 64: early stopping
<PrefetchDataset element_spec=(TensorSpec(shape=(None, None, 12), dtype=tf.float64, name=None), TensorSpec(shape=(None, None, 1), dtype=tf.float64, name=None))>
legend = list()

fig, axs = plt.subplots(1, 2, figsize=(25,5))

def plot_graphs(metric, val, ax, upper):
    ax.plot(val['history'].history[metric])
    ax.plot(val['history'].history[f'val_{metric}'])
    ax.set_title(key)
    ax.legend([metric, f"val_{metric}"])
    ax.set_xlabel('epochs')
    ax.set_ylabel(metric)
    ax.set_ylim([0, upper])
    
for (key, val), ax in zip(model_configs.items(), axs.flatten()):
    plot_graphs('loss', val, ax, 0.2)
print("Loss Curves")
Loss Curves

legend = list()

fig, axs = plt.subplots(1, 2, figsize=(20,5))

def plot_graphs(metric, val, ax, upper):
    ax.plot(val['history'].history[metric])
    ax.plot(val['history'].history[f'val_{metric}'])
    ax.set_title(key)
    ax.legend([metric, f"val_{metric}"])
    ax.set_xlabel('epochs')
    ax.set_ylabel(metric)
    ax.set_ylim([0, upper])
    
for (key, val), ax in zip(model_configs.items(), axs.flatten()):
    plot_graphs('loss', val, ax, 0.2)
print("Loss Curves")
Loss Curves

print("MAE Curves")
fig, axs = plt.subplots(1, 2, figsize=(25,5))
for (key, val), ax in zip(model_configs.items(), axs.flatten()):
    plot_graphs('mse', val, ax, 0.6)
MAE Curves

names = list()
performance = list()

for key, value in model_configs.items():
    names.append(key)
    mae = value['model'].evaluate(value['test_ds'])
    performance.append(mae[1])
    
performance_df = pd.DataFrame(performance, index=names, columns=['mae'])
performance_df['error_mw'] = performance_df['mae'] * df_prep['BTCUSDT_Close'].mean()
print(performance_df)
1/1 [==============================] - 0s 403ms/step - loss: 3.8369e-06 - mse: 7.6738e-06
1/1 [==============================] - 0s 75ms/step - loss: 0.0015 - mse: 0.0030
           mae    error_mw
lstm  0.000008    0.262968
gru   0.003018  103.418190
fig, axs = plt.subplots(2, 1, figsize=(18, 10))
days = 120

vline = np.linspace(0, 23, 24)

for (key, val), ax in zip(model_configs.items(), axs):

    test = val['test_ds']
    preds = val['model'].predict(test)

    xbatch, ybatch = iter(test).get_next()

    ax.plot(ybatch.numpy().reshape(-1))
    ax.plot(preds.reshape(-1))
    ax.set_title(key)
    #ax.vlines(vline, ymin=., ymax=.30, linestyle='dotted', transform = ax.get_xaxis_transform())
    ax.legend(["Actual", "Predicted"])

plt.xlabel("Hours Cumulative")
print('First Two Weeks of Predictions')
1/1 [==============================] - 1s 979ms/step
1/1 [==============================] - 1s 853ms/step
First Two Weeks of Predictions

df_prep.head()
df_prep.to_csv('multivar.csv')
 
 
 
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
Num GPUs Available:  2
 
