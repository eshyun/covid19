import os
import time
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import io
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as offline
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
from flask import Flask, request, redirect, url_for

# register datetime converter for a matplotlib plotting method
register_matplotlib_converters()

# if there's Korean font issue
matplotlib.rc('font', family='AppleMyungjo')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = (10, 7) # (w, h)

app = Flask(__name__, static_url_path='', static_folder='.')

def get_data():
	# get data from google drive
	# https://docs.google.com/spreadsheets/d/1fODH5PZJw9jxwV2GRe85BRQgc3mxdyyIpQ0I6MDJKXc/edit#gid=0
	fileid = '1fODH5PZJw9jxwV2GRe85BRQgc3mxdyyIpQ0I6MDJKXc'
	url = f'https://docs.google.com/spreadsheet/ccc?key={fileid}&output=csv'
	r = requests.get(url)
	content = r.content
	data = content.decode('utf-8')
	data.replace('\r\n', '\r')
	df = pd.read_csv(io.StringIO(data))
	df = df.fillna(0)
	return df

def sigmoid(x, k, x0):
	return 1.0 / (1 + np.exp(-k * (x - x0)))

def transform(arr, index, scaler):
	# transform 1D-array of data using scaler
	if isinstance(arr, list):
		arr = np.array(arr)
	arr = arr.flatten()

	data = np.zeros((len(arr), 2))
	data[:,index] = arr
	data = scaler.transform(data)
	return data[:,index]

def inverse_transform(arr, index, scaler):
	# inverse transform 1D-array of data using scaler
	if isinstance(arr, list):
		arr = np.array(arr)
	arr = arr.flatten()
	
	data = np.zeros((len(arr), 2))
	data[:,index] = arr
	data = scaler.inverse_transform(data)
	return data[:,index]

def rsquared(x, y, yhat, scaler):
	# be cautious that R-squared Is Not Valid for Nonlinear Regression
	# https://statisticsbyjim.com/regression/r-squared-invalid-nonlinear-regression/
	scaled_x = transform(x, 0, scaler)
	scaled_y = transform(y, 1, scaler)

	# residual sum of squares
	residuals = y - yhat
	ss_res = np.sum(residuals**2)

	# total sum of squares
	deviation = y - np.mean(y)
	ss_tot = np.sum(deviation**2)

	# r_squared value
	r_squared = 1 - (ss_res / ss_tot)

	return r_squared

def plotly_plot(df, yhat, title):
	df_hat = pd.DataFrame()
	df_hat = df_hat.reindex(pd.date_range(name='날짜', start=df.index.min(), end=df.index.max() + pd.DateOffset(10)))
	df_hat['yhat'] = yhat

	# Create figure with secondary y-axis
	fig = make_subplots(specs=[[{"secondary_y": True}]])

	fig.add_trace(
	    go.Scatter(x=df_hat.index, y=df_hat.yhat, mode='lines', name="트렌드 예측"),
	    secondary_y=False,
	)

	fig.add_trace(
	    go.Scatter(x=df_hat.index, y=df['누적 확진자수'], mode='markers', name="누적 확진자"),
	    secondary_y=False,
	)

	fig.add_trace(
	    go.Bar(x=df.index, y=df['확진자수'], name="신규 확진자", marker={'opacity': 0.6}),
	    secondary_y=True,
	)

	fig.add_trace(
	    go.Bar(x=df.index, y=df['격리해제'], name="격리 해제", marker={'opacity': 0.6}),
	    secondary_y=True,
	)

	# Add figure title
	fig.update_layout(
		title_text=title
	)

	# Set x-axis title
	# fig.update_xaxes(title_text="date")

	# Set y-axes titles
	# fig.update_yaxes(title_text="<b>primary</b> yaxis title", secondary_y=False)
	# fig.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)

	offline.plot(fig, include_mathjax='cdn', filename='covid19.html', auto_open=False)

def plot(df, yhat, title):
	df_hat = pd.DataFrame()
	df_hat = df_hat.reindex(pd.date_range(name='날짜', start=df.index.min(), end=df.index.max() + pd.DateOffset(10)))
	df_hat['yhat'] = yhat

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(df_hat.index, df_hat.yhat, '--', c='blue', ms=20, label='predicted')
	ax.plot(df.index, df['누적 확진자수'], 'o', c='purple', ms=5, alpha=0.7, label='cumulative cases')
	ax.legend(loc='best')

	ax.set_xticks(df.index)
	# ax.set_xticks(ax.get_xticks()[::2])

	ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
	ax.xaxis.set_minor_formatter(mdates.DateFormatter("%m/%d"))
	plt.xticks(rotation=30, fontsize=7)

	# make a twin axis
	ax2 = ax.twinx()
	ax2.bar(df.index, df['확진자수'], color='orange', alpha=.5, label='new cases')
	ax2.bar(df.index, df['격리해제'], color='cyan', alpha=.5, label='released cases')
	ax2.legend(loc='center right')
	
	plt.title(title)
	plt.savefig('covid19.png', dpi=300)
	plt.close()

def process_data(method='plotly'):
	if method == 'plotly':
		file = 'covid19.html'
	else:
		file = 'covid19.png'
	if os.path.exists(file):
		diff = time.time() - os.path.getmtime(file)
		if diff < 60 * 30:  # less than 30 minutes
			return file

	df = get_data()
	df['날짜'] = pd.to_datetime(df['날짜']).dt.date
	df = df.drop_duplicates(subset='날짜', keep='last')
	df = df.set_index('날짜')
	idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1D')
	df = df.reindex(idx)

	# fill 누적 확진자수
	df['누적 확진자수'] = df['누적 확진자수'].fillna(method='ffill')
	# recaculate 확진자수 증감
	df['확진자수'] = df['누적 확진자수'] - df['누적 확진자수'].shift(1)
	# set first value 
	df.loc[df.index.min(),'확진자수'] = 1

	df['격리해제'] = df['누적 격리해제'] - df['누적 격리해제'].shift(1)
	df['격리해제'] = df['격리해제'].fillna(0)

	y = np.array(df['누적 확진자수'].values)
	x = np.arange(1, len(y)+1)

	scaler = MinMaxScaler()
	data = np.array(list(zip(x,y)))
	scaled = scaler.fit_transform(data)

	popt, pcov = curve_fit(sigmoid, scaled[:,0], scaled[:,1])
	estimated_k, estimated_x0 = popt

	xhat = np.arange(1, len(x) + 10 + 1)
	scaled_xhat = transform(xhat, 0, scaler)
	scaled_yhat = sigmoid(scaled_xhat, k=estimated_k, x0=estimated_x0)  # = sigmoid(scaled_xhat, *popt)
	yhat = inverse_transform(scaled_yhat, 1, scaler)
	title = r"$f(x)=\frac{1}{1 + e^{%.2f(x - %.2f)}}, R^2=%.3f$" % (-estimated_k, estimated_x0, rsquared(x,y,yhat[:len(y)],scaler))

	if method == 'plotly':
		plotly_plot(df, yhat, title)
	else:
		plot(df, yhat, title)
	return file

@app.route('/')
def route_root():
	return redirect(url_for('route_plot'))

@app.route('/plot')
def route_plot():
	q = request.args.get('q')
	if q and q == 'png':
		file = process_data(method='png')
	else:
		file = process_data(method='plotly')
	return app.send_static_file(file)

if __name__ == '__main__':
	app.run()
