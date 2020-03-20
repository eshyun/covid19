import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import io
import os
import click
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as offline
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters

from selenium import webdriver
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re

from flask import Flask, request, redirect, url_for, render_template

# register datetime converter for a matplotlib plotting method
register_matplotlib_converters()

# if there's Korean font issue
matplotlib.rc('font', family='AppleMyungjo')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = (10, 7)  # (w, h)

app = Flask(__name__, static_url_path='', static_folder='.', template_folder='.')

def get_links(last_date, include=False):
	url = "https://www.cdc.go.kr/menu.es?mid=a20501000000"
	options = webdriver.ChromeOptions()
	options.add_argument('headless')
	driver = webdriver.Chrome('/usr/local/bin/chromedriver', options=options)
	driver.get(url)
	content = driver.page_source
	soup = BeautifulSoup(content, features="lxml")
	soup.findAll('a', text='코로나바이러스감염증-19 국내 발생 현황(3월 19일, 0시 기준)')

	if isinstance(last_date, str):
		last_date = datetime.strptime(last_date, '%Y-%m-%d')
	if not include:
		start = last_date + relativedelta(days=1)

	today = datetime.today()
	today_md = today.strftime('%-m월 %-d일')
	today_ymd = today.strftime('%Y-%m-%d')
	links = {}
	pattern = re.compile(r'.*?\((\d+월 \d+일), (.*?)\)')
	for tag in soup.findAll(lambda tag:tag.name == "a" and "코로나바이러스감염증-19 국내 발생 현황" in tag.text):
		# print(tag.text)
		match = re.search(pattern, tag.text)
		if match:
			date = datetime.strptime(match.group(1), '%m월 %d일') + relativedelta(years=120)
			kind = match.group(2)
			if date >= start:
				if not links.get(date):
					links[date] = tag.get('href')
	driver.close()
	return links

def get_newest_data(last_date, include=False):
	links = get_links(last_date, include=include)
	res = pd.DataFrame()
	for date, link in links.items():
		url = f"https://www.cdc.go.kr{link}"
		dfs = pd.read_html(url)
		df = dfs[0]
		df = df.drop(0, axis=0)
		df = df.drop(0, axis=1)
		cols = [x.replace(' ', '') for x in df.iloc[0]]
		df.columns = cols
		df['날짜'] = date.strftime('%Y-%m-%d')
		df['날짜'] = pd.to_datetime(df['날짜'])
		df = df.set_index('날짜')
		df = df.iloc[-2:-1,]
		df = df.rename(columns={'확진자': '누적 확진자수', '격리해제': '누적 격리해제'})
		df = df[['누적 확진자수', '누적 격리해제']]
		df = df.astype(float)
		res = pd.concat([res, df])
	return res

def get_data(file=None, newest=True):
	if file is None:
		# get data from google drive
		# https://docs.google.com/spreadsheets/d/1fODH5PZJw9jxwV2GRe85BRQgc3mxdyyIpQ0I6MDJKXc/edit#gid=0
		fileid = '1fODH5PZJw9jxwV2GRe85BRQgc3mxdyyIpQ0I6MDJKXc'
		url = f'https://docs.google.com/spreadsheet/ccc?key={fileid}&output=csv'
		r = requests.get(url)
		content = r.content
		data = content.decode('utf-8')
		data.replace('\r\n', '\r')
		df = pd.read_csv(io.StringIO(data))
	else:
		df = pd.read_csv(file)
	df['날짜'] = pd.to_datetime(df['날짜']).dt.date
	df = df.drop_duplicates(subset='날짜', keep='last')
	df = df.set_index('날짜')
	idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1D')
	df = df.reindex(idx)

	df['누적 확진자수'] = df['누적 확진자수'].fillna(method='ffill')
	df['누적 격리해제'] = df['누적 격리해제'].fillna(method='ffill')

	if newest:
		newdf = get_newest_data(df.index.max())
		if len(newdf) > 0:
			df = pd.concat([df, newdf], sort=True)
			df = df.sort_index()
			
	df = df.fillna(0)

	# recaculate 확진자수 증감
	df['확진자수'] = df['누적 확진자수'] - df['누적 확진자수'].shift(1)
	# set first value
	df.loc[df.index.min(),'확진자수'] = 1

	df['격리해제'] = df['누적 격리해제'] - df['누적 격리해제'].shift(1)
	df['격리해제'] = df['격리해제'].fillna(0)

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

def plotly_plot(df, yhat, title, notebook=False):
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

	if notebook:
		fig.show()
	else:
		# offline.plot(fig, include_mathjax='cdn')
		return fig

def plot(df, yhat, title):
	file = 'covid19.png'

	df_hat = pd.DataFrame()
	df_hat = df_hat.reindex(pd.date_range(name='날짜', start=df.index.min(), end=df.index.max() + pd.DateOffset(10)))
	df_hat['yhat'] = yhat

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(df_hat.index, df_hat.yhat, '--', c='blue', ms=20, label='트렌드 예측')
	ax.plot(df.index, df['누적 확진자수'], 'o', c='purple', ms=5, alpha=0.7, label='누적 확진자수')
	ax.legend(loc='best')

	ax.set_xticks(df.index)
	# ax.set_xticks(ax.get_xticks()[::2])

	ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
	ax.xaxis.set_minor_formatter(mdates.DateFormatter("%m/%d"))
	plt.xticks(rotation=30, fontsize=7)

	# make a twin axis
	ax2 = ax.twinx()
	ax2.bar(df.index, df['확진자수'], color='orange', alpha=.5, label='일별 확진자수')
	ax2.bar(df.index, df['격리해제'], color='cyan', alpha=.5, label='일별 격리해제')
	ax2.legend(loc='center right')
	
	plt.title(title)
	plt.savefig(file, dpi=300)
	plt.close()
	return file

def add_data(df, new_cases=None, released_cases=None):
	data = dict()
	if new_cases is not None:
		last_cum_cases = df.loc[df.index.max(), '누적 확진자수']
		data['누적 확진자수'] = last_cum_cases + new_cases
		data['확진자수'] = new_cases
	if released_cases is not None:
		data['격리해제'] = released_cases
	if len(data) > 0:
		df.loc[df.index.max() + pd.DateOffset(1)] = pd.Series(data)

def process_data(method='plotly'):
	# if os.path.exists(file):
	# 	diff = time.time() - os.path.getmtime(file)
	# 	if diff < 60 * 30:  # less than 30 minutes
	# 		return file

	df = get_data()

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
		return plotly_plot(df, yhat, title)
	else:
		return plot(df, yhat, title)

@app.route('/')
def route_root():
	return redirect(url_for('route_plot'))

@app.route('/plot')
def route_plot():
	q = request.args.get('q')
	if q and q == 'png':
		file = process_data(method='png')
		return app.send_static_file(file)
	else:
		fig = process_data(method='plotly')
		div = offline.plot(fig, show_link=False, output_type="div", include_plotlyjs=False)
		return render_template('index.html', chart=div)


if __name__ == '__main__':
	app.run()
