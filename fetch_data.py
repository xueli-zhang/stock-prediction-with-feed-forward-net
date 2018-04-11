import numpy as np
import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt 



FOLDER = 'data'


def load_data(data_usage = "train", folder = FOLDER, show_plot = True):


	if not os.path.exists(folder):
		print('Fail to load dataset, make sure path is correct...')

		return

	raw_data = pd.read_csv(folder+'\s&p500.csv')

	raw_data = raw_data.drop(['Date', 'Adj Close'], axis = 1)

	close_data = raw_data['Close']
	open_data = raw_data['Open']


	close_data = np.array(close_data)[:7021]
	open_data = np.array(open_data)[:7021]

	print("close_data: ")
	print(len(close_data))
	print("open_data: ")
	print(len(open_data))


	if data_usage == "train":

		close_data_train = close_data[:6010]
		open_data_train = open_data[:6010]

		close_t_m_10 = close_data_train[:-10]
		close_t_m_9 = close_data_train[1:-9]
		close_t_m_8 = close_data_train[2:-8]
		close_t_m_7 = close_data_train[3:-7]
		close_t_m_6 = close_data_train[4:-6]
		close_t_m_5 = close_data_train[5:-5]
		close_t_m_4 = close_data_train[6:-4]
		close_t_m_3 = close_data_train[7:-3]
		close_t_m_2 = close_data_train[8:-2]
		close_t_m_1 = close_data_train[9:-1]
		close_t = close_data_train[10:]

		open_t_m_10 = open_data_train[:-10]
		open_t_m_9 = open_data_train[1:-9]
		open_t_m_8 = open_data_train[2:-8]
		open_t_m_7 = open_data_train[3:-7]
		open_t_m_6 = open_data_train[4:-6]
		open_t_m_5 = open_data_train[5:-5]
		open_t_m_4 = open_data_train[6:-4]
		open_t_m_3 = open_data_train[7:-3]
		open_t_m_2 = open_data_train[8:-2]
		open_t_m_1 = open_data_train[9:-1]
		open_t = open_data_train[10:]

		train_data = np.asmatrix([close_t_m_1, close_t_m_3, open_t_m_1, open_t_m_3]).reshape(6000, 4)
		print("train data: ")
		print(train_data.shape)
		print(train_data)

		train_label = np.asmatrix([open_t]).reshape(6000, 1)
		print("train label: ")
		print(train_label.shape)
		print(train_label)

		if show_plot == True:
			correl_matrix = np.corrcoef([close_t_m_10,close_t_m_9,close_t_m_8,close_t_m_7,close_t_m_6,close_t_m_5,
				close_t_m_4,close_t_m_3,close_t_m_2,close_t_m_1,close_t,open_t_m_10,open_t_m_9,open_t_m_8,open_t_m_7,
				open_t_m_6,open_t_m_5,open_t_m_4,open_t_m_3,open_t_m_2,open_t_m_1,open_t])


			print("correlated matrix:")
			print(correl_matrix)

			plt.figure('Cross Correlated Matrix:', figsize = (10,8))
			sns.set(style='white')

			ax = plt.axes()

			sns.heatmap(correl_matrix, ax = ax, yticklabels = ['close(t-10)','close(t-9)','close(t-8)','close(t-7)',
				'close(t-6)','close(t-5)','close(t-4)','close(t-3)','close(t-2)','close(t-1)','close(t)','open(t-10)'
				,'open(t-9)','open(t-8)','open(t-7)','open(t-6)','open(t-5)','open(t-4)','open(t-3)','open(t-2)'
				,'open(t-1)','open(t)'], xticklabels = ['close(t-10)','close(t-9)','close(t-8)','close(t-7)',
				'close(t-6)','close(t-5)','close(t-4)','close(t-3','close(t-2)','close(t-1)','close(t)','open(t-10)'
				,'open(t-9)','open(t-8)','open(t-7)','open(t-6)','open(t-5)','open(t-4)','open(t-3)','open(t-2)'
				,'open(t-1)','open(t)'])

			ax.set_title('Cross Correlated Matrix\nRelationship between Inputs(Close price, prev Open price)and Prediction(Open price)')

			print("From the cross correlated matrix figure:\nClose and open data can be used to predict open data after 3 days and tomorrow.")

			plt.show()

			#grab close(t-1), close(t-3), open(t-3), open(t-1) as input data

		return train_data, train_label

	if data_usage == "validation":

		close_data_valid = close_data[6011:]
		open_data_valid = open_data[6011:]


		close_v_t_m_10 = close_data_valid[:-10]
		close_v_t_m_9 = close_data_valid[1:-9]
		close_v_t_m_8 = close_data_valid[2:-8]
		close_v_t_m_7 = close_data_valid[3:-7]
		close_v_t_m_6 = close_data_valid[4:-6]
		close_v_t_m_5 = close_data_valid[5:-5]
		close_v_t_m_4 = close_data_valid[6:-4]
		close_v_t_m_3 = close_data_valid[7:-3]
		close_v_t_m_2 = close_data_valid[8:-2]
		close_v_t_m_1 = close_data_valid[9:-1]
		close_v_t = close_data_valid[10:]

		open_v_t_m_10 = open_data_valid[:-10]
		open_v_t_m_9 = open_data_valid[1:-9]
		open_v_t_m_8 = open_data_valid[2:-8]
		open_v_t_m_7 = open_data_valid[3:-7]
		open_v_t_m_6 = open_data_valid[4:-6]
		open_v_t_m_5 = open_data_valid[5:-5]
		open_v_t_m_4 = open_data_valid[6:-4]
		open_v_t_m_3 = open_data_valid[7:-3]
		open_v_t_m_2 = open_data_valid[8:-2]
		open_v_t_m_1 = open_data_valid[9:-1]
		open_v_t = open_data_valid[10:]


		valid_data = np.asmatrix([close_v_t_m_1, close_v_t_m_3, open_v_t_m_1, open_v_t_m_3]).reshape(1000, 4)

		valid_label = np.asmatrix([open_v_t]).reshape(1000, 1)

		return valid_data, valid_label

def main():
	load_data("validation")


if __name__ == '__main__':
	main()