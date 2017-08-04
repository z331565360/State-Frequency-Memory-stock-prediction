import build
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
import distutils.util

np.set_printoptions(threshold=np.nan)

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

#Main Run Thread
if __name__=='__main__':
    parser = argparse.ArgumentParser()
	# n-step prediction
    parser.add_argument('-s','--step', type=int, default=1)
	# data path
    parser.add_argument('-d','--data_file', type=str, default='../dataset/data.npy')
	# visualization
    parser.add_argument('-v','--visualization', type=distutils.util.strtobool, default='false')
    args = parser.parse_args()
    step = args.step
	
    global_start_time = time.time()

    print '> Loading data... '
    
    data_file = args.data_file
    X_train, y_train, X_val, y_val, X_test, y_test, gt_test, max_data, min_data = build.load_data(data_file, step)
    test_len = X_test.shape[1]-X_val.shape[1]

    print '> Data Loaded. Compiling...'
    #dimension of hidden states
    if step == 1:
	    hidden_dim = 10
    elif step == 3:
	    hidden_dim = 50
    elif step == 5:
        hidden_dim = 50
    else:
        raise Exception("Don't have the model pretrained with the n-step prediction.")
    #number of frequencies
    freq = 10
    model = build.build_model([1, hidden_dim, 1], freq, 0.01)
	#loading model
    if step == 1:
	    model_path = './snapshot/1d_10_10_5.57_0.000809.hdf5'
    elif step == 3:
	    model_path = './snapshot/3d_50_10_17.00_0.00233.hdf5'
    elif step == 5:
        model_path = './snapshot/5d_50_10_28.90_0.00384.hdf5'
    else:
        raise Exception("Don't have the model pretrained with the n-step prediction.")
	
    model.load_weights(model_path)
    #predition
    print '> Predicting... '
    predicted = model.predict(X_test)
    #denormalization   
    prediction = (predicted[:,:, 0] * (max_data - min_data) + (max_data + min_data))/2

    error = np.sum((prediction[:,-test_len:] - gt_test[:,-test_len:])**2) / (test_len* prediction.shape[0])
    print 'The mean square error is: %f' % error
    
    if args.visualization:
        for ii in range(0,len(prediction)):
            plot_results(prediction[ii, -test_len:], gt_test[ii, -test_len:])