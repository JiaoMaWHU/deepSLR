import argparse
import tensorflow as tf
import logging
import numpy as np
import math
import os
import LoadData as DATA
from time import time
from sklearn.metrics import mean_squared_error
from tensorflow.contrib.layers import batch_norm as batch_norm


def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepSLR.")
    parser.add_argument('--process', nargs='?', default='train',
                        help='Process type: train, evaluate.')
    parser.add_argument('--path', nargs='?', default='../data/',
                        help='Input data path.')
    parser.add_argument('--epoch', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=-1,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save to '
                             'pretrain file')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, '
                             'MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to show the performance of each epoch (0 or 1)')
    parser.add_argument('--batch_norm', type=int, default=1,
                        help='Whether to perform batch normaization (0 or 1)')

    return parser.parse_args()


class SLR():
    def __init__(self, features_M, pretrain_flag, save_file, epoch, batch_size, learning_rate,
                 optimizer_type, batch_norm, verbose, random_seed=2018):
        # bind params to class
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_file = save_file
        self.pretrain_flag = pretrain_flag
        self.features_M = features_M
        self.epoch = epoch
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.verbose = verbose

        # performance of each epoch
        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []

        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        self.emgx = tf.placeholder(tf.float32, [None, 400, 8, 3],name='cnn_X_emg')
        self.imux = tf.placeholder(tf.float32, [None, 400, 13,3],name='cnn_X_imu')
        # Input data.
        W_conv1 = weight_variable([1, 9, 1, ])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(
            tf.nn.conv2d(X_, W_conv1, strides=[1, 1, 2, 1], padding='SAME') + b_conv1)

        # Model.

        # _________out _________

        # Compute the square loss.

        # Optimizer.

        # init
        self.sess = self._init_session()
        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # print number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)
            logging.info("#params: %d" % total_parameters)

    def _init_session(self):
        # adaptively growing video memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)


    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    def train(self, Train_data, Validation_data, Test_data):  # fit a dataset
        return 0
 
def make_save_file(args):
    pretrain_path = '../pretrain/fm_%s_%d' % (args.dataset, args.hidden_factor)
    if not os.path.exists(pretrain_path):
        os.makedirs(pretrain_path)
    save_file = pretrain_path + '/%s_%d' % (args.dataset, args.hidden_factor)
    return save_file

def make_log_file(args):
    pretrain_path = '../pretrain/fm_%s_%d.log' % (args.dataset, args.hidden_factor)
    if not os.path.exists(pretrain_path):
        f = open(pretrain_path, 'w')
        f.close()
    return pretrain_path



def train(args):
    # Data loading
    enl,etl,enr,etr,anl,atl,anr,atr,gnl,gtl,gnr,gtr,lnl,ltl,lnr,ltr,onl,otl,onr,otr,y_train,y_test=DATA.LoadData("data").getdata()
    print(y_train)
    if args.verbose > 0:
        #下面的需要改
        print(
            "FM:   #epoch=%d, batch=%d, lr=%.4f,  optimizer=%s, batch_norm=%d"
            % (  args.epoch, args.batch_size, args.lr, 
               args.optimizer, args.batch_norm))
        logging.info(
            "FM:   #epoch=%d, batch=%d,so lr=%.4f,  optimizer=%s, batch_norm=%d"
            % ( args.epoch, args.batch_size, args.lr, 
               args.optimizer, args.batch_norm))

    # Training
    t1 = time()
    model = SLR()
    model.train(data.Train_data, data.Validation_data, data.Test_data)

    # Find the best validation result across iterations
    best_valid_score = 0
    best_valid_score = min(model.valid_rmse)
    best_epoch = model.valid_rmse.index(best_valid_score)
    print("Best Iter(validation)= %d\t train = %.4f, valid = %.4f [%.1f s]"
          % (best_epoch + 1, model.train_rmse[best_epoch], model.valid_rmse[best_epoch], time() - t1))
    logging.info("Best Iter(validation)= %d\t train = %.4f, valid = %.4f [%.1f s]"
                 % (best_epoch + 1, model.train_rmse[best_epoch], model.valid_rmse[best_epoch], time() - t1))


def evaluate(args):
    # load test data
    data = DATA.LoadData(args.path).Test_data
    save_file = make_save_file(args)

    # load the graph
    weight_saver = tf.train.import_meta_graph(save_file + '.meta')
    pretrain_graph = tf.get_default_graph()

    # load tensors
    feature_embeddings = pretrain_graph.get_tensor_by_name('feature_embeddings:0')
    nonzero_embeddings = pretrain_graph.get_tensor_by_name('nonzero_embeddings:0')
    feature_bias = pretrain_graph.get_tensor_by_name('feature_bias:0')
    bias = pretrain_graph.get_tensor_by_name('bias:0')
    fm = pretrain_graph.get_tensor_by_name('fm:0')
    fm_out = pretrain_graph.get_tensor_by_name('fm_out:0')
    out = pretrain_graph.get_tensor_by_name('out:0')
    train_features = pretrain_graph.get_tensor_by_name('train_features_fm:0')
    train_labels = pretrain_graph.get_tensor_by_name('train_labels_fm:0')
    dropout_keep = pretrain_graph.get_tensor_by_name('dropout_keep_fm:0')
    train_phase = pretrain_graph.get_tensor_by_name('train_phase_fm:0')

    # restore session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    weight_saver.restore(sess, save_file)

    # start evaluation
    num_example = len(data['Y'])
    feed_dict = {train_features: data['X'], train_labels: [[y] for y in data['Y']], dropout_keep: 1.0,
                 train_phase: False}
    ne, fe = sess.run((nonzero_embeddings, feature_embeddings), feed_dict=feed_dict)
    _fm, _fm_out, predictions = sess.run((fm, fm_out, out), feed_dict=feed_dict)

    # calculate rmse
    y_pred = np.reshape(predictions, (num_example,))
    y_true = np.reshape(data['Y'], (num_example,))

    predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
    predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(y_true))  # bound the higher values
    RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))

    print("Test RMSE: %.4f" % (RMSE))
    logging.info("Test RMSE: %.4f" % (RMSE))

if __name__ == '__main__':

    args = parse_args()

    #log_file = make_log_file(args)
    #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG, filename=log_file)

    # initialize the optimal parameters
    # if args.mla:
    #     args.lr = 0.05
    #     args.keep = 0.7
    #     args.batch_norm = 0
    # else:
    #     args.lr = 0.01
    #     args.keep = 0.7
    #     args.batch_norm = 1

    if args.process == 'train':
       train(args)
    #elif args.process == 'evaluate':
    #    evaluate(args)