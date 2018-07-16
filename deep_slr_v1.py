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
    def __init__(self):
        # bind params to class
# =============================================================================
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.save_file = save_file
#         self.pretrain_flag = pretrain_flag
#         self.features_M = features_M
#         self.epoch = epoch
#         self.random_seed = random_seed
#         self.optimizer_type = optimizer_type
#         self.batch_norm = batch_norm
#         self.verbose = verbose
# 
#         # performance of each epoch
#         self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []
# =============================================================================

        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        self.emgl = tf.placeholder(tf.float32, [None, 402, 8, 1],name='cnn_left_emg')
        self.emgr = tf.placeholder(tf.float32, [None, 402, 8,1],name='cnn_right_emg')
        self.accl = tf.placeholder(tf.float32, [None, 402, 3, 1],name='cnn_left_acc')
        self.accr = tf.placeholder(tf.float32, [None, 402, 3,1],name='cnn_right_acc')
        self.gyrl = tf.placeholder(tf.float32, [None, 402, 3, 1],name='cnn_left_gyr')
        self.gyrr = tf.placeholder(tf.float32, [None, 402, 3,1],name='cnn_right_gyr')
        self.oll = tf.placeholder(tf.float32, [None, 400, 3, 1],name='cnn_left_ol')
        self.olr = tf.placeholder(tf.float32, [None, 400, 3,1],name='cnn_right_ol')        
        self.oril = tf.placeholder(tf.float32, [None, 400, 4, 1],name='cnn_left_ori')
        self.orir = tf.placeholder(tf.float32, [None, 400, 4,1],name='cnn_right_ori')  
        self.target     = tf.placeholder(tf.int32, [None, 8], name="target")
        self.label = tf.placeholder(tf.float32, [None, 8, 20],name='label') 
        self.target_len = tf.placeholder(tf.int32, [None], name="target_len")
        label_1=tf.transpose(self.label,[1,0,2])
# =============================================================================
#         input is emg  402*8=>400*6*1
# =============================================================================
        
        W_conv1 = self.weight_variable([3, 3, 1, 1])
        b_conv1 = self.bias_variable([1])
        h_conv1 = tf.nn.relu(
            tf.nn.conv2d(self.emgl, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1)

# =============================================================================
#         input is emg  402*8=>400*6*1
# =============================================================================
        
        W_conv2 = self.weight_variable([3, 3, 1, 1])
        b_conv2 = self.bias_variable([1])
        h_conv2 = tf.nn.relu(
            tf.nn.conv2d(self.emgr, W_conv2, strides=[1, 1, 1, 1], padding='VALID') + b_conv2)
        
# =============================================================================
#         input is acc  402*3=>400*1*1
# =============================================================================
        
        W_conv3 = self.weight_variable([3, 3, 1, 1])
        b_conv3 = self.bias_variable([1])
        h_conv3 = tf.nn.relu(
            tf.nn.conv2d(self.accl, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3)
        
# =============================================================================
#         input is acc  402*3=>400*1*1
# =============================================================================
        
        W_conv4 = self.weight_variable([3, 3, 1, 1])
        b_conv4 = self.bias_variable([1])
        h_conv4 = tf.nn.relu(
            tf.nn.conv2d(self.accr, W_conv4, strides=[1, 1, 1, 1], padding='VALID') + b_conv4)
        
# =============================================================================
#         input is gyr  402*3=>400*1*1
# =============================================================================
        
        W_conv5 = self.weight_variable([3, 3, 1, 1])
        b_conv5 = self.bias_variable([1])
        h_conv5 = tf.nn.relu(
            tf.nn.conv2d(self.gyrl, W_conv5, strides=[1, 1, 1, 1], padding='VALID') + b_conv5)
        
# =============================================================================
#         input is gyr  402*3=>400*1*1
# =============================================================================
        
        W_conv6 = self.weight_variable([3, 3, 1, 1])
        b_conv6 = self.bias_variable([1])
        h_conv6 = tf.nn.relu(
            tf.nn.conv2d(self.gyrr, W_conv6, strides=[1, 1, 1, 1], padding='VALID') + b_conv6)
        

        
        multisensor1=tf.concat([h_conv1, h_conv2,h_conv3,h_conv4,h_conv5,h_conv6,self.oll,self.olr,self.oril,self.orir],2)
        multisensor=tf.transpose(tf.reduce_sum(multisensor1, reduction_indices=[3]),[1,0,2])
# =============================================================================
#         h_flat = tf.contrib.layers.flatten(multisensor1)
#         print(h_flat.get_shape()) 
#         self.multisensor = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)
#         self.multisensor=self.multisensor.unstack(tf.transpose(multisensor1,[1,0,2,3]))
#         #h_flat = tf.contrib.layers.flatten(self.multisensor)
#         print(self.multisensor.stack().get_shape())
#         self.n=tf.constant(400)
#         self.i=tf.constant(0)
#         
#         temp=self.multisensor.read(self.i)
# =============================================================================
        W_fc = self.weight_variable([400,30,10])
        b_fc = self.bias_variable([10])
        h_fc = tf.nn.softmax(tf.matmul( multisensor,W_fc) + b_fc)
        #output=tf.transpose(h_fc,[1,0,2])
        print(h_fc.get_shape())
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope("decoder"):
            self.W_c = tf.get_variable("W_c", shape=[512,256],
                                       initializer=initializer)
            self.b_c = tf.get_variable("b_c", shape=[256],
                                       initializer=initializer) 
            self.proj_W = tf.get_variable("W", shape=[256, 20],
                                          initializer=initializer)
            self.proj_b = tf.get_variable("b", shape=[20],
                                          initializer=initializer)
            self.proj_Wo = tf.get_variable("Wo", shape=[20,5000],
                                           initializer=initializer)
            self.proj_bo = tf.get_variable("bo", shape=[5000],
                                           initializer=initializer)
# =============================================================================
#        source and encoder part
# =============================================================================
        with tf.variable_scope("encoder"):
            self.s_proj_W = tf.get_variable("s_proj_W", shape=[10, 256],
                                            initializer=initializer)
            self.s_proj_b = tf.get_variable("s_proj_b", shape=[256],
                                            initializer=initializer)
            cell = tf.nn.rnn_cell.BasicLSTMCell(256, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.8)
            self.encoder = tf.nn.rnn_cell.MultiRNNCell([cell]*2, state_is_tuple=True)
# =============================================================================
#        source and decoder part
# =============================================================================
        with tf.variable_scope("decoder"):
            self.t_proj_W = tf.get_variable("t_proj_W", shape=[20, 256],
                                            initializer=initializer)
            self.t_proj_b = tf.get_variable("t_proj_b", shape=[256],
                                            initializer=initializer)                
            cell = tf.nn.rnn_cell.BasicLSTMCell(256, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.8)
            self.decoder = tf.nn.rnn_cell.MultiRNNCell([cell]*2, state_is_tuple=True)
        
# =============================================================================
#       encoder network
# =============================================================================        
        s = self.encoder.zero_state(128, tf.float32)
        encoder_hs = []
        for t in range(400):
            if t > 0: tf.get_variable_scope().reuse_variables()
            x = h_fc[t]
            x = tf.matmul(x, self.s_proj_W) + self.s_proj_b
            h, s = self.encoder(x, s)
            encoder_hs.append(h)
        encoder_hs = tf.stack(encoder_hs)
        
        
        s = self.decoder.zero_state(128, tf.float32)
        logits = []
        probs  = []
        for t in range(8):
            if t > 0: tf.get_variable_scope().reuse_variables()
            x = label_1[t]
            x = tf.matmul(x, self.t_proj_W) + self.t_proj_b
            h_t, s = self.decoder(x, s)
            h_tld = self.attention(h_t, encoder_hs)

            oemb  = tf.matmul(h_tld, self.proj_W) + self.proj_b
            logit = tf.matmul(oemb, self.proj_Wo) + self.proj_bo
            prob  = tf.nn.softmax(logit)
            logits.append(logit)
            probs.append(prob)
        logits     = logits[:-1]
        targets    = tf.split(1, 8, self.target)[1:]
        weights    = tf.unstack(tf.sequence_mask(self.target_len - 1, 7,
                                                dtype=tf.float32), None, 1)
      
        self.loss  = tf.contrib.legacy_seq2seq.sequence_loss(logits, targets, weights)
        self.probs = tf.transpose(tf.stack(probs), [1, 0, 2])
        print(self.probs.get_shape())
        self.optim = tf.contrib.layers.optimize_loss(self.loss, None,
                1.0, "SGD", clip_gradients=5.,
                summaries=["learning_rate", "loss", "gradient_norm"])

        #result = tf.while_loop(self.cond, self.body, loop_vars=[self.i+1, temp])
        #time,out=result.stack()
        #print(out.get_shape())
        
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
    def attention(self, h_t, encoder_hs):
        #scores = [tf.matmul(tf.tanh(tf.matmul(tf.concat(1, [h_t, tf.squeeze(h_s, [0])]),
        #                    self.W_a) + self.b_a), self.v_a)
        #          for h_s in tf.split(0, self.max_size, encoder_hs)]
        #scores = tf.squeeze(tf.pack(scores), [2])
        scores = tf.reduce_sum(tf.multiply(encoder_hs, h_t), 2)
        a_t    = tf.nn.softmax(tf.transpose(scores))
        a_t    = tf.expand_dims(a_t, 2)
        c_t    = tf.matmul(tf.transpose(encoder_hs, perm=[1,2,0]), a_t)
        c_t    = tf.squeeze(c_t, [2])
        h_tld  = tf.tanh(tf.matmul(tf.concat([h_t, c_t],1), self.W_c) + self.b_c)

        return h_tld

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

    # restore session;
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