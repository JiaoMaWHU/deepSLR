import argparse
import tensorflow as tf
import logging
import numpy as np
import random
import os
import LoadData as DATA

os.environ['CUDA_VISIBLE_DEVICES']='1'




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
    parser.add_argument('--lr', type=float, default=0.1,
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
    def __init__(self,sjnum,args):
        self.batch_size=300
        self.lr_init  =args.lr
        self.sjnum=sjnum-1
        self.word_em=4
        self.wordnum=36
        self.bacc=0
        self.num_of_seq=30
        self.num_of_hidden1=256
        self.num_of_hidden2=128
        print(sjnum)
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
        self.emgl = tf.placeholder(tf.float32, [self.batch_size, 402, 8, 1],name='cnn_left_emg')
        self.emgr = tf.placeholder(tf.float32, [self.batch_size, 402, 8,1],name='cnn_right_emg')
        self.accl = tf.placeholder(tf.float32, [self.batch_size, 402, 3, 1],name='cnn_left_acc')
        self.accr = tf.placeholder(tf.float32, [self.batch_size, 402, 3,1],name='cnn_right_acc')
        self.gyrl = tf.placeholder(tf.float32, [self.batch_size, 402, 3, 1],name='cnn_left_gyr')
        self.gyrr = tf.placeholder(tf.float32, [self.batch_size, 402, 3,1],name='cnn_right_gyr')
        self.oll = tf.placeholder(tf.float32, [self.batch_size, 400, 3, 1],name='cnn_left_ol')
        self.olr = tf.placeholder(tf.float32, [self.batch_size, 400, 3,1],name='cnn_right_ol')        
        self.oril = tf.placeholder(tf.float32, [self.batch_size, 400, 4, 1],name='cnn_left_ori')
        self.orir = tf.placeholder(tf.float32, [self.batch_size, 400, 4,1],name='cnn_right_ori')  
        self.cnn_label = tf.placeholder(tf.float32, [self.batch_size,self.num_of_seq], name="cnnlabel")
# =============================================================================
#         input is emg  402*8=>400*6*3
# =============================================================================
        W_conv1 = self.weight_variable([3, 3, 1, 32])
        
        b_conv1 = self.bias_variable([1])
        h_conv1 = tf.nn.relu(
            tf.nn.conv2d(self.emgl, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1)
# =============================================================================
#         input is emg  402*8=>400*6*3
# =============================================================================
        
        W_conv2 = self.weight_variable([3, 3, 1, 32])
        b_conv2 = self.bias_variable([1])
        h_conv2 = tf.nn.relu(
            tf.nn.conv2d(self.emgr, W_conv2, strides=[1, 1, 1, 1], padding='VALID') + b_conv2)
# =============================================================================
#         input is acc  402*3=>400*1*3
# =============================================================================
        
        W_conv3 = self.weight_variable([3, 3, 1, 32])
        b_conv3 = self.bias_variable([1])
        h_conv3 = tf.nn.relu(
            tf.nn.conv2d(self.accl, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3)
# =============================================================================
#         input is acc  402*3=>400*1*3
# =============================================================================
        
        W_conv4 = self.weight_variable([3, 3, 1, 32])
        b_conv4 = self.bias_variable([1])
        h_conv4 = tf.nn.relu(
            tf.nn.conv2d(self.accr, W_conv4, strides=[1, 1, 1, 1], padding='VALID') + b_conv4)
# =============================================================================
#         input is gyr  402*3=>400*1*3
# =============================================================================
        
        W_conv5 = self.weight_variable([3, 3, 1, 32])
        b_conv5 = self.bias_variable([1])
        h_conv5 = tf.nn.relu(
            tf.nn.conv2d(self.gyrl, W_conv5, strides=[1, 1, 1, 1], padding='VALID') + b_conv5)
# =============================================================================
#         input is gyr  402*3=>400*1*3
# =============================================================================
        
        W_conv6 = self.weight_variable([3, 3, 1, 32])
        b_conv6 = self.bias_variable([1])
        h_conv6 = tf.nn.relu(
            tf.nn.conv2d(self.gyrr, W_conv6, strides=[1, 1, 1, 1], padding='VALID') + b_conv6)
# =============================================================================
#         input is ol  400*3=>400*1*3
# =============================================================================       
        W_conv7 = self.weight_variable([1, 3, 1, 32])
        b_conv7 = self.bias_variable([1])
        h_conv7 = tf.nn.relu(
            tf.nn.conv2d(self.oll, W_conv7, strides=[1, 1, 1, 1], padding='VALID') + b_conv7)
# =============================================================================
#         input is ol  400*3=>400*1*3
# =============================================================================        
        W_conv8 = self.weight_variable([1, 3, 1, 32])
        b_conv8 = self.bias_variable([1])
        h_conv8 = tf.nn.relu(
            tf.nn.conv2d(self.olr, W_conv8, strides=[1, 1, 1, 1], padding='VALID') + b_conv8)        
# =============================================================================
#         input is ori  400*4=>400*1*3
# =============================================================================        
        W_conv9 = self.weight_variable([1, 4, 1, 32])
        b_conv9 = self.bias_variable([1])
        h_conv9 = tf.nn.relu(
            tf.nn.conv2d(self.oril, W_conv9, strides=[1, 1, 1, 1], padding='VALID') + b_conv9)      
# =============================================================================
#         input is ori  400*4=>400*1*3
# =============================================================================        
        W_conv10 = self.weight_variable([1, 4, 1,32])
        b_conv10 = self.bias_variable([1])
        h_conv10 = tf.nn.relu(
            tf.nn.conv2d(self.oril, W_conv10, strides=[1, 1, 1, 1], padding='VALID') + b_conv10)     
        
        multisensor1=tf.concat([h_conv1, h_conv2,h_conv3,h_conv4,h_conv5,h_conv6,h_conv7,h_conv8,h_conv9,h_conv10],2)
# =============================================================================
#         input is 400*20*32=>400*640
# =============================================================================       
        shape = multisensor1.get_shape().as_list()    
        dim = np.prod(shape[2:]) 
        multisensor=tf.transpose(tf.reshape(multisensor1, [-1,400,dim]),[1,0,2])

        W_fc = self.weight_variable([400,dim,self.num_of_hidden1])
        b_fc = self.bias_variable([self.num_of_hidden1])
        h_fc = tf.nn.relu(tf.matmul( multisensor,W_fc) + b_fc)
        
        W_fc1 = self.weight_variable([400,self.num_of_hidden1,self.num_of_hidden2])
        b_fc1 = self.bias_variable([self.num_of_hidden2])
        h_fc1 = tf.nn.relu(tf.matmul(  h_fc,W_fc1) + b_fc1)
        
        W_fc2 = self.weight_variable([400,self.num_of_hidden2,self.wordnum])
        b_fc2 = self.bias_variable([self.wordnum])
        h_fc2 = tf.nn.relu(tf.matmul(  h_fc1,W_fc2) + b_fc2)

        '''h_fc2 is the input of lstm(another py project),we will print it to txt when we load the best model for testing
        '''
        h_trans =  tf.transpose(h_fc2,[1,0,2])
        shape = h_trans.get_shape().as_list()
        dim = np.prod(shape[1:]) 
        
        h_fc_flat= tf.reshape(h_trans, [-1,dim])
        print(h_fc_flat.get_shape())
        W_fc3 = self.weight_variable([dim,self.num_of_seq])
        b_fc3 = self.bias_variable([self.num_of_seq])
        logit = tf.matmul(h_fc_flat,W_fc3) + b_fc3
        y=tf.nn.softmax(logit)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=self.cnn_label))
        self.optim = tf.contrib.layers.optimize_loss(self.loss, None,
                    self.lr_init, "SGD", clip_gradients=5.,
                    summaries=["learning_rate", "loss", "gradient_norm"])
        correct_prediction = tf.equal(tf.argmax(y, 1),tf.argmax(self.cnn_label,1))
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        self.sess = self._init_session()
        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess.run(init)


        
    def _init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)


    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)



    def train(self, enl,enr,anl,anr,gnl,gnr,lnl,lnr,onl,onr,y_train,ohtr,tr_len,cnn_label):
        for i in range(1000):
            arr=random.sample(range(self.sjnum),self.batch_size)
            a=[]
            b=[]
            c=[]
            d=[]
            e=[]

            a1=[]
            b1=[]
            c1=[]
            d1=[]
            e1=[]

            h=[]
            for j in range(self.batch_size):
                a.append(enl[arr[j]])
                a1.append(enr[arr[j]])
                b.append(anl[arr[j]])
                b1.append(anr[arr[j]])
                c.append(gnl[arr[j]])
                c1.append(gnr[arr[j]])
                d.append(lnl[arr[j]])
                d1.append(lnr[arr[j]])
                e.append(onl[arr[j]])
                e1.append(onr[arr[j]])
                h.append(cnn_label[arr[j]])
            output=self.sess.run([self.loss,self.acc,self.optim], feed_dict={self.emgl:a,
                                                 self.emgr:a1,
                                                 self.accl:b,
                                                 self.accr:b1,
                                                 self.gyrl:c,
                                                 self.gyrr:c1,
                                                 self.oll:d,
                                                 self.olr:d1,    
                                                 self.oril:e,
                                                 self.orir:e1,
                                                 self.cnn_label:h})
            print('batch ',i,'\'s acc:',output[1])
            if output[1]>0.99:
                break;
            print('loss:',output[0])
            if output[1]>self.bacc:
                self.bacc=output[1]
#==============================================================================
#             for start,end in zip(range(0,self.sjnum,self.batch_size),range(self.batch_size,self.sjnum+1,self.batch_size)):
#                 a=[]
#                 b=[]
#                 c=[]
#                 d=[]
#                 e=[]
#                 a1=[]
#                 b1=[]
#                 c1=[]
#                 d1=[]
#                 e1=[]
#                 h=[]
#                 for j in range(start,end):
#                     a.append(enl[j])
#                     a1.append(enr[j])
#                     b.append(anl[j])
#                     b1.append(anr[j])
#                     c.append(gnl[j])
#                     c1.append(gnr[j])
#                     d.append(lnl[j])
#                     d1.append(lnr[j])
#                     e.append(onl[j])
#                     e1.append(onr[j])
#                     h.append(cnn_label[j])
#                 output=self.sess.run([self.loss,self.acc], feed_dict={self.emgl:a,
#                                                          self.emgr:a1,
#                                                          self.accl:b,
#                                                          self.accr:b1,
#                                                          self.gyrl:c,
#                                                          self.gyrr:c1,
#                                                          self.oll:d,
#                                                          self.olr:d1,    
#                                                          self.oril:e,
#                                                          self.orir:e1,
#                                                          self.cnn_label:h})
#                 
#                 print('acc:',output[1])
#                 print('loss:',output[0])
#                 if output[1]>self.bacc:
#                     self.bacc=output[1]
#==============================================================================
            print('newest bacc:',self.bacc)
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
    load=DATA.LoadData("data")
    data=load.getdata()
    cnn_label=load.getcnn()
    print(len(cnn_label))
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

    sjnum=len(data[20])
    model = SLR(sjnum,args)
    model.train(data[0],data[2],data[4],data[6],data[8],data[10],data[12],data[14],data[16],data[18],data[20],data[22],data[23],cnn_label)



def evaluate(args):
    return 0

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