import argparse
import tensorflow as tf
import logging
import numpy as np
import random
import os
import LoadData as DATA

os.environ['CUDA_VISIBLE_DEVICES']='0'




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
    def __init__(self,sjnum):
        self.batch_size=150
        self.lr_init  =0.1
        self.sjnum=sjnum
        self.word_em=5
        self.wordnum=36
        self.bacc=0
        self.num_of_hidden=32
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
        self.target     = tf.placeholder(tf.int32, [self.batch_size, self.word_em], name="target")
        self.label = tf.placeholder(tf.float32, [self.batch_size, self.word_em, self.wordnum],name='label') 
        self.target_len = tf.placeholder(tf.int32, [self.batch_size], name="target_len")
        self.dropout    = tf.placeholder(tf.float32, name="dropout")
        label_1=tf.transpose(self.label,[1,0,2])
# =============================================================================
#         input is emg  402*8=>400*6*3
# =============================================================================
        W_conv1 = self.weight_variable([3, 3, 1, 3])
        
        b_conv1 = self.bias_variable([1])
        h_conv1 = tf.nn.relu(
            tf.nn.conv2d(self.emgl, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1)
# =============================================================================
#         input is emg  402*8=>400*6*3
# =============================================================================
        
        W_conv2 = self.weight_variable([3, 3, 1, 3])
        b_conv2 = self.bias_variable([1])
        h_conv2 = tf.nn.relu(
            tf.nn.conv2d(self.emgr, W_conv2, strides=[1, 1, 1, 1], padding='VALID') + b_conv2)
# =============================================================================
#         input is acc  402*3=>400*1*3
# =============================================================================
        
        W_conv3 = self.weight_variable([3, 3, 1, 3])
        b_conv3 = self.bias_variable([1])
        h_conv3 = tf.nn.relu(
            tf.nn.conv2d(self.accl, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3)
# =============================================================================
#         input is acc  402*3=>400*1*3
# =============================================================================
        
        W_conv4 = self.weight_variable([3, 3, 1, 3])
        b_conv4 = self.bias_variable([1])
        h_conv4 = tf.nn.relu(
            tf.nn.conv2d(self.accr, W_conv4, strides=[1, 1, 1, 1], padding='VALID') + b_conv4)
# =============================================================================
#         input is gyr  402*3=>400*1*3
# =============================================================================
        
        W_conv5 = self.weight_variable([3, 3, 1, 3])
        b_conv5 = self.bias_variable([1])
        h_conv5 = tf.nn.relu(
            tf.nn.conv2d(self.gyrl, W_conv5, strides=[1, 1, 1, 1], padding='VALID') + b_conv5)
# =============================================================================
#         input is gyr  402*3=>400*1*3
# =============================================================================
        
        W_conv6 = self.weight_variable([3, 3, 1, 3])
        b_conv6 = self.bias_variable([1])
        h_conv6 = tf.nn.relu(
            tf.nn.conv2d(self.gyrr, W_conv6, strides=[1, 1, 1, 1], padding='VALID') + b_conv6)
# =============================================================================
#         input is ol  400*3=>400*1*3
# =============================================================================       
        W_conv7 = self.weight_variable([1, 3, 1, 3])
        b_conv7 = self.bias_variable([1])
        h_conv7 = tf.nn.relu(
            tf.nn.conv2d(self.oll, W_conv7, strides=[1, 1, 1, 1], padding='VALID') + b_conv7)
# =============================================================================
#         input is ol  400*3=>400*1*3
# =============================================================================        
        W_conv8 = self.weight_variable([1, 3, 1, 3])
        b_conv8 = self.bias_variable([1])
        h_conv8 = tf.nn.relu(
            tf.nn.conv2d(self.olr, W_conv8, strides=[1, 1, 1, 1], padding='VALID') + b_conv8)        
# =============================================================================
#         input is ori  400*4=>400*1*3
# =============================================================================        
        W_conv9 = self.weight_variable([1, 4, 1, 3])
        b_conv9 = self.bias_variable([1])
        h_conv9 = tf.nn.relu(
            tf.nn.conv2d(self.oril, W_conv9, strides=[1, 1, 1, 1], padding='VALID') + b_conv9)      
# =============================================================================
#         input is ori  400*4=>400*1*3
# =============================================================================        
        W_conv10 = self.weight_variable([1, 4, 1,3])
        b_conv10 = self.bias_variable([1])
        h_conv10 = tf.nn.relu(
            tf.nn.conv2d(self.oril, W_conv10, strides=[1, 1, 1, 1], padding='VALID') + b_conv10)     
        
        multisensor1=tf.concat([h_conv1, h_conv2, h_conv3, h_conv4, h_conv5, h_conv6, h_conv7, h_conv8, h_conv9,
                                h_conv10], 2)
        src_data=tf.concat([self.emgl, self.emgr,self.accl,self.accr,self.gyrl,self.gyrr],2)
        src=tf.slice(src_data,[0,0,0,0],[self.batch_size,400,28,1])
        shape = src.get_shape().as_list()    
        dim = np.prod(shape[2:]) 
        src_trans=tf.transpose(tf.reshape(src, [-1,400,dim]),[1,0,2])
        
# =============================================================================
#         input is 400*20*32=>400*640
# =============================================================================       
        shape = multisensor1.get_shape().as_list()    
        dim = np.prod(shape[2:]) 
        multisensor=tf.transpose(tf.reshape(multisensor1, [-1,400,dim]),[1,0,2])

        W_fc = self.weight_variable([400,60,self.num_of_hidden])
        b_fc = self.bias_variable([self.num_of_hidden])
        h_fc = tf.nn.relu(tf.matmul( multisensor,W_fc) + b_fc)
        
        W_fc1 = self.weight_variable([400,self.num_of_hidden,8])
        b_fc1 = self.bias_variable([8])
        h_fc1 = tf.nn.relu(tf.matmul( h_fc,W_fc1) + b_fc1)
        #output=tf.transpose(h_fc,[1,0,2])

        src_link = tf.concat([h_fc1,src_trans],2)
        
        print(src_link)
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope("decoder"):
            self.W_c = tf.get_variable("W_c", shape=[2*256,256],
                                       initializer=initializer)
            tf.add_to_collection("looss",tf.contrib.layers.l2_regularizer(0.5)(self.W_c ))
            self.b_c = tf.get_variable("b_c", shape=[256],
                                       initializer=initializer) 
            self.proj_W = tf.get_variable("W", shape=[256, self.wordnum],
                                          initializer=initializer)
            tf.add_to_collection("looss",tf.contrib.layers.l2_regularizer(0.5)(self.proj_W ))
            self.proj_b = tf.get_variable("b", shape=[self.wordnum],
                                          initializer=initializer)
            self.proj_Wo = tf.get_variable("Wo", shape=[self.wordnum,self.wordnum],
                                           initializer=initializer)
            tf.add_to_collection("looss",tf.contrib.layers.l2_regularizer(0.5)(self.proj_Wo ))
            self.proj_bo = tf.get_variable("bo", shape=[self.wordnum],
                                           initializer=initializer)
# =============================================================================
#        source and encoder part
# =============================================================================
        with tf.variable_scope("encoder"):
            self.s_proj_W = tf.get_variable("s_proj_W", shape=[self.wordnum, 256],
                                            initializer=initializer)
            self.s_proj_b = tf.get_variable("s_proj_b", shape=[256],
                                            initializer=initializer)
            cell = tf.nn.rnn_cell.BasicLSTMCell(256, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-self.dropout)
            self.encoder = tf.nn.rnn_cell.MultiRNNCell([cell]*2, state_is_tuple=True)
# =============================================================================
#        source and decoder part
# =============================================================================
        with tf.variable_scope("decoder"):
            self.t_proj_W = tf.get_variable("t_proj_W", shape=[self.wordnum, 256],
                                            initializer=initializer)
            tf.add_to_collection("looss",tf.contrib.layers.l2_regularizer(0.5)(self.t_proj_W ))
            self.t_proj_b = tf.get_variable("t_proj_b", shape=[256],
                                            initializer=initializer)                
            cell = tf.nn.rnn_cell.BasicLSTMCell(256, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-self.dropout)
            self.decoder = tf.nn.rnn_cell.MultiRNNCell([cell]*2, state_is_tuple=True)
        
# =============================================================================
#       encoder network
# =============================================================================        
        s = self.encoder.zero_state(self.batch_size, tf.float32)
        encoder_hs = []
        for t in range(400):
            if t > 0: tf.get_variable_scope().reuse_variables()
            x = src_link[t]
            x = tf.matmul(x, self.s_proj_W) + self.s_proj_b
            h, s = self.encoder(x, s)
            encoder_hs.append(h)
        encoder_hs = tf.stack(encoder_hs)
        
        
        s = self.decoder.zero_state(self.batch_size, tf.float32)
        logits = []
        probs  = []
        for t in range(self.word_em):
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
        plogits = []
        pprobs = []
        prob = label_1[0]
        s = self.decoder.zero_state(self.batch_size, tf.float32)
        for t in range(self.word_em):
            if t > 0: tf.get_variable_scope().reuse_variables()
            x = prob
            x = tf.matmul(x, self.t_proj_W) + self.t_proj_b
            h_t, s = self.decoder(x, s)
            h_tld = self.attention(h_t, encoder_hs)
            oemb = tf.matmul(h_tld, self.proj_W) + self.proj_b
            logit = tf.matmul(oemb, self.proj_Wo) + self.proj_bo
            prob = tf.nn.softmax(logit)
            plogits.append(logit)
            pprobs.append(prob)
        with tf.variable_scope("1mmm",reuse=tf.AUTO_REUSE):
            logits     = logits[:-1]
            targets    = tf.transpose(self.target,[1,0])[1:]
            print(tf.stack(targets).get_shape())  
            print(tf.stack(logits).get_shape())    
            weights    = tf.unstack(tf.sequence_mask(self.target_len - 1, self.word_em-1,
                                                    dtype=tf.float32), None, 1)
          
            self.loss  = tf.contrib.seq2seq.sequence_loss(tf.stack(logits), targets, tf.stack(weights))
            self.probs = tf.transpose(tf.stack(probs), [1, 0, 2])
            self.maxa=tf.cast(tf.argmax(self.probs,2), dtype=tf.int32)
            plogits = plogits[:-1]

            self.ploss = tf.contrib.seq2seq.sequence_loss(tf.stack(plogits), targets, tf.stack(weights))
            self.pprobs = tf.transpose(tf.stack(pprobs), [1, 0, 2])
            self.pmaxa = tf.cast(tf.argmax(self.pprobs, 2), dtype=tf.int32)
            self.optim = tf.contrib.layers.optimize_loss(self.loss, None,
                    self.lr_init, "SGD", clip_gradients=5.,
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

    def lcs(self,x,y,lenx,leny):
        a=np.zeros([lenx+1,leny+1])
        for i in range(lenx+1):
            for j in range(leny+1):
                if i==0 or j==0:
                    a[i,j]=0
                else:
                    if x[i-1]==y[j-1]:
                        a[i,j]=a[i-1,j-1]+1
                    else:
                        a[i,j]=max(a[i-1,j],a[i,j-1])
        return a[lenx,leny]/lenx

    def train(self, enl,enr,anl,anr,gnl,gnr,lnl,lnr,onl,onr,y_train,ohtr,tr_len):
        
        for i in range(1000):
            totacc=0
            arr=random.sample(range(self.sjnum),self.batch_size)
            a=[]
            b=[]
            c=[]
            d=[]
            e=[]
            f=[]
            a1=[]
            b1=[]
            c1=[]
            d1=[]
            e1=[]
            f1=[]
            g=[]
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
                f.append(y_train[arr[j]])
                f1.append(ohtr[arr[j]])
                g.append(tr_len[arr[j]])
            output = self.sess.run([self.optim,self.maxa], feed_dict={self.emgl:a,
                                                 self.emgr:a1,
                                                 self.accl:b,
                                                 self.accr:b1,
                                                 self.gyrl:c,
                                                 self.gyrr:c1,
                                                 self.oll:d,
                                                 self.olr:d1,    
                                                 self.oril:e,
                                                 self.orir:e1,
                                                 self.target:f,
                                                 self.label:f1,
                                                 self.target_len :g,
                                                 self.dropout:0.0 })
            for j in range(self.batch_size):
                totacc = totacc + self.lcs(f[j][1:], output[1][j][:self.word_em - 1], g[j] - 1, self.word_em - 1)
            totacc=totacc/self.batch_size
            print('train acc:',totacc)
            totacc=0
            for start,end in zip(range(0,self.sjnum,self.batch_size),range(self.batch_size,self.sjnum+1,self.batch_size)):
                a=[]
                b=[]
                c=[]
                d=[]
                e=[]
                f=[]
                a1=[]
                b1=[]
                c1=[]
                d1=[]
                e1=[]
                f1=[]
                g=[]
                for j in range(start,end):
                    a.append(enl[j])
                    a1.append(enr[j])
                    b.append(anl[j])
                    b1.append(anr[j])
                    c.append(gnl[j])
                    c1.append(gnr[j])
                    d.append(lnl[j])
                    d1.append(lnr[j])
                    e.append(onl[j])
                    e1.append(onr[j])
                    f.append(y_train[j])
                    f1.append(ohtr[j])
                    g.append(tr_len[j])
                output=self.sess.run([self.ploss,self.pmaxa], feed_dict={self.emgl:a,
                                                         self.emgr:a1,
                                                         self.accl:b,
                                                         self.accr:b1,
                                                         self.gyrl:c,
                                                         self.gyrr:c1,
                                                         self.oll:d,
                                                         self.olr:d1,    
                                                         self.oril:e,
                                                         self.orir:e1,
                                                         self.target:f,
                                                         self.label:f1,
                                                         self.target_len :g,
                                                         self.dropout:0.0 })
                
                for j in range(self.batch_size):
                    totacc=totacc+self.lcs(f[j][1:],output[1][j][:self.word_em-1],g[j]-1,self.word_em-1)
                print('loss',output[0])

            totacc=totacc/end
            if totacc>self.bacc:
                self.bacc=totacc
            print('epoch ',i,'\'s acc',totacc)
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
    data=DATA.LoadData("data").getdata()
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
    model = SLR(sjnum)
    model.train(data[0],data[2],data[4],data[6],data[8],data[10],data[12],data[14],data[16],data[18],data[20],data[22],data[23])



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