import argparse
import tensorflow as tf
import logging
import numpy as np
import random
import os
import LoadData as DATA

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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
    parser.add_argument('--lr', type=float, default=0.3,
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
    def __init__(self, sjnum, sdnum, arg):
        self.batch_size = 100
        self.lr_init = arg.lr
        self.sjnum = sjnum-10
        self.sdnum = sdnum-10
        self.word_em = 5
        self.wordnum = 36
        self.bacc = 0
        self.num_of_hidden = 256
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
        self.emgl = tf.placeholder(tf.float32, [self.batch_size, 402, 8, 1], name='cnn_left_emg')
        self.emgr = tf.placeholder(tf.float32, [self.batch_size, 402, 8, 1], name='cnn_right_emg')
        self.accl = tf.placeholder(tf.float32, [self.batch_size, 402, 3, 1], name='cnn_left_acc')
        self.accr = tf.placeholder(tf.float32, [self.batch_size, 402, 3, 1], name='cnn_right_acc')
        self.gyrl = tf.placeholder(tf.float32, [self.batch_size, 402, 3, 1], name='cnn_left_gyr')
        self.gyrr = tf.placeholder(tf.float32, [self.batch_size, 402, 3, 1], name='cnn_right_gyr')

        self.target = tf.placeholder(tf.int32, [self.batch_size, self.word_em], name="target")
        self.label = tf.placeholder(tf.float32, [self.batch_size, self.word_em, self.wordnum], name='label')
        self.target_len = tf.placeholder(tf.int32, [self.batch_size], name="target_len")
        self.dropout = tf.placeholder(tf.float32, name="dropout")
        self.lossa=tf.placeholder(tf.float32, [1,1], name='lossa')
        label_1 = tf.transpose(self.label, [1, 0, 2])

        multisensor1 = tf.concat([self.emgl, self.emgr, self.accl, self.accr, self.gyrl, self.gyrr], 2)
        # =============================================================================
        #         input is 400*16*32=>400*512
        # =============================================================================
        shape = multisensor1.get_shape().as_list()
        dim = np.prod(shape[2:])
        multisensor = tf.transpose(tf.reshape(multisensor1, [-1, 402, dim]), [1, 0, 2])

        # output=tf.transpose(h_fc,[1,0,2])
        print(multisensor.get_shape())
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope("decoder"):
            self.W_c = tf.get_variable("W_c", shape=[2 * self.num_of_hidden, self.num_of_hidden],
                                       initializer=initializer)
            tf.add_to_collection("looss", tf.contrib.layers.l2_regularizer(0.5)(self.W_c))
            self.b_c = tf.get_variable("b_c", shape=[self.num_of_hidden],
                                       initializer=initializer)
            self.proj_W = tf.get_variable("W", shape=[self.num_of_hidden, self.wordnum],
                                          initializer=initializer)
            tf.add_to_collection("looss", tf.contrib.layers.l2_regularizer(0.5)(self.proj_W))
            self.proj_b = tf.get_variable("b", shape=[self.wordnum],
                                          initializer=initializer)
            self.proj_Wo = tf.get_variable("Wo", shape=[self.wordnum, self.wordnum],
                                           initializer=initializer)
            tf.add_to_collection("looss", tf.contrib.layers.l2_regularizer(0.5)(self.proj_Wo))
            self.proj_bo = tf.get_variable("bo", shape=[self.wordnum],
                                           initializer=initializer)

        # =============================================================================
        #        source and encoder part
        # =============================================================================
        with tf.variable_scope("encoder"):
            x = tf.reshape(multisensor, [-1, 28])  # -1,表示样本数量不固定
            # 拆分成n_step组
            x = tf.split(x, 402)
            # lstm cell
            lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(128)
            lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(128)

            # dropout
            lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - self.dropout))
            lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - self.dropout))

            # forward and backward
            encoder_hs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                lstm_cell_fw,
                lstm_cell_bw,
                inputs=x,
                dtype=tf.float32
            )
            encoder_hs = tf.stack(encoder_hs)
        # =============================================================================
        #        source and decoder part
        # =============================================================================
        with tf.variable_scope("decoder"):
            self.t_proj_W = tf.get_variable("t_proj_W", shape=[self.wordnum, self.num_of_hidden],
                                            initializer=initializer)
            tf.add_to_collection("looss", tf.contrib.layers.l2_regularizer(0.5)(self.t_proj_W))
            self.t_proj_b = tf.get_variable("t_proj_b", shape=[self.num_of_hidden],
                                            initializer=initializer)
            cell = tf.nn.rnn_cell.BasicLSTMCell(256, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1 - self.dropout)
            self.decoder = tf.nn.rnn_cell.MultiRNNCell([cell] * 2, state_is_tuple=True)

        # =============================================================================
        #       encoder network
        # =============================================================================
        s = self.decoder.zero_state(self.batch_size, tf.float32)
        logits = []
        probs = []
        for t in range(self.word_em):
            if t > 0: tf.get_variable_scope().reuse_variables()
            x = label_1[t]
            x = tf.matmul(x, self.t_proj_W) + self.t_proj_b
            h_t, s = self.decoder(x, s)
            h_tld = self.attention(h_t, encoder_hs)
            oemb = tf.matmul(h_tld, self.proj_W) + self.proj_b
            logit = tf.matmul(oemb, self.proj_Wo) + self.proj_bo
            prob = tf.nn.softmax(logit)
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
        with tf.variable_scope("1mmm", reuse=tf.AUTO_REUSE):
            logits = logits[:-1]
            targets = tf.transpose(self.target, [1, 0])[1:]
            weights = tf.unstack(tf.sequence_mask(self.target_len - 1, self.word_em - 1,
                                                  dtype=tf.float32), None, 1)

            self.loss = tf.contrib.seq2seq.sequence_loss(tf.stack(logits), targets, tf.stack(weights))
            self.reloss= tf.reduce_sum(self.lossa)+self.loss
            self.probs = tf.transpose(tf.stack(probs), [1, 0, 2])
            self.maxa = tf.cast(tf.argmax(self.probs, 2), dtype=tf.int32)
            plogits = plogits[:-1]

            self.ploss = tf.contrib.seq2seq.sequence_loss(tf.stack(plogits), targets, tf.stack(weights))
            self.pprobs = tf.transpose(tf.stack(pprobs), [1, 0, 2])
            self.pmaxa = tf.cast(tf.argmax(self.pprobs, 2), dtype=tf.int32)
            self.optim = tf.contrib.layers.optimize_loss(self.reloss, None,
                                                         self.lr_init, "Adagrad", clip_gradients=5.,
                                                         summaries=["learning_rate", "loss", "gradient_norm"])

        # result = tf.while_loop(self.cond, self.body, loop_vars=[self.i+1, temp])
        # time,out=result.stack()
        # print(out.get_shape())

        # Model.

        # _________out _________

        # Compute the square loss.

        # Optimizer.

        # init
        self.sess = self._init_session()
        self.saver = tf.train.Saver(max_to_keep=1)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _init_session(self):
        # adaptively growing video memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def attention(self, h_t, encoder_hs):
        # scores = [tf.matmul(tf.tanh(tf.matmul(tf.concat(1, [h_t, tf.squeeze(h_s, [0])]),
        #                    self.W_a) + self.b_a), self.v_a)
        #          for h_s in tf.split(0, self.max_size, encoder_hs)]
        # scores = tf.squeeze(tf.pack(scores), [2])
        scores = tf.reduce_sum(tf.multiply(encoder_hs, h_t), 2)
        a_t = tf.nn.softmax(tf.transpose(scores))
        a_t = tf.expand_dims(a_t, 2)
        c_t = tf.matmul(tf.transpose(encoder_hs, perm=[1, 2, 0]), a_t)
        c_t = tf.squeeze(c_t, [2])
        h_tld = tf.tanh(tf.matmul(tf.concat([h_t, c_t], 1), self.W_c) + self.b_c)

        return h_tld

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    def hb(self,x,lenx):
        a=[]
        a.append(x[0])
        for i in range(1,lenx):
            if x[i]!=x[i-1]:
                a.append(x[i])
        return a 
    
    def lcs(self,x, y, lenx, leny):
        a = np.zeros([lenx + 1, leny + 1])
        for i in range(lenx + 1):
            a[i][0] = i
        for j in range(leny + 1):
            a[0][j] = j
        for i in range(1, lenx + 1):
            for j in range(1, leny + 1):
                if x[i - 1] == y[j - 1]:
                    a[i, j] = a[i - 1, j - 1]
                else:
                    a[i, j] = a[i - 1, j - 1] + 1
                a[i][j] = min(a[i][j], a[i][j - 1] + 1)
                a[i][j] = min(a[i][j], a[i - 1][j] + 1)
        return 1 - (a[lenx, leny] / leny)

    def train(self, data1, tdata, cdata):
        enl = tdata[0]
        enr = tdata[2]
        anl = tdata[4]
        anr = tdata[6]
        gnl = tdata[8]
        gnr = tdata[10]
        y_train = tdata[20]
        y_test = tdata[21]
        ohtr = tdata[22]
        tr_len = tdata[23]
        ohte = tdata[24]
        te_len = tdata[25]

        data = np.argmax(data1, 1)
        src = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
               29, 30}

        for i in range(2000):
            arr = random.sample(range(self.sjnum), self.sjnum - 1)
            a = []
            b = []
            c = []
            d = []
            e = []
            f = []
            a1 = []
            b1 = []
            c1 = []
            d1 = []
            e1 = []
            f1 = []
            g = []
            j = 0
            num = 0
            totacc=0
            while num < self.batch_size:
                if data[arr[j]] in src:
                    a.append(enl[arr[j]])
                    a1.append(enr[arr[j]])
                    b.append(anl[arr[j]])
                    b1.append(anr[arr[j]])
                    c.append(gnl[arr[j]])
                    c1.append(gnr[arr[j]])
                    f.append(y_train[arr[j]])
                    f1.append(ohtr[arr[j]])
                    g.append(tr_len[arr[j]])
                    num = num + 1
                j = j + 1
            maxa=self.sess.run(self.maxa, feed_dict={self.emgl: a,
                                                 self.emgr: a1,
                                                 self.accl: b,
                                                 self.accr: b1,
                                                 self.gyrl: c,
                                                 self.gyrr: c1,
                                                 self.target: f,
                                                 self.label: f1,
                                                 self.target_len: g,
                                                 self.dropout: 0.5,
                                                 self.lossa:[[1.0]]})
            for k in range(self.batch_size):
                hb_maxa=self.hb( maxa[k][:self.word_em - 1],self.word_em - 1)
                aq = self.lcs(f[k][1:], hb_maxa, g[k] - 1, len(hb_maxa))
                totacc = totacc + aq
            totacc=totacc/self.batch_size
            mineloss=pow((3-totacc),2)
            self.sess.run(self.optim, feed_dict={self.emgl: a,
                                                 self.emgr: a1,
                                                 self.accl: b,
                                                 self.accr: b1,
                                                 self.gyrl: c,
                                                 self.gyrr: c1,
                                                 self.target: f,
                                                 self.label: f1,
                                                 self.target_len: g,
                                                 self.dropout: 0.5,
                                                 self.lossa:[[mineloss]]})
            totacc = 0
            a = []
            b = []
            c = []
            f = []
            a1 = []
            b1 = []
            c1 = []
            f1 = []
            aa = np.zeros(36)
            bb = np.zeros(36)
            g = []
            da = []
            num = 0
            znum = 0
            for j in range(self.sjnum):
                if data[j] in src:
                    a.append(enl[j])
                    a1.append(enr[j])
                    b.append(anl[j])
                    b1.append(anr[j])
                    c.append(gnl[j])
                    c1.append(gnr[j])
                    f.append(y_train[j])
                    f1.append(ohtr[j])
                    g.append(tr_len[j])
                    da.append(data[j])
                    num = num + 1
                    znum = znum + 1
                if num == self.batch_size:
                    maxa, rloss = self.sess.run([self.pmaxa, self.ploss], feed_dict={self.emgl: a,
                                                                                     self.emgr: a1,
                                                                                     self.accl: b,
                                                                                     self.accr: b1,
                                                                                     self.gyrl: c,
                                                                                     self.gyrr: c1,
                                                                                     self.target: f,
                                                                                     self.label: f1,
                                                                                     self.target_len: g,
                                                                                     self.dropout: 0.5,
                                                                                     self.lossa:[[1.0]]})

                    for k in range(self.batch_size):
                        hb_maxa=self.hb( maxa[k][:self.word_em - 1],self.word_em - 1)
                        aq = self.lcs(f[k][1:], hb_maxa, g[k] - 1, len(hb_maxa))
                        aa[da[k]] = aa[da[k]] + aq
                        bb[da[k]] = bb[da[k]] + 1
                        totacc = totacc + aq
                    num = 0
                    a = []
                    b = []
                    c = []
                    f = []
                    a1 = []
                    b1 = []
                    c1 = []
                    f1 = []
                    da = []
                    g = []
            for j in range(36):
                print('seq ', j, '\'sacc:', aa[j] / bb[j], ' ', aa[j], ' ', bb[j])
            totacc = totacc / (znum - num)
            print('epoch ', i, '\'s acc', totacc)
            totacc = 0
            for start, end in zip(range(0, self.sdnum, self.batch_size),
                                  range(self.batch_size, self.sdnum + 1, self.batch_size)):
                a = []
                b = []
                c = []
                f = []
                a1 = []
                b1 = []
                c1 = []
                f1 = []
                g = []
                for j in range(start, end):
                    a.append(cdata[0][j])
                    a1.append(cdata[1][j])
                    b.append(cdata[2][j])
                    b1.append(cdata[3][j])
                    c.append(cdata[4][j])
                    c1.append(cdata[5][j])
                    f.append(y_test[j])
                    f1.append(ohte[j])
                    g.append(te_len[j])
                maxa, rloss = self.sess.run([self.pmaxa, self.ploss], feed_dict={self.emgl: a,
                                                                                 self.emgr: a1,
                                                                                 self.accl: b,
                                                                                 self.accr: b1,
                                                                                 self.gyrl: c,
                                                                                 self.gyrr: c1,
                                                                                 self.target: f,
                                                                                 self.label: f1,
                                                                                 self.target_len: g,
                                                                                 self.dropout: 0.5})

                for k in range(self.batch_size):
                    hb_maxa=self.hb( maxa[k][:self.word_em - 1],self.word_em - 1)
                    aq = self.lcs(f[k][1:], hb_maxa, g[k] - 1, len(hb_maxa))
                    totacc = totacc + aq
            totacc = totacc / end
            print('epoch ', i, ' test\'s acc', totacc)
            if totacc > self.bacc:
                self.bacc = totacc
                self.saver.save(self.sess, "Model_biLSTM/model.ckpt")
            print('newest bacc:', self.bacc)
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
    load = DATA.LoadData("data")
    data = load.getdata()
    cdata = load.cdata()
    cnn_label = load.getcnn()
    if args.verbose > 0:
        # 下面的需要改
        print(
            "FM:   #epoch=%d, batch=%d, lr=%.4f,  optimizer=%s, batch_norm=%d"
            % (args.epoch, args.batch_size, args.lr,
               args.optimizer, args.batch_norm))
        logging.info(
            "FM:   #epoch=%d, batch=%d,so lr=%.4f,  optimizer=%s, batch_norm=%d"
            % (args.epoch, args.batch_size, args.lr,
               args.optimizer, args.batch_norm))

    sjnum = len(data[20])
    sdnum = len(data[21])
    model = SLR(sjnum, sdnum, args)
    model.train(cnn_label, data, cdata)


def evaluate(args):
    return 0


if __name__ == '__main__':

    args = parse_args()

    # log_file = make_log_file(args)
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG, filename=log_file)

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
    # elif args.process == 'evaluate':
    #    evaluate(args)