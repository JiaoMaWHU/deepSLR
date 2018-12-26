import argparse
import tensorflow as tf
import logging
import numpy as np
import random
import os
import LoadData as DATA
import nlp as nlp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide Warning
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepSLR.")
    parser.add_argument('--dataset', nargs='?', default='chkdata',
                        help='Dataset name.')
    parser.add_argument('--process', nargs='?', default='train',
                        help='Process type: train, evaluate.')
    parser.add_argument('--epoch', type=int, default=1,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=-1,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save to '
                             'pretrain file')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.5,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, '
                             'MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to show the performance of each epoch (0 or 1)')
    parser.add_argument('--batch_norm', type=int, default=1,
                        help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--num_of_hidden', type=int, default=256,
                        help='Number of hidden vector in Encoder (0 or 1)')
    parser.add_argument('--L2_regularization', type=float, default=0.1,
                        help='L2_regularization')
    parser.add_argument('--Dropout_value', type=float, default=0.5,
                        help='Dropout')
    return parser.parse_args()


class SLR():
    def __init__(self, train_data_length, test_data_length, arg):
        self.epoch = arg.epoch
        self.batch_size = arg.batch_size
        self.lr_init = arg.lr
        self.num_of_hidden = arg.num_of_hidden
        self.pretrain_flag = arg.pretrain
        self.optimizer_type = arg.optimizer
        self.sjnum = train_data_length - 10  # ??
        self.sdnum = test_data_length - 10
        self.word_em = 5
        self.wordnum = 36
        self.bacc = 0
        self.save_file = make_save_file(arg)
        self.log_file = make_log_file(arg)
        self.l2_reg = tf.contrib.layers.l2_regularizer(arg.L2_regularization)
        self.Dropout_value = arg.Dropout_value
        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        self.emgl = tf.placeholder(tf.float32, [self.batch_size, 402, 8, 1], name='left_emg')
        self.emgr = tf.placeholder(tf.float32, [self.batch_size, 402, 8, 1], name='right_emg')
        self.accl = tf.placeholder(tf.float32, [self.batch_size, 402, 3, 1], name='left_acc')
        self.accr = tf.placeholder(tf.float32, [self.batch_size, 402, 3, 1], name='right_acc')
        self.gyrl = tf.placeholder(tf.float32, [self.batch_size, 402, 3, 1], name='left_gyr')
        self.gyrr = tf.placeholder(tf.float32, [self.batch_size, 402, 3, 1], name='right_gyr')
        self.oll = tf.placeholder(tf.float32, [self.batch_size, 400, 3, 1], name='left_ol')
        self.olr = tf.placeholder(tf.float32, [self.batch_size, 400, 3, 1], name='right_ol')
        self.oril = tf.placeholder(tf.float32, [self.batch_size, 400, 4, 1], name='left_ori')
        self.orir = tf.placeholder(tf.float32, [self.batch_size, 400, 4, 1], name='right_ori')
        self.target = tf.placeholder(tf.int32, [self.batch_size, self.word_em], name="target")
        self.label = tf.placeholder(tf.float32, [self.batch_size, self.word_em, self.wordnum], name='label')
        self.target_len = tf.placeholder(tf.int32, [self.batch_size], name="target_len")
        self.dropout = tf.placeholder(tf.float32, name="dropout")
        self.lossa = tf.placeholder(tf.float32, [1, 1], name='lossa')
        # ????
        label_1 = tf.transpose(self.label, [1, 0, 2])
        # =============================================================================
        # Attention - Conv - MaxPooling - Emg left - 402*8=>400*6*3
        # =============================================================================
        W_conv1 = self.weight_variable([2, 2, 1, 3], name="W_conv1")
        b_conv1 = self.bias_variable([3])
        h_conv1 = tf.nn.relu(
            tf.nn.conv2d(self.emgl, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1)
        p12 = tf.nn.max_pool(h_conv1, [1, 2, 2, 1], [1, 1, 1, 1], padding='VALID')
        shape = p12.get_shape().as_list()
        dim = np.prod(shape[2:])
        p12 = tf.transpose(tf.reshape(p12, [-1, 400, dim]), [1, 0, 2])  # ->
        # Attention Part
        W_a = self.weight_variable([100, 18], name="W_a")
        W_x = self.weight_variable([100, 18], name="W_x")
        W_t = self.weight_variable([100, 18], name="W_t")
        multiCH = tf.zeros([100, 18])
        p1_t = []
        for i in range(400):
            x = p12[i]
            multiCH = tf.nn.softmax(
                tf.multiply(tf.nn.tanh(tf.multiply(W_a, multiCH) + tf.multiply(W_x, x)), W_t))
            p1_t.append(tf.multiply(x, multiCH))
        p1 = tf.reshape(tf.transpose(tf.stack(p1_t), [1, 0, 2]), [100, 400, 6, 3])

        # =============================================================================
        # Conv - MaxPooling - Emg right - 402*8=>400*6*3
        # =============================================================================
        W_conv2 = self.weight_variable([2, 2, 1, 3], name="W_conv2")
        b_conv2 = self.bias_variable([3])
        h_conv2 = tf.nn.relu(
            tf.nn.conv2d(self.emgr, W_conv2, strides=[1, 1, 1, 1], padding='VALID') + b_conv2)
        p2 = tf.nn.max_pool(h_conv2, [1, 2, 2, 1], [1, 1, 1, 1], padding='VALID')

        # =============================================================================
        # Conv - MaxPooling - Acc left -  402*3=>400*1*1
        # =============================================================================
        W_conv3 = self.weight_variable([2, 2, 1, 3], name="W_conv3")
        b_conv3 = self.bias_variable([3])
        h_conv3 = tf.nn.relu(
            tf.nn.conv2d(self.accl, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3)
        p3 = tf.nn.max_pool(h_conv3, [1, 2, 2, 1], [1, 1, 1, 1], padding='VALID')

        # =============================================================================
        # Conv - MaxPooling - Acc right -  402*3=>400*1*1
        # =============================================================================
        W_conv4 = self.weight_variable([2, 2, 1, 3], name="W_conv4")
        b_conv4 = self.bias_variable([3])
        h_conv4 = tf.nn.relu(
            tf.nn.conv2d(self.accr, W_conv4, strides=[1, 1, 1, 1], padding='VALID') + b_conv4)
        p4 = tf.nn.max_pool(h_conv4, [1, 2, 2, 1], [1, 1, 1, 1], padding='VALID')

        # =============================================================================
        # Conv - MaxPooling - Gyr left -  402*3=>400*1*1
        # =============================================================================
        W_conv5 = self.weight_variable([2, 2, 1, 3], name="W_conv5")
        b_conv5 = self.bias_variable([3])
        h_conv5 = tf.nn.relu(
            tf.nn.conv2d(self.gyrl, W_conv5, strides=[1, 1, 1, 1], padding='VALID') + b_conv5)
        p5 = tf.nn.max_pool(h_conv5, [1, 2, 2, 1], [1, 1, 1, 1], padding='VALID')

        # =============================================================================
        # Conv - MaxPooling - Gyr right -  402*3=>400*1*1
        # =============================================================================
        W_conv6 = self.weight_variable([2, 2, 1, 3], name="W_conv6")
        b_conv6 = self.bias_variable([3])
        h_conv6 = tf.nn.relu(
            tf.nn.conv2d(self.gyrr, W_conv6, strides=[1, 1, 1, 1], padding='VALID') + b_conv6)
        p6 = tf.nn.max_pool(h_conv6, [1, 2, 2, 1], [1, 1, 1, 1], padding='VALID')

        # =============================================================================
        # Conv - Ol left -  400*3=>400*1*1
        # =============================================================================
        W_conv7 = self.weight_variable([1, 3, 1, 1], name="W_conv7")
        b_conv7 = self.bias_variable([1])
        h_conv7 = tf.nn.relu(
            tf.nn.conv2d(self.oll, W_conv7, strides=[1, 1, 1, 1], padding='VALID') + b_conv7)

        # =============================================================================
        # Conv - Ol right -  400*3=>400*1*1
        # =============================================================================
        W_conv8 = self.weight_variable([1, 3, 1, 1], name="W_conv8")
        b_conv8 = self.bias_variable([1])
        h_conv8 = tf.nn.relu(
            tf.nn.conv2d(self.olr, W_conv8, strides=[1, 1, 1, 1], padding='VALID') + b_conv8)

        # =============================================================================
        # Conv - Ori left -  400*3=>400*1*1
        # =============================================================================
        W_conv9 = self.weight_variable([1, 4, 1, 1], name="W_conv9")
        b_conv9 = self.bias_variable([1])
        h_conv9 = tf.nn.relu(
            tf.nn.conv2d(self.oril, W_conv9, strides=[1, 1, 1, 1], padding='VALID') + b_conv9)

        # =============================================================================
        # Conv - Ori right -  400*3=>400*1*1
        # =============================================================================
        W_conv10 = self.weight_variable([1, 4, 1, 1], name="W_conv10")
        b_conv10 = self.bias_variable([1])
        h_conv10 = tf.nn.relu(
            tf.nn.conv2d(self.orir, W_conv10, strides=[1, 1, 1, 1], padding='VALID') + b_conv10)

        # =============================================================================
        # Concat
        # =============================================================================
        data_for_slice1 = tf.concat([self.emgl, self.emgr, self.accl, self.accr, self.gyrl, self.gyrr], 2)
        _data_for_slice1 = tf.slice(data_for_slice1, [0, 0, 0, 0], [self.batch_size, 400, 28, 1])
        origin_data = tf.concat([_data_for_slice1, self.oll, self.olr, self.oril, self.orir], 2);
        _origin_data = tf.reduce_sum(origin_data, 3)

        afterpool_data = tf.concat([p1, p2, p3, p4, p5, p6], 2)
        shape = afterpool_data.get_shape().as_list()
        dim = np.prod(shape[2:])
        _afterpool_data = tf.reshape(afterpool_data, [-1, 400, dim])

        afterconv_data = tf.concat([h_conv7, h_conv8, h_conv9, h_conv10], 2)
        shape = afterconv_data.get_shape().as_list()
        dim = np.prod(shape[2:])
        _afterconv_data = tf.reshape(afterconv_data, [-1, 400, dim])

        final_data = tf.concat([_afterconv_data, _afterpool_data, _origin_data], 2)
        multisensor = tf.transpose(final_data, [1, 0, 2])  # (400, 100, 94)

        # =============================================================================
        #        Source and Encoder part
        # =============================================================================
        with tf.variable_scope("Encoder"):
            x = tf.reshape(multisensor, [-1, 94])
            x = tf.split(x, 400)

            # lstm cell
            lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(128)  # ?????????
            lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(128)

            # dropout
            lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - self.dropout))  # ????????
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
        #        Source and Decoder part
        # =============================================================================
        with tf.variable_scope("Decoder"):
            self.W_c = tf.get_variable("W_c", shape=[2 * self.num_of_hidden, self.num_of_hidden],
                                       regularizer=self.l2_reg,
                                       initializer=tf.contrib.layers.xavier_initializer())
            self.b_c = tf.get_variable("b_c", shape=[self.num_of_hidden], regularizer=self.l2_reg,
                                       initializer=tf.contrib.layers.xavier_initializer())
            self.proj_W = tf.get_variable("proj_W", shape=[self.num_of_hidden, self.wordnum], regularizer=self.l2_reg,
                                          initializer=tf.contrib.layers.xavier_initializer())
            self.proj_b = tf.get_variable("proj_b", shape=[self.wordnum], regularizer=self.l2_reg,
                                          initializer=tf.contrib.layers.xavier_initializer())
            self.proj_Wo = tf.get_variable("proj_Wo", shape=[self.wordnum, self.wordnum], regularizer=self.l2_reg,
                                           initializer=tf.contrib.layers.xavier_initializer())
            self.proj_bo = tf.get_variable("proj_bo", shape=[self.wordnum], regularizer=self.l2_reg,
                                           initializer=tf.contrib.layers.xavier_initializer())
            self.t_proj_W = tf.get_variable("t_proj_W", shape=[self.wordnum, self.num_of_hidden],
                                            regularizer=self.l2_reg,
                                            initializer=tf.contrib.layers.xavier_initializer())
            self.t_proj_b = tf.get_variable("t_proj_b", shape=[self.num_of_hidden], regularizer=self.l2_reg,
                                            initializer=tf.contrib.layers.xavier_initializer())
            cell = tf.nn.rnn_cell.BasicLSTMCell(256, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1 - self.dropout)
            self.decoder = tf.nn.rnn_cell.MultiRNNCell([cell] * 2, state_is_tuple=True)  # ????????

        # =============================================================================
        #       Encoder network
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

        with tf.variable_scope("Run_decoder", reuse=tf.AUTO_REUSE):
            logits = logits[:-1]
            targets = tf.transpose(self.target, [1, 0])[1:]
            weights = tf.unstack(tf.sequence_mask(self.target_len - 1, self.word_em - 1,
                                                  dtype=tf.float32), None, 1)

            self.loss = tf.contrib.seq2seq.sequence_loss(tf.stack(logits), targets, tf.stack(weights))

            self.probs = tf.transpose(tf.stack(probs), [1, 0, 2])
            self.maxa = tf.cast(tf.argmax(self.probs, 2), dtype=tf.int32)
            plogits = plogits[:-1]

            self.ploss = tf.contrib.seq2seq.sequence_loss(tf.stack(plogits), targets, tf.stack(weights))

            self.pprobs = tf.transpose(tf.stack(pprobs), [1, 0, 2], name="pprobs")
            self.pmaxa = tf.cast(tf.argmax(self.pprobs, 2), dtype=tf.int32)
            self.optim = tf.contrib.layers.optimize_loss(self.loss, None,
                                                         self.lr_init, "Adagrad", clip_gradients=5.,
                                                         summaries=["learning_rate", "loss", "gradient_norm"])
        # =============================================================================
        # Init
        # =============================================================================
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
        scores = tf.reduce_sum(tf.multiply(encoder_hs, h_t), 2)
        a_t = tf.nn.softmax(tf.transpose(scores))
        a_t = tf.expand_dims(a_t, 2)
        c_t = tf.matmul(tf.transpose(encoder_hs, perm=[1, 2, 0]), a_t)
        c_t = tf.squeeze(c_t, [2])
        h_tld = tf.tanh(tf.matmul(tf.concat([h_t, c_t], 1), self.W_c) + self.b_c)
        return h_tld

    def weight_variable(self, shape, name):
        return tf.get_variable(name=name, shape=shape, regularizer=self.l2_reg,
                               initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def hb(self, x, lenx):
        a = []
        a.append(x[0])
        for i in range(1, lenx):
            if x[i] != x[i - 1]:
                a.append(x[i])
        return a

    def lcs(self, x, y, lenx, leny):
        a = np.zeros([lenx + 1, leny + 1])
        b = np.zeros([lenx + 1, leny + 1])
        c = np.zeros([lenx + 1, leny + 1])
        for i in range(1, lenx + 1):
            a[i][0] = i
            b[i][0] = i
            c[i][0] = 0
        for j in range(1, leny + 1):
            a[0][j] = j
            b[0][j] = 0
            c[0][j] = j
        for i in range(1, lenx + 1):
            for j in range(1, leny + 1):
                if x[i - 1] == y[j - 1]:
                    a[i, j] = a[i - 1, j - 1]
                    b[i, j] = b[i - 1, j - 1]
                    c[i, j] = c[i - 1, j - 1]
                else:
                    a[i, j] = a[i - 1, j - 1] + 1
                    b[i, j] = b[i - 1, j - 1]
                    c[i, j] = c[i - 1, j - 1]
                z = 0
                if a[i][j - 1] < a[i - 1][j]:
                    z = 1
                if z == 1:
                    if a[i][j] > a[i][j - 1] + 1:
                        a[i][j] = a[i][j - 1] + 1
                        c[i][j] = c[i][j - 1] + 1
                else:
                    if a[i][j] > a[i - 1][j] + 1:
                        a[i][j] = a[i - 1][j] + 1
                        b[i][j] = b[i - 1][j] + 1
        cs = leny
        if leny < lenx:
            cs = lenx
        return 1 - (a[lenx, leny] / cs), (b[lenx, leny] / cs), (c[lenx, leny] / cs)

    def train(self, data1, tdata, cdata):
        # sentence_label, data, data_for_validation
        enl = tdata[0]
        enr = tdata[2]
        anl = tdata[4]
        anr = tdata[6]
        gnl = tdata[8]
        gnr = tdata[10]
        oll = tdata[12]
        olr = tdata[14]
        orl = tdata[16]
        orr = tdata[18]
        y_train = tdata[20]
        y_test = tdata[21]
        ohtr = tdata[22]
        tr_len = tdata[23]
        ohte = tdata[24]
        te_len = tdata[25]
        data = np.argmax(data1, 1)
        src = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
               29, 30}

        for i in range(self.epoch):
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
            totacc = 0
            while num < self.batch_size:
                if data[arr[j]] in src:
                    a.append(enl[arr[j]])
                    a1.append(enr[arr[j]])
                    b.append(anl[arr[j]])
                    b1.append(anr[arr[j]])
                    c.append(gnl[arr[j]])
                    c1.append(gnr[arr[j]])
                    d.append(oll[arr[j]])
                    d1.append(olr[arr[j]])
                    e.append(orl[arr[j]])
                    e1.append(orr[arr[j]])
                    f.append(y_train[arr[j]])
                    f1.append(ohtr[arr[j]])
                    g.append(tr_len[arr[j]])
                    num = num + 1
                j = j + 1

            self.sess.run(self.optim, feed_dict={self.emgl: a,
                                                 self.emgr: a1,
                                                 self.accl: b,
                                                 self.accr: b1,
                                                 self.gyrl: c,
                                                 self.gyrr: c1,
                                                 self.oll: d,
                                                 self.olr: d1,
                                                 self.oril: e,
                                                 self.orir: e1,
                                                 self.target: f,
                                                 self.label: f1,
                                                 self.target_len: g,
                                                 self.dropout: self.Dropout_value})

            totacc = 0
            a = []
            b = []
            c = []
            f = []
            a1 = []
            b1 = []
            c1 = []
            f1 = []
            d = []
            e = []
            d1 = []
            e1 = []
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
                    d.append(oll[j])
                    d1.append(olr[j])
                    e.append(orl[j])
                    e1.append(orr[j])
                    f.append(y_train[j])
                    f1.append(ohtr[j])
                    g.append(tr_len[j])
                    da.append(data[j])
                    num = num + 1
                    znum = znum + 1
                if num == self.batch_size:
                    prob, rloss = self.sess.run([self.pprobs, self.ploss], feed_dict={self.emgl: a,
                                                                                      self.emgr: a1,
                                                                                      self.accl: b,
                                                                                      self.accr: b1,
                                                                                      self.gyrl: c,
                                                                                      self.gyrr: c1,
                                                                                      self.oll: d,
                                                                                      self.olr: d1,
                                                                                      self.oril: e,
                                                                                      self.orir: e1,
                                                                                      self.target: f,
                                                                                      self.label: f1,
                                                                                      self.target_len: g,
                                                                                      self.dropout: self.Dropout_value})

                    for k in range(self.batch_size):
                        nl = nlp.nlp(prob[k][:self.word_em - 1])
                        c = nl.getans()
                        hb_maxa = self.hb(c, len(c))
                        aq, _, _ = self.lcs(f[k][1:], hb_maxa, g[k] - 1, len(hb_maxa))
                        aa[da[k]] = aa[da[k]] + aq
                        bb[da[k]] = bb[da[k]] + 1
                        totacc = totacc + aq
                    num = 0
                    a = []
                    b = []
                    c = []
                    d = []
                    e = []
                    f = []
                    a1 = []
                    b1 = []
                    c1 = []
                    e1 = []
                    d1 = []

                    f1 = []
                    da = []
                    g = []
            print('Epoch %d' % (i + 1))
            logging.info('Epoch %d' % (i + 1))
            for j in range(30):
                print('Sentence %d train acc: %.6f'%(j+1, aa[j] / bb[j]))
                logging.info('Sentence %d train acc: %.6f'%(j+1, aa[j] / bb[j]))
            totacc = totacc / (znum - num)
            print('Overall accuracy %.6f' % totacc)
            logging.info('Overall accuracy %.6f' % totacc)
            totacc = 0
            totir = 0
            totdr = 0
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
                d = []
                e = []
                d1 = []
                e1 = []
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
                    d.append(cdata[6][j])
                    d1.append(cdata[7][j])
                    e.append(cdata[8][j])
                    e1.append(cdata[9][j])
                    g.append(te_len[j])
                prob, rloss = self.sess.run([self.pprobs, self.ploss], feed_dict={self.emgl: a,
                                                                                  self.emgr: a1,
                                                                                  self.accl: b,
                                                                                  self.accr: b1,
                                                                                  self.gyrl: c,
                                                                                  self.gyrr: c1,
                                                                                  self.oll: d,
                                                                                  self.olr: d1,
                                                                                  self.oril: e,
                                                                                  self.orir: e1,
                                                                                  self.target: f,
                                                                                  self.label: f1,
                                                                                  self.target_len: g,
                                                                                  self.dropout: self.Dropout_value})

                for k in range(self.batch_size):
                    nl = nlp.nlp(prob[k][:self.word_em - 1])
                    c = nl.getans()
                    hb_maxa = self.hb(c, len(c))
                    aq, isr, idr = self.lcs(f[k][1:], hb_maxa, g[k] - 1, len(hb_maxa))
                    totacc = totacc + aq
                    totir = totir + isr
                    totdr = totdr + idr
            totacc = totacc / end
            totir = totir / end
            totdr = totdr / end
            print('Test overall accuracy: %.6f, Ir: %.6f, Dr: %.6f' % (totacc, totir, totdr))
            logging.info('Test overall accuracy: %.6f, Ir: %.6f, Dr: %.6f' % (totacc, totir, totdr))
            if totacc > self.bacc:
                self.bacc = totacc
                self.saver.save(self.sess, self.save_file)
            print('Best accuracy: %.6f\n' % self.bacc)
            logging.info('Best accuracy: %.6f\n' % self.bacc)
            writer = tf.summary.FileWriter('./Graphs', tf.get_default_graph())
            writer.close()
        return 0


def make_save_file(args):
    pretrain_path = './Model/%s' % (args.dataset)
    if not os.path.exists(pretrain_path):
        os.makedirs(pretrain_path)
    save_file = pretrain_path + '/%s_model.ckpt' % (args.dataset)
    return save_file


def make_log_file(args):
    pretrain_path = './Model/%s.log' % (args.dataset)
    if not os.path.exists(pretrain_path):
        f = open(pretrain_path, 'w')
        f.close()
    return pretrain_path


def train(args):
    # Data loading
    load = DATA.LoadData(args.dataset)
    data = load.getdata()
    data_for_validation = load.cdata()
    sentence_label = load.getcnn()
    if args.verbose > 0:
        print(
            "args: #batch_size=%d, epoch=%d, batch=%d, lr=%.4f,  optimizer=%s, batch_norm=%d, L2_regularization=%f"
            % (args.batch_size, args.epoch, args.batch_size, args.lr,
               args.optimizer, args.batch_norm, args.L2_regularization))
        logging.info(
            "args: #batch_size=%d, epoch=%d, batch=%d, lr=%.4f,  optimizer=%s, batch_norm=%d, L2_regularization=%f"
            % (args.batch_size, args.epoch, args.batch_size, args.lr,
               args.optimizer, args.batch_norm, args.L2_regularization))
    train_data_length = len(data[20])
    test_data_length = len(data[21])
    model = SLR(train_data_length, test_data_length, args)
    model.train(sentence_label, data, data_for_validation)


def evaluate(args):
    return 0


if __name__ == '__main__':
    args = parse_args()
    log_file = make_log_file(args)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG, filename=log_file)

    if args.process == 'train':
        train(args)
    else:
        evaluate(args)
