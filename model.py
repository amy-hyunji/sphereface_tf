# -*- coding: utf-8 -*-
import tensorflow as tf

class Model(object):
    def __init__(self, images, labels, embedding_dim, loss_type):
        self.images = images
        self.labels = labels
        self.embedding_dim = embedding_dim
        self.loss_type = loss_type
        self.embeddings = self.__get_embeddings()
        self.pred_prob, self.loss = self.__get_loss()
        self.predictions = self.__get_pred()
        self.accuracy = self.__get_accuracy()

    def __get_embeddings(self):
        # output after getting through network -> size: embedding dimension
        return self.network(inputs=self.images, embedding_dim = self.embedding_dim)

    def __get_loss(self):
        if self.loss_type==0: return self.Original_Softmax_Loss(self.embeddings, self.labels)
        if self.loss_type==1: return self.Modified_Softmax_Loss(self.embeddings, self.labels)
        if self.loss_type==2: return self.Angular_Softmax_Loss(self.embeddings, self.labels)

    def __get_pred(self):
        # return the highest prediction probability from axis = 1 (each rows)
        return tf.argmax(self.pred_prob, axis=1)

    def __get_accuracy(self):
        # check how many labels it got correct
        correct_predictions = tf.equal(self.predictions, self.labels)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))
        return accuracy

    @staticmethod
    def network(inputs, embedding_dim = 2):
        def prelu(inputs, name=''):
            alpha = tf.get_variable(name, shape=inputs.get_shape(), 
                        initializer=tf.constant_initializer(0.0), dtype=inputs.dtype)
            return tf.maximum(alpha*inputs, inputs)
        def conv(inputs, filters, kernel_size, strides, w_init, padding='same', suffix='', scope=None):
            conv_name = 'conv'+suffix
            relu_name = 'relu'+suffix

            with tf.name_scope(name=scope):
                if w_init == 'xavier': w_init = tf.contrib.layers.xavier_initializer(uniform=True)
                if w_init == 'gaussian': w_init = tf.contrib.layers.xavier_initializer(uniform=False)
                input_shape = inputs.get_shape().as_list()
                net = tf.layers.conv2d(inputs, filters, kernel_size, strides, padding=padding, kernel_initializer=w_init, name=conv_name)
                output_shape = net.get_shape().as_list()
                print("=================================================")
                print("layer: %8s, input shape: %8s, output shape: %8s" %(conv_name, str(input_shape), str(output_shape)))
                print("=================================================")
                net = prelu(net, name=relu_name)
                return net
        def resnet_block(net, blocks, suffix=''):
            n = len(blocks)
            for i in range(n):
                if n == 2 and i == 0: identity = net
                net = conv(inputs = net, filters = blocks[i]['filters'], kernel_size=blocks[i]['kernel_size'], strides=blocks[i]['strides'], w_init=blocks[i]['w_init'], padding=blocks[i]['padding'], suffix=suffix+'_'+blocks[i]['suffix'], scope='conv'+suffix+'_'+blocks[i]['suffix'])
                if n==3 and i ==0: identity = net
            return identity + net

        res1_3 = [
            {'filters':64, 'kernel_size':3, 'strides':2, 'w_init':'xavier', 'padding':'same', 'suffix':'1'},
            {'filters':64, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'2'},
            {'filters':64, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'3'},
        ]
        res2_3 = [
            {'filters':128, 'kernel_size':3, 'strides':2, 'w_init':'xavier', 'padding':'same', 'suffix':'1'},
            {'filters':128, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'2'},   
            {'filters':128, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'3'},
        ]
        res2_5 = [
            {'filters':128, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'4'},
            {'filters':128, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'5'},
        ]
        res3_3 = [
            {'filters':256, 'kernel_size':3, 'strides':2, 'w_init':'xavier', 'padding':'same', 'suffix':'1'},
            {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'2'},   
            {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'3'},
        ]
        res3_5 = [
            {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'4'},
            {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'5'},
        ]
        res3_7 = [
            {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'6'},
            {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'7'},
        ]
        res3_9 = [
            {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'8'},
            {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'9'},
        ]
        res4_1 = [
            {'filters':512, 'kernel_size':3, 'strides':2, 'w_init':'xavier', 'padding':'same', 'suffix':'1'},
            {'filters':512, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'2'},
            {'filters':512, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'3'},
        ]

        net = inputs
        for suffix, blocks in zip(('1', '2', '2', '3', '3', '3', '3', '4'),
                                    (res1_3, res2_3, res2_5, res3_3, res3_5, res3_7, res3_9, res4_1)):
            net = resnet_block(net, blocks, suffix=suffix)
        net = tf.layers.flatten(net)
        embeddings = tf.layers.dense(net, units=embedding_dim, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        return embeddings

    def Original_Softmax_Loss(embeddings, labels):
        '''
        This is the original softmax loss
        '''
        with tf.variable_scope("softmax"):
            weights = tf.get_variable(name='embedding_weights', shape=[embedding.get_shape().as_list()[-1], labels.get_shape().as_list()[-1]], initializer = tf.contrib.layers.xavier_initializer())
            logits = tf.matmul(embeddings, weights)
            pred_prob = tf.nn.softmax(logits=logits)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            return pred_prob, loss

    @staticmethod
    def Modified_Softmax_Loss(embeddings, labels):
        '''
        This loss is slightly different from the original softmax loss. 
        Main difference is that L2-norm of the weights are constrained to 1.
        Then the decision boundary will only depend on the angle between weights and embeddings
        '''
        with tf.variable_scope("softmax"):
            weights = tf.get_variable(name="embedding_weights", shape=[embeddings.get_shape().as_list()[-1], labels.get_shape().as_list()[-1]], initializer=tf.contrib.layers.xavier_initializer())
            weights_norm = tf.norm(weights, axis=0, keepdims=True)
            logits = tf.matmul(embeddings, weights)
            pred_prob = tf.nn.softmax(logits=logits)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            return pred_prob, loss

    @staticmethod
    def Angular_Softmax_Loss(embeddings, labels, margin=4):
        l = 0.
        embeddings_norm = tf.norm(embeddings, axis=1)

        with tf.variable_scope("softmax"):
            weights = tf.get_variable(name='embedding_weights', shape=[embeddings.get_shape().as_list()[-1], labels.get_shape().as_list()[-1]], intializer=tf.contrib.layers.xavier_intializer())
            weights = tf.nn.l2_normalize(weights, axis=0)
            original_logits = tf.matmul(embeddings, weights)
            N = embeddings.get_shape()[0] # get batch_size
            single_sample_label_index = tf.stack([tf.constant(list(range(N)), tf.int64), labels], axis=1)
            selected_logits = tf.gather_nd(original_logits, single_sample_label_index)
            cos_theta = tf.div(selected_logits, embedding_norm)
            cos_theta_power = tf.square(cos_theta)
            cos_theta_biq = tf.pow(cos_theta, 4)
            sign0 = tf.sign(cos_theta)
            sign3 = tf.multiply(tf.sign(2*cos_theta_power-1), sign0)
            sign4 = 2*sign0 + sign3 - 3
            result = sign3*(8*cos_theta_biq-8*cos_theta_power+1) + sign4

            margin_logits = tf.multiply(result, embeddings_norm)
            f = 1.0 / (1.0+l)
            ff = 1.0 - f
            combined_logits = tf.add(original_logits, tf.scatter_nd(single_sample_label_index, tf.subtract(margin_logits, selected_logits), original_logits.get_shape()))
            updated_logits = ff*original_logits + f*combined_logits
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=updated_logits))
            pred_prob = tf.nn.softmax(logits=updated_logits)
            return pred_prob, loss


