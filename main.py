import tensorflow as tf
import numpy as np
import model
from tqdm import tqdm
import os,sys,cv2,random,datetime
import dataloader

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 256, "define batch size")
flags.DEFINE_integer("image_size", 256, "define image size")
flags.DEFINE_integer("epoch_num", 40, "define number of epoch")
flags.DEFINE_integer("train_batchs", 40, "define num of train batch")
flags.DEFINE_integer("test_batchs", 20, "define num of test batch")
flags.DEFINE_integer("embedding_dim", 2, "define num of embedding dimension")
flags.DEFINE_integer("loss_type", 1, "define the loss type")
flags.DEFINE_string("data_path", "../Dataset/CASIA-WebFace", "path of the dataset")

loss_type = FLAGS.loss_type

def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')

def train(loss_type):
    
    '''
    original_softmax_loss_network = network(0)
    modified_softmax_loss_network = network(1)
    angular_softmax_loss_network = network(2)
    '''
    
    # define parameters
    bs = FLAGS.batch_size
    imgsize = FLAGS.image_size
    epochs = FLAGS.epoch_num
    train_batchs = FLAGS.train_batchs
    test_batchs = FLAGS.test_batchs
    embedding_dim = FLAGS.embedding_dim
    lr = 0.001
    num_classes = os.listdir(FLAGS.data_path)

    # define input placeholders for network
    images = tf.placeholder(tf.float32, shape = [bs, imgsize, imgsize, 3], name='input')
    labels = tf.placeholder(tf.int64, [bs, ])
    global_step = tf.Variable(0, trainable=False)
    add_step_op = tf.assign_add(global_step, tf.constant(1))

    # define network
    network = model.Model(images, labels, embedding_dim, loss_type)
    accuracy = network.accuracy
    loss = network.loss

    tf.summary.scalar(name="Loss", tensor=loss)
    tf.summary.scalar(name="Accuracy", tensor=accuracy)
    tf.summary.image(name="Input", tensor=images)
    summary = tf.summary.merge_all()

    # define optimizer and lr
    decay_lr = tf.train.exponential_decay(lr, global_step, 500, 0.9)
    optimizer = tf.train.AdamOptimizer(decay_lr)
    train_op = optimizer.minimize(loss)

    sess = tf.Session()
    network.loss_type = loss_type
    print("Currently using loss type is {}".format(loss_type))

    # tensorboard
    saver = tf.train.Saver()
    dir = os.path.join('./tensorboard', 'checkpoints')
    if not os.path.exists(dir): os.makedirs(dir)
    train_writer = tf.summary.FileWriter(logdir=dir+'/Train', graph=sess.graph)
    test_writer = tf.summary.FileWriter(logdir=dir+'/Test')

    sess.run(tf.global_variables_initializer())
    
    # TRAINing process
    for epoch in range(epochs):
        nlabels = np.zeros((train_batchs*bs, ), dtype = np.int32)
        embeddings = np.zeros((train_batchs*bs, embedding_dim), dtype=np.float32)
        train_acc = 0.
        for batch in tqdm(range(train_batchs)):
            i, j = batch*bs, (batch+1)*bs
            batch_images, batch_labels = dataloader.get_next_batch(bs, imgsize)
            #batch_labels = tf.one_hot(np.reshape(batch_labels, [-1]), num_classes)
            feed_dict = {images: batch_images, labels: batch_labels}
            _, s, batch_loss, batch_acc, embeddings[i:j, :] = sess.run([train_op, summary,  loss, accuracy, network.embeddings], feed_dict)
            nlabels[i:j] = batch_labels
            f.write(" ".join(map(str,[batch_acc, batch_loss]))+"\n")
            train_acc += batch_acc
        
        train_writer.add_summary(s, epoch)
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoint')
        saver.save(sess, save_path=os.path.join('./checkpoints', str(epoch)))
        
        train_acc /= train_batchs
        print("epoch %2d ------------- train accuracy: %.4f"%(epoch+1, train_acc))
        # visualize(embeddings, nlabels, epoch, train_acc, picname="./image/%d/%d.jpg"%(loss_type, epoch))

    # TESTing process


if __name__ == "__main__":
    #gif = ['original_softmax_loss.gif', 'modified_softmax_loss.gif', 'angular_softmax_loss.gif']
    #path = './image/%d'%loss_type
    #gif_name = './image/%s'%gif[loss_type]
    train(loss_type=loss_type)
    #create_gif(gif_name, path)
