from pathlib import Path

import numpy as np
import tensorflow as tf


def simple_model(x:tf.Tensor,scope:str='simple_model'):
    weights_scope=scope+'/weights'
    with tf.variable_scope(weights_scope):
        # simple_model/weightsにa,b(パラメータ)を定義(初期値は0.0としている)
        a=tf.get_variable(
            name=weights_scope+'_a',
            shape=[],dtype=tf.float32,
            initializer=tf.initializers.constant(
                value=0.0,dtype=tf.float32
            ),
            trainable=True
        )
        b=tf.get_variable(
            name=weights_scope+'_b',
            shape=[],dtype=tf.float32,
            initializer=tf.initializers.constant(
                value=0.0,dtype=tf.float32
            ),
            trainable=True
        )
    with tf.name_scope(scope+'/formula'):
        y=a*x+b
    with tf.name_scope(scope+'/log'):
        log_op=tf.strings.format('a - {} / b - {}',[a,b])
    return y,log_op

def teacher(x:np.float32):
    y=5.0*x+8.0
    return y

def train(args):
    with tf.variable_scope('inputs'):
        x=tf.placeholder(dtype=tf.float32,shape=[])
        y=tf.placeholder(dtype=tf.float32,shape=[])
    
    # setup model
    y_hat,log_op=simple_model(x)
    loss_op=tf.math.squared_difference(y,y_hat)

    # setup tensorboard log(Webブラウザに表示するためのもの)
    path=Path('./tmp/sgd_func')
    tf.summary.scalar('loss',loss_op)
    summary_op=tf.summary.merge_all()

    # setup stdout log
    logs_op=tf.print(
        tf.strings.join([log_op,tf.strings.format('loss - {}',loss_op)],'/')
    )

    # setup hyper parameter
    learning_rate=1e-3

    # setup optimizer
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_step=optimizer.minimize(loss_op) # loss_opを小さくするという最適化
    init_op=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        writer=tf.summary.FileWriter(path,sess.graph)

        for step in range(1000):
            # summary_op:Webブラウザで見るための記録
            # train_step:最適化したい問題(loss_opの出所である、simple_model関数まで遡る)
            # logs_op   :コマンドラインに出力する記録
            _x=np.random.uniform()
            _y=teacher(_x)
            feed_dict={x:_x,y:_y}
            if step%100==0:
                summary,_,_=sess.run([summary_op,train_step,logs_op],feed_dict)
            else:
                summary,_=sess.run([summary_op,train_step],feed_dict)
            writer.add_summary(summary,step)

def parse():
    # 今回はコマンドラインから取得しない
    return None

def main():
    args=parse()
    train(args)

if __name__=='__main__':
    main()
