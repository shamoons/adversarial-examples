import argparse
import tensorflow as tf
from cleverhans.dataset import MNIST
from cleverhans.attacks import FastGradientMethod
from cleverhans.model_zoo.basic_cnn import ModelBasicCNN

parser = argparse.ArgumentParser()
parser.add_argument('--attack', required=True,
                    choices=['fgsm'])

args = parser.parse_args()
sess = tf.Session(config=tf.ConfigProto())


if args.attack == 'fgsm':
    model = ModelBasicCNN('model1', 10, 64)
    attack = FastGradientMethod(model, sess=sess)

mnist = MNIST(train_start=0, train_end=60000)
x_train, y_train = mnist.get_set('train')

fgsm_params = {
    'eps': 0.3,
    'clip_min': 0.,
    'clip_max': 1,
    'y_target': 0
}

for i in range(len(x_train)):
    new_x = attack.generate(x_train[i], **fgsm_params)
    print(x_train[i])
    print(new_x)
    print('\n')
# fgsm2.generate(x, **fgsm_params)
