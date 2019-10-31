# !/usr/bin/env python3
from mxnet import autograd
import time, logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)-8s %(message)s', \
                    datefmt='%Y-%m-%d %H:%M:%S')

def one_epoch(dataloader, model, trainer, ctx, is_train, epoch, class_weight=None):
    '''
    this function trains model for one epoch if `is_train` is True
    also calculates loss/metrics whether in training or dev
    '''
    loss_val = 0.
    for n_batch, batch_sample in enumerate(dataloader):
        original, paraphrase = batch_sample
        original = original.as_in_context(ctx)
        paraphrase = paraphrase.as_in_context(ctx)
        if is_train:
            with autograd.record():
                l = model(original, paraphrase)
            # backward calculate
            l.backward()
            # update parmas
            trainer.step(original.shape[0])

        else:
            l = model(original, paraphrase)

        # keep result for metric
        batch_loss = l.mean().asscalar()
        loss_val += batch_loss

    # metric
    loss_val /= n_batch + 1

    if is_train:
        logging.info('epoch %d, learning_rate %.5f, train_loss %.4f' %
                    (epoch, trainer.learning_rate, loss_val))
        # declay lr
        if epoch % 4 == 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.9)
    else:
        logging.info('valid_loss %.4f' % (loss_val))

def train_valid(dataloader_train, dataloader_test, model, trainer, num_epoch, ctx):
    '''
    wrapper for training and "test" the model
    '''
    for epoch in range(1, num_epoch + 1):
        start = time.time()
        # train
        is_train = True
        one_epoch(dataloader_train, model, trainer, ctx, is_train, epoch)

        # valid
        is_train = False
        one_epoch(dataloader_test, model, trainer, ctx, is_train, epoch)
        end = time.time()
        logging.info('time %.2f sec' % (end-start))
        logging.info("*"*48)