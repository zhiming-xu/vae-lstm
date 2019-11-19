# !/usr/bin/env python3
from mxnet import autograd
import time, logging
from tqdm import tqdm
from mxnet import nd

logging.basicConfig(level=logging.INFO, \
                    format='%(asctime)s %(module)s %(levelname)-8s %(message)s', \
                    datefmt='%Y-%m-%d %H:%M:%S', \
                    handlers=[
                        logging.FileHandler("vae-lstm.log"),
                        logging.StreamHandler()
                    ])

def kl_anneal_function(anneal_function, step, k=.0025, x0=2500):
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            return min(1, step/x0)

def one_epoch(dataloader, model, trainer, ctx, is_train, epoch, lr_decay=False):
    '''
    this function trains model for one epoch if `is_train` is True, also calculates loss 
    in both training and valid
    '''
    loss_val, ppl = 0., 0.
    for n_batch, batch_sample in enumerate(tqdm(dataloader)):
        original, paraphrase = batch_sample
        original = original.as_in_context(ctx)
        paraphrase = paraphrase.as_in_context(ctx)
        if is_train:
            kl_weight = kl_anneal_function('logistic', epoch)
            with autograd.record():
                kl_loss, ce_loss = model(original, paraphrase)
                l = kl_weight * kl_loss + ce_loss
            # backward calculate
            l.backward()
            # update parmas
            trainer.step(original.shape[0])
        else:
            kl_loss, ce_loss = model(original, paraphrase)
            l = kl_loss + ce_loss
        # keep result for metric
        batch_loss = l.mean().asscalar()
        loss_val += batch_loss

    # metric
    loss_val /= (n_batch + 1)
    
    if is_train:
        logging.info('epoch %d, learning_rate %.5f, train_loss %.3f' %
                    (epoch, trainer.learning_rate, loss_val))
        # declay lr
        if epoch % 5 == 0 and lr_decay:
            trainer.set_learning_rate(trainer.learning_rate * 0.9)
    else:
        logging.info('valid_loss %.3f' % (loss_val))
    return loss_val

def train_valid(dataloader_train, dataloader_test, model, trainer, \
                num_epoch, ctx, ckpt_interval = 10, lr_decay=False):
    '''
    wrapper for training and test the model
    '''
    for epoch in range(1, num_epoch + 1):
        start = time.time()
        # train
        is_train = True
        one_epoch(dataloader_train, model, trainer, ctx, is_train, epoch, lr_decay=lr_decay)

        # valid
        is_train = False
        loss = one_epoch(dataloader_test, model, trainer, ctx, is_train, epoch)
        end = time.time()
        logging.info('time %.2f sec' % (end-start))
        logging.info("*"*48)
        if epoch % ckpt_interval == 0:
            # save params as a checkpoint every `ckpt_interval` epochs
            model.save_parameters('./params/vae-lstm%.4f.params' % loss)