import tensorflow as tf
from tensorflow import distributions
import numpy as np


def evaluate_sample(model, data_sample, session):
    """
    Samples parameter realizations from the variational posterior distributions and 
    performs inference
    
    Args: 
        model: the model to evaluate
        data_iterator: an iterator that iterates over the samples and lables
        session: the session to run the evaluation in
    """
    with tf.name_scope("Evaluation"):
        q       = model.q._probs

        res = []
        ref = []
        while True:
            try:
                r,y = session.run((q, data_sample['y']))
                res.append(r)
                ref.append(y)
            except tf.errors.OutOfRangeError:
                break

        res=np.argmax(np.concatenate(res, axis=0),axis = -1)
        ref=np.argmax(np.concatenate(ref, axis=0),axis = -1)

        return np.float(np.sum(res==ref)) / len(ref)

def evaluate_sample_xchange(model, data_sample, session):
    """
    Samples parameter realizations from the variational posterior distributions and
    performs inference

    Args:
        model: the model to evaluate
        data_iterator: an iterator that iterates over the samples and lables
        session: the session to run the evaluation in
    """
    with tf.name_scope("Evaluation"):
        q = model.q._probs
        q_uni = model.q_uniform._probs
        res = []
        ref = []
        res_uni = []
        ref_uni = []
        while True:
            try:
                r, y = session.run((q, data_sample['y']))
                r_uni,y_uni  = session.run((q_uni,data_sample['y']))
                res.append(r)
                ref.append(y)
                res_uni.append(r_uni)
                ref_uni.append(y_uni)
            except tf.errors.OutOfRangeError:
                break

        res = np.argmax(np.concatenate(res, axis=0), axis=-1)
        ref = np.argmax(np.concatenate(ref, axis=0), axis=-1)
        res_uni = np.argmax(np.concatenate(res_uni, axis=0), axis=-1)
        ref_uni = np.argmax(np.concatenate(ref_uni, axis=0), axis=-1)

        return np.float(np.sum(res == ref)) / len(ref),np.float(np.sum(res_uni == ref_uni)) / len(ref_uni)