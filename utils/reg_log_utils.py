import tensorflow as tf
import numpy as np
import random
import sklearn.metrics as metrics
from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from keras.regularizers import l1, l2

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

OPTIMIZERS = [
  tf.keras.optimizers.Adadelta, # decay
  tf.keras.optimizers.Adagrad, # decay
  tf.keras.optimizers.Adam, # decay
  tf.keras.optimizers.Adamax, # decay
  tf.keras.optimizers.Ftrl, # decay
  tf.keras.optimizers.Nadam, # decay
  tf.keras.optimizers.RMSprop, # decay
  tf.keras.optimizers.SGD, # decay
]

ACTIVATIONS = [
  tf.keras.activations.elu,
  tf.keras.activations.exponential,
  tf.keras.activations.gelu,
  tf.keras.activations.hard_sigmoid,
  tf.keras.activations.linear,
  tf.keras.activations.relu,
  tf.keras.activations.selu,
  tf.keras.activations.sigmoid,
  tf.keras.activations.softplus,
  tf.keras.activations.softsign,
  tf.keras.activations.tanh
]

LOSSES = [
  # tf.keras.losses.MeanSquaredError,
  tf.keras.losses.BinaryCrossentropy
]

LEARNING_RATES = np.linspace(1e-5, 1e-1, 50)
DECAYS = np.linspace(1e-5, 1e-1, 50)
NUM_NEURONS = list(range(5, 30, 5))
EPOCHS = list(range(10, 80, 5))
BATCH_SIZES = [4, 8, 16, 32, 64, 128]

F_NAMES = {
  'act': 0,
  'opt': 1,
  'loss': 2,
  'unit': 3, 
  'lr': 4, 
  'decay': 5,
  'epoch': 6,
  'bs': 7,
  'tr_loss': 8,
  'tr_acc': 9,
  'val_loss': 10,
  'val_acc': 11}

class SVMScikit:
    def __init__(self) -> None:
        self.regr = make_pipeline(StandardScaler(), SVR(C=5, epsilon=0.15))
    
    def fit(
            self,
            X,
            y,
            epochs,
            batch_size,
            callbacks=[]
            ):
        X = np.concatenate(X, axis=1)
        self.regr.fit(X, y)

        y_pred = self.regr.predict(X)
        eval = ((y - y_pred) ** 2).mean()
        print("Fit loss: ", eval)
    
    def predict(self, X, verbose=0):
        X = np.concatenate(X, axis=1)
        y_pred = self.regr.predict(X)
        y_pred = np.reshape(y_pred, (y_pred.shape[0], -1))
        return y_pred
    
    def evaluate(self, X, y):
        X = np.concatenate(X, axis=1)
        y_pred = self.regr.predict(X)

        eval = ((y - y_pred) ** 2).mean()
        print("Eval loss: ", eval)
        return eval
        
def load_dataset(dataset_name: str = 'rejafada', 
                 seed: int = 42):
    data = np.loadtxt(f'./datasets/log_reg/{dataset_name}.csv', 
                      delimiter=',')
    X = data[:, :data.shape[1] - 1]
    y = data[:, data.shape[1] - 1].astype(int)

    # Train+Val and Test sets splitting
    X_train, X_test, y_train, y_test = \
      train_test_split(X, y, 
                      test_size=0.2,
                      random_state=seed)
    
    # Train and Val sets splitting
    X_train, X_val, y_train, y_val = \
      train_test_split(X_train, y_train, 
                      test_size=0.2,
                      random_state=seed)
    
    sc_demom = X_train.max(axis=0)
    cond = sc_demom != 0
    X_train[:, cond] /= sc_demom[cond]
    X_val[:, cond] /= sc_demom[cond]
    X_test[:, cond] /= sc_demom[cond]

    return X_train, X_val, X_test, y_train, y_val, y_test

def generate_system_features(num_systems) -> np.ndarray:
    """
    It generates system features
    INPUT:
      num_systems - int
    OUTPUT:
      x_systems - arrays, shape(num_systems, len(F_NAMES))
    """
    x_system = np.zeros((num_systems, len(F_NAMES)))

    for i in range(num_systems):
        print(f'System feature: {i + 1}')
        # neurons
        x_system[i, F_NAMES['unit']] = random.choice(NUM_NEURONS)
        # activation
        activation = random.choice(ACTIVATIONS)
        x_system[i, F_NAMES['act']] = ACTIVATIONS.index(activation)
        # optimizer
        optimizer = random.choice(OPTIMIZERS)
        x_system[i, F_NAMES['opt']] = OPTIMIZERS.index(optimizer)
        # learning rate
        x_system[i, F_NAMES['lr']] = random.choice(LEARNING_RATES)
        # decays
        x_system[i, F_NAMES['decay']] = random.choice(DECAYS)
        # loss
        loss = random.choice(LOSSES)
        x_system[i, F_NAMES['loss']] = LOSSES.index(loss)
        # epochs
        x_system[i, F_NAMES['epoch']] = random.choice(EPOCHS)
        # batch size
        x_system[i, F_NAMES['bs']] = random.choice(BATCH_SIZES)

    return x_system

def generate_system(x_system, n_features):
    model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(n_features, )),
      tf.keras.layers.Normalization(),
      tf.keras.layers.Dense(units=int(x_system[F_NAMES['unit']]),
                             activation=ACTIVATIONS[int(x_system[1])]),
      tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    optimizer = OPTIMIZERS[int(x_system[F_NAMES['opt']])](
        learning_rate=x_system[F_NAMES['lr']],           
        decay=x_system[F_NAMES['decay']])
    model.compile(
        optimizer=optimizer,
        loss=LOSSES[int(x_system[F_NAMES['loss']])](),
        metrics=['accuracy'])

    return model

def skip_connection(x, num_hidden_units, activations, is_batch=True):
    
    if len(num_hidden_units) != len(activations):
        raise ValueError("Numbers of hidden units and activations must be equal")
    y = x
    for units, activation in zip(num_hidden_units, activations):
        x = tf.keras.layers.Dense(units, activation=activation)(x)

        if is_batch:
            x = tf.keras.layers.BatchNormalization()(x)
        
    return tf.keras.layers.Concatenate(axis=1)([x, y])

def simple_assessor(n_features: int, n_system_input) -> tf.keras.Model:
    
    instance_input = tf.keras.layers.Input(shape=(n_features, ))
    ass_instance = tf.keras.layers.Normalization()(instance_input)
    ass_instance = skip_connection(ass_instance, [128, 64], ['elu', 'elu'])

    system_input = tf.keras.layers.Input(shape=(n_system_input, ))
    ass_system = tf.keras.layers.Normalization()(system_input)
    ass_system = skip_connection(ass_system, [64, 128], ['elu', 'elu'])

    x = tf.keras.layers.Concatenate(axis=1)([ass_instance, ass_system])
    x = tf.keras.layers.Dense(64, activation='elu')(x)
    x = tf.keras.layers.Dense(64, 
                              activation='elu',
                              activity_regularizer=l2(0.05))(x)
    last_layer = tf.keras.layers.Dense(1)(x)

    ass_model = tf.keras.Model([instance_input, system_input], last_layer)

    ass_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-2, decay=1e-1),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )

    return ass_model

def one_hot_encoding_system(x_system):
    
    n_systems, n_system_features = x_system.shape

    opt_types = np.arange(len(OPTIMIZERS))
    act_types = np.arange(len(ACTIVATIONS))
    loss_types = np.arange(len(LOSSES))

    # One-hot encoding
    one_hot_num = n_system_features + \
          len(act_types) + \
          len(opt_types) + \
          len(loss_types) - 3
    
    tr_x_system = np.zeros((n_systems,
                            one_hot_num))

    # Activation
    enc = np.zeros((n_systems, len(act_types)))
    enc[np.arange(n_systems), x_system[:, F_NAMES['act']].astype(int)] = 1
    col_st = 0
    col_en = len(act_types)
    tr_x_system[:, col_st:col_en] = enc
    col_st = col_en

    # Optimizer
    enc = np.zeros((n_systems, len(opt_types)))
    enc[np.arange(n_systems), x_system[:, F_NAMES['opt']].astype(int)] = 1
    col_en = col_st + len(opt_types)
    tr_x_system[:, col_st:col_en] = enc
    col_st = col_en

    # Loss
    enc = np.zeros((n_systems, len(loss_types)))
    enc[np.arange(n_systems), x_system[:, F_NAMES['loss']].astype(int)] = 1
    col_en = col_st + len(loss_types)
    tr_x_system[:, col_st:col_en] = enc
    col_st = col_en

    #the rest
    col_st = len(act_types) + len(opt_types) + len(loss_types) - 3
    tr_x_system[:, 
                col_st+F_NAMES['unit']:col_st+F_NAMES['val_acc']+1] =\
                    x_system[:, F_NAMES['unit']:F_NAMES['val_acc']+1]
    
    # Normalizing features
    cols = (tr_x_system.max(axis=0) - tr_x_system.min(axis=0)) != 0
    tr_x_system[:, cols] = (tr_x_system[:, cols] - tr_x_system[:, cols].min(axis=0)) / \
    (tr_x_system[:, cols].max(axis=0) - tr_x_system[:, cols].min(axis=0))
    
    return tr_x_system

def plot_regression_exp3(x_system, y_system, a_acc, b_acc):

    # Embeding 
    tsne = TSNE(n_components=2, n_iter=2000, perplexity=5)
    new_x_system = tsne.fit_transform(x_system)

    plt.rcParams["figure.figsize"] = (8,5)
    plt.scatter(
        new_x_system[y_system==0, 0], 
        new_x_system[y_system==0, 1], 
        c='blue',
        label='Employed systems')

    text = [f'{item}' for item in np.argwhere(y_system == 0)[:, 0] + 1]
    for i, txt in enumerate(text):
        plt.annotate(
            text[i],
            (new_x_system[y_system==0, 0][i], new_x_system[y_system==0, 1][i]))

    plt.scatter(
        new_x_system[y_system==1, 0], 
        new_x_system[y_system==1, 1], 
        c='red',
        label='Interval accuracy ({:.2f}, {:.2f})'.format(a_acc, b_acc))
    text = [f'{item}' for item in np.argwhere(y_system == 1)[:, 0] + 1]
    for i, txt in enumerate(text):
        plt.annotate(
            text[i],
            (new_x_system[y_system==1, 0][i], new_x_system[y_system==1, 1][i]))

    plt.scatter(
        new_x_system[y_system==2, 0], 
        new_x_system[y_system==2, 1], 
        c='black',
        label="Infinity loss")
    text = [f'{item}' for item in np.argwhere(y_system == 2)[:, 0] + 1]
    for i, txt in enumerate(text):
        plt.annotate(
            text[i],
            (new_x_system[y_system==2, 0][i], new_x_system[y_system==2, 1][i]))

    plt.xticks([])
    plt.yticks([])
    plt.legend(title='')

    plt.savefig('./image-results/log_reg/exp3.svg')
    plt.savefig('./image-results/log_reg/exp3.eps')

    plt.close()

def label_log_reg(y, threshold=0.5):
    y[y >= threshold] = 1
    y[y < threshold] = 0
    y = y.astype(int)
    return y

def accuracy_correction(y_pred, y_pred_error, ep, is_worst=False):
    # the system
    y_pred_label = label_log_reg(y_pred.copy())
    # correct the system prediction 
    # by the error predicted by the assessor
    cond = np.abs(y_pred_error) < ep
    if is_worst:
        cond = np.abs(y_pred_error) > ep
    y = y_pred.copy()
    y[cond] += y_pred_error[cond]
    # y += y_pred_error

    y_corr_pred_label = label_log_reg(y)

    return y_pred_label, y_corr_pred_label

def scores(y_true, y_pred, out):
    out[0] = metrics.accuracy_score(y_true, y_pred)
    out[1] = metrics.balanced_accuracy_score(y_true, y_pred)
    out[2] = metrics.precision_score(y_true, y_pred)
    out[3] = metrics.recall_score(y_true, y_pred)
    out[4] = metrics.f1_score(y_true, y_pred)

def train_assessor(X, 
                   y, 
                   x_system, 
                   systems, 
                   ass_model, 
                   random=True,
                   n_updates: int = 10):
    # Train assessor model
    # Generate assessor dataset from training dataset
    # Selecting systems randomly
    n_train = X.shape[0]
    n_systems, n_system_features = x_system.shape
    
    y_ass_train = np.zeros((n_train, ))
    x_ass_train = np.zeros((n_train, n_system_features))

    system_indices = None

    def update(epochs=1):
        for i, model in enumerate(systems):
            cond = i == system_indices
            if sum(cond) == 0: continue
            y_pred = model.predict(X[cond], verbose=0)
            y_ass_train[cond] = y_pred[:, 0]
            x_ass_train[cond] = x_system[i]

        y_ass_delta = y - y_ass_train

        print('='*20, "Training assessor on the assessor dataset", '='*20)
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=1e-8,
            patience=4,
            restore_best_weights=True)
        ass_model.fit(
            [X, x_ass_train],
            y_ass_delta,
            epochs=epochs,
            batch_size=64,
            callbacks=[])
        
        del callback
        tf.keras.backend.clear_session()

    if random:
        # Randomly take system
        system_indices = np.random.randint(0, n_systems, n_train)
        update(epochs=10)
    
    for i in range(n_updates):
        # select best systems
        a_preds = np.zeros((n_train, n_systems))
        input_x_system = np.zeros((n_train, n_system_features))
        for j in range(n_systems):
            input_x_system[np.arange(X.shape[0])] = x_system[j]
            a_preds[:, j] = ass_model.predict([X, input_x_system])[:, 0]

        system_indices = np.argmin(np.abs(a_preds), axis=1)

        update(epochs=20)

def constraint_gpu(tf, memory_in_gbs):
    
    if memory_in_gbs == 0:
        # use only CPU shortage of Memory of GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        return
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
      try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(
            memory_limit=memory_in_gbs * 1024)])
      except RuntimeError as e:
        print(e)

def release_gpu():
    from numba import cuda 
    device = cuda.get_current_device()