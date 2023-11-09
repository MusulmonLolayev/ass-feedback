import tensorflow as tf
import numpy as np
import random
import sklearn.metrics as metrics
from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt
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
  # tf.keras.activations.linear,
  tf.keras.activations.relu,
  tf.keras.activations.selu,
  tf.keras.activations.sigmoid,
  # tf.keras.activations.softplus,
  # tf.keras.activations.softsign,
  tf.keras.activations.tanh
]

LOSSES = [
  # tf.keras.losses.Hinge,
  # tf.keras.losses.Huber,
  # tf.keras.losses.KLDivergence,
  # tf.keras.losses.LogCosh,
  # tf.keras.losses.MeanAbsoluteError,
  # tf.keras.losses.MeanAbsolutePercentageError,
  tf.keras.losses.MeanSquaredError,
  # tf.keras.losses.MeanSquaredLogarithmicError,
  # tf.keras.losses.Poisson,
  # tf.keras.losses.SquaredHinge,
]

LEARNING_RATES = np.linspace(1e-5, 1e-1, 100)
DECAYS = np.linspace(1e-5, 1e-1, 100)
NUM_NEURONS = list(range(100, 500, 5))
EPOCHS = list(range(10, 100, 5))
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
  'rate': 8,
  'val_rate': 9,
  'mse': 10,
  'mse_val': 11,
  'tr_params': 12,
  'non_tr_params': 13}

def scaling(X, sc_denom):
    cond = sc_denom != 0
    X[:, cond] /= sc_denom[cond]

def load_dataset(dataset_name: str = 'abalone', 
                 seed: int = 42):
    data = np.loadtxt(f'./datasets/reg-one/{dataset_name}.csv', 
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
    """
    x_system = np.zeros((num_systems, len(F_NAMES)))

    for i in range(num_systems):
        print(f'System feature: {i + 1}')
        # activation
        activation = random.choice(ACTIVATIONS)
        x_system[i, F_NAMES['act']] = ACTIVATIONS.index(activation)
        # optimizer
        optimizer = random.choice(OPTIMIZERS)
        x_system[i, F_NAMES['opt']] = OPTIMIZERS.index(optimizer)
        # loss
        loss = random.choice(LOSSES)
        x_system[i, F_NAMES['loss']] = LOSSES.index(loss)
        # neurons
        x_system[i, F_NAMES['unit']] = random.choice(NUM_NEURONS)
        # learning rate
        x_system[i, F_NAMES['lr']] = random.choice(LEARNING_RATES)
        # decays
        x_system[i, F_NAMES['decay']] = random.choice(DECAYS)
        # epochs
        x_system[i, F_NAMES['epoch']] = random.choice(EPOCHS)
        # batch size
        x_system[i, F_NAMES['bs']] = random.choice(BATCH_SIZES)

    return x_system

def generate_system(x_system, n_features):
    model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(n_features, )),
      tf.keras.layers.Normalization(),
      tf.keras.layers.Dense(units=int(x_system[F_NAMES['unit']]), activation=ACTIVATIONS[int(x_system[F_NAMES['act']])]),
      tf.keras.layers.Dense(units=1)
    ])
    optimizer = OPTIMIZERS[int(x_system[F_NAMES['opt']])](
        learning_rate=x_system[F_NAMES['lr']],           
        decay=x_system[F_NAMES['decay']])
    model.compile(
        optimizer=optimizer,
        loss=LOSSES[int(x_system[F_NAMES['loss']])](),
        metrics=[tf.keras.metrics.MeanAbsoluteError()])
    x_system[F_NAMES['tr_params']] = np.sum([np.prod(v.get_shape().as_list()) \
                                             for v in model.trainable_variables])
    x_system[F_NAMES['non_tr_params']] = np.sum([np.prod(v.get_shape().as_list()) \
                                             for v in model.variables]) - \
                                             x_system[F_NAMES['tr_params']]
    return model

def rate_system(y_true, y_pred, max_error):
    errors = np.abs(y_true - y_pred)
    
    pips = 1 - errors / max_error
    pips[errors > max_error] = 0

    rate = np.mean(pips)
    return rate

def skip_connection(x, num_hidden_units, activations, is_batch=True):
    
    if len(num_hidden_units) != len(activations):
        raise ValueError("Numbers of hidden units and activations must be equal")
    y = x
    for units, activation in zip(num_hidden_units, activations):
        x = tf.keras.layers.Dense(units, activation=activation)(x)

        if is_batch:
            x = tf.keras.layers.BatchNormalization()(x)
        
    return tf.keras.layers.Concatenate(axis=1)([x, y])

def simple_assessor_reg(n_features: int, n_system_input) -> tf.keras.Model:
    
    instance_input = tf.keras.layers.Input(shape=(n_features, ))
    ass_instance = tf.keras.layers.Normalization()(instance_input)
    ass_instance = skip_connection(ass_instance, [256, 128], ['gelu', 'gelu'])

    system_input = tf.keras.layers.Input(shape=(n_system_input, ))
    ass_system = tf.keras.layers.Normalization()(system_input)
    ass_system = skip_connection(ass_system, [32, 64, 128], ['gelu', 'gelu', 'gelu'])

    combained_layer = tf.keras.layers.Concatenate(axis=1)([ass_instance, ass_system])
    combained_layer = tf.keras.layers.Dense(128, activation='elu')(combained_layer)
    combained_layer = tf.keras.layers.Dense(64, activation='elu')(combained_layer)
    last_layer = tf.keras.layers.Dense(1)(combained_layer)

    ass_model = tf.keras.Model([instance_input, system_input], last_layer)

    ass_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2, decay=5e-2),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=tf.keras.metrics.MeanAbsoluteError()
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
                col_st+F_NAMES['unit']:col_st+F_NAMES['non_tr_params']+1] =\
                    x_system[:, F_NAMES['unit']:F_NAMES['non_tr_params']+1]
    
    # Normalizing features
    cols = (tr_x_system.max(axis=0) - tr_x_system.min(axis=0)) != 0
    tr_x_system[:, cols] = (tr_x_system[:, cols] - tr_x_system[:, cols].min(axis=0)) / \
    (tr_x_system[:, cols].max(axis=0) - tr_x_system[:, cols].min(axis=0))
    
    return tr_x_system

def train_assessor(X, 
                   y, 
                   x_system,
                   x_system_categ,
                   systems, 
                   ass_model,
                   max_error,
                   noise_std=0,
                   random=True,
                   n_updates: int = 2):
    # input()
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

        errors = y - y_ass_train

        if noise_std:
            errors += np.random.normal(
                loc=0, 
                scale=noise_std, 
                size=n_train) * max_error

        # Augmentation: just duplicating
        X_aug, X_ass_aug, err_aug = [], [], []
        sys_indx, sys_counts = np.unique(system_indices, return_counts=True)
        indices = np.arange(n_train)
        for indx, count in zip(sys_indx, sys_counts):
            if count:
                val_acc = x_system_categ[indx, F_NAMES['val_rate']]
                aug_count = int(count * (2 * val_acc - 1))
                cond = np.logical_and(indx == system_indices, errors > max_error)

                if any(cond) and aug_count > 0:
                    print(aug_count, sum(cond))
                    aug_indices = np.random.choice(indices[cond], aug_count)

                    X_aug_current = X[aug_indices]
                    error_aug_current = errors[aug_indices] 

                    X_aug_current += 0.1 * np.random.normal(0.5, 0.01, X_aug_current.shape)
                    error_aug_current += 0.1 * np.random.normal(0, 0.01, error_aug_current.shape)
                    
                    X_aug.append(X_aug_current)
                    X_ass_aug.append(x_ass_train[aug_indices])
                    err_aug.append(error_aug_current)

        
        X_train = np.concatenate([X] + X_aug, axis=0)
        X_ass = np.concatenate([x_ass_train] + X_ass_aug, axis=0)
        y_errors = np.concatenate([errors] + err_aug)
        
        # X_train = X
        # X_ass = x_ass_train
        # y_errors = errors

        print("Class-imbalance: ", sum(y_errors < max_error) / X_train.shape[0])
        # input("")

        error_trunc = 50

        # truncating higher errors during traning
        cond = y_errors < error_trunc
        print(f"Number of truncated errors (training): {sum(cond)} / {len(cond)}")
        X_train = X_train[cond]
        X_ass = X_ass[cond]
        y_errors = y_errors[cond]

        print('='*20, "Training assessor on the assessor dataset", '='*20)
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='mean_absolute_error',
            min_delta=1e-4,
            patience=4,
            restore_best_weights=True)
        ass_model.fit(
            [X_train, X_ass],
            y_errors,
            epochs=epochs,
            batch_size=64,
            callbacks=[],
            verbose=2)
        
        del callback
        tf.keras.backend.clear_session()

    # Warm up by randomly training assessor model
    if random:
        # Randomly take system
        for i in range(5):
            system_indices = np.random.randint(0, n_systems, n_train)
            update(epochs=500)
    
    for i in range(n_updates):
        # select best systems
        a_preds = np.zeros((n_train, n_systems))
        input_x_system = np.zeros((n_train, n_system_features))
        for j in range(n_systems):
            input_x_system[np.arange(X.shape[0])] = x_system[j]
            a_preds[:, j] = ass_model.predict([X, input_x_system], batch_size=8, verbose=0)[:, 0]

        system_indices = np.argmin(np.abs(a_preds), axis=1)

        update(epochs=500)

def plot_regression_exp1(tendency, save_file_name):
    indices = np.arange(tendency.shape[0]) + 1

    plt.rcParams["figure.figsize"] = (15,4)

    # plt.step(indices, tendency[:, 0], label='Accuracy')
    plt.plot(indices, tendency[:, 0], 'o--', alpha=0.3, label='Accuracy')

    # plt.step(indices, tendency[:, 1], where='mid', label='Balanced accuracy')
    plt.plot(indices, tendency[:, 1], '*--', alpha=0.3, label='Balanced accuracy')

    # plt.step(indices, tendency[:, 2], where='mid', label='$F_1$ score')
    plt.plot(indices, tendency[:, 2], '+--', alpha=0.3, label='$F_1$ score')

    # plt.step(indices, tendency[:, 3], where='mid', label='Precision')
    plt.plot(indices, tendency[:, 3], '*--', alpha=0.3, label='Precision')

    # plt.step(indices, tendency[:, 4], where='mid', label='Recall')
    plt.plot(indices, tendency[:, 4], '+--', alpha=0.3, label='Recall')

    plt.xticks(indices)
    plt.yticks(np.linspace(0, 1, 11))
    plt.ylim(0, 1)
    plt.plot([0, 62], [0.5, 0.5])

    plt.xlabel('Test set number')
    plt.ylabel('Average assessor accuracy')
    plt.grid(axis='x', color='0.95')
    plt.legend()

    # plt.title('Accuracies of the best systems on 60 splits of the test set')
    plt.savefig(save_file_name + '.svg')
    plt.savefig(save_file_name + '.eps')

    plt.close()

def plot_regression_exp2(
        best_errors,
        best_corrector_errors,
        worst_errors,
        worst_corrector_errors,
        sel_errors,
        sel_corrector_errors,
        RES_DIR):
    indices = np.arange(len(best_errors)) + 1
    plt.rcParams["figure.figsize"] = (15,4)
    s = 20

    plt.scatter(indices, best_errors, label='The best', s=s, marker='*')
    plt.scatter(indices, best_corrector_errors, label='The best (corrected)', s=s, marker='*')
    plt.scatter(indices, worst_errors, label='The worst', s=s, marker='x')
    plt.scatter(indices, worst_corrector_errors, label='The worst (corrected)', s=s, marker='x')
    plt.scatter(indices, sel_errors, label='The selected', s=s)
    plt.scatter(indices, sel_corrector_errors, label='The selected (corrected)', s=s)


    plt.xticks(indices)
    # plt.yticks(np.arange(30))
    # plt.ylim(0, 40)

    plt.grid(axis='x', color='0.90')
    plt.legend()
    plt.xlabel('Test set number')
    plt.ylabel('Absolute average error')
    
    plt.savefig(f'{RES_DIR}/exp2.svg')
    plt.savefig(f'{RES_DIR}/exp2.eps')

    plt.close()

def plot_regression_exp3(x_system, y_system, baseline_acc, RES_DIR):
    x_system = one_hot_encoding_system(x_system)
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
        label='Low-interval accuracy ({:.2f})'.format(baseline_acc))
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

    plt.savefig(f'{RES_DIR}/exp3.svg')
    plt.savefig(f'{RES_DIR}/exp3.eps')

    plt.close()

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