from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import Huber

from models.dl import *
from models.baselinemodels import LSTMModel, GRUModel, MLPModel, TCN_Vanilla

# List model name
model1_str = "Chrono"

# comparing model names
tcn_model_str = "TCN"
lstm_model_str = "LSTM"
gru_model_str = "GRU"
mlp_model_str = "MLP"


def build_callbacks(tensorboard_log_dir='logs', tensorboard_name=None):
    """Control and tracking learning process during training phase"""
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   mode='min')

    reduceLR = ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)

    tensorboard_callback = TensorBoard(tensorboard_log_dir)

    callbacks = [
        early_stopping,
        reduceLR,
        tensorboard_callback
    ]
    return callbacks


def compile_model(model, config):
    # from tensorflow.keras.models import Sequential
    # from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense, Input
    #
    # model = Sequential()
    # model.add(Input(shape=(24, 7, 1)))
    # for i in range(config['nb_stacks']):
    #     model.add(Conv2D(filters=config['nb_filters'],
    #                      kernel_size=config['kernel_size'],
    #                      padding='same',
    #                      activation='relu',
    #                      name=f'Conv{i}'))
    #     model.add(Dropout(rate=0.0))
    #     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=f'MaxPool{i}'))
    #
    # model.add(Flatten())
    # model.add(Dense(50, activation='relu', name='Act_Final'))
    # model.add(Dense(units=1))
    # model.summary()

    # print model
    input_test = Input(shape=(7, 24, 1))
    # model.build(input_test)
    model.summary(input_test)
    # Build model
    model.compile(loss=Huber(),
                  optimizer=config['optimizer'],
                  metrics=config['metrics'])

    return model


def initialize_model1(config):
    model = ChronoPatternNet(nb_filters=config['nb_filters'],
                             kernel_size=config['kernel_size'],
                             nb_stacks=config['nb_stacks'],
                             target_size=config['output_length'])

    return compile_model(model, config)


def initialize_tcn_model(config):
    from tcnbased.tcn_family import TCN_Vanilla
    model = TCN_Vanilla(input_width=config['input_width'],
                        dilations=config['list_dilation'],
                        nb_filters=config['nb_filters'],
                        kernel_size=config['kernel_size'],
                        num_features=len(config['features']),
                        target_size=config['output_length'])

    return compile_model(model, config)


def initialize_baseline_model(config, model_class):
    baseline_model = model_class(input_width=config['input_width'],
                                 num_hidden_layer=config['num_hidden_layer'],
                                 num_features=len(config['features']),
                                 output_length=config['output_length'])

    baseline_model.compile_model(optimizer=config['optimizer'],
                                 metrics=config['metrics'])

    return baseline_model.model


def get_model(model_name: str, config) -> object:
    model_name = model_name.upper()
    if model_name == model1_str.upper():
        return initialize_model1(config)
    elif model_name == tcn_model_str.upper():
        return initialize_tcn_model(config)
    elif model_name == lstm_model_str.upper():
        return initialize_baseline_model(config, LSTMModel)
    elif model_name == gru_model_str.upper():
        return initialize_baseline_model(config, GRUModel)
    elif model_name == mlp_model_str.upper():
        return initialize_baseline_model(config, MLPModel)

    return None
