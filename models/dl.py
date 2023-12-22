#  Copyright (c) 2024 Andrew
#  Email: andrewlee1807@gmail.com

import tensorflow as tf
from tensorflow.keras import layers


class ChronoPatternNet(tf.keras.Model):
    def __init__(self,
                 nb_filters=64,
                 kernel_size=3,
                 nb_stacks=2,
                 padding='same',
                 target_size=24,
                 use_skip_connections=True,
                 dropout_rate=0.2):
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_skip_connections = use_skip_connections
        self.nb_stacks = nb_stacks

        super(ChronoPatternNet, self).__init__()
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        assert padding in ['causal', 'same', 'valid']

        self.clinical_blocks = []
        # for i in range(nb_stacks):
        #     self.clinical_blocks.append(layers.Conv2D(filters=self.nb_filters,
        #                                               kernel_size=self.kernel_size,
        #                                               padding=padding,
        #                                               activation='relu',
        #                                               name=f'Conv{i}'))
        #     self.clinical_blocks.append(layers.Dropout(rate=dropout_rate))
        #     self.clinical_blocks.append(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name=f'MaxPool{i}'))

        # self.clinical_blocks.append(layers.Conv2D(filters=32,
        #                                           kernel_size=(7, 3),
        #                                           padding=padding,
        #                                           activation='relu',
        #                                           name=f'Conv0'))
        self.clinical_blocks.append(
            layers.SeparableConv2D(filters=32, kernel_size=7, padding='same'))
        # self.clinical_blocks.append(layers.BatchNormalization(axis=-1, name="block0_conv0_bn"))
        # self.clinical_blocks.append(layers.LayerNormalization(name="block0_conv0_ln"))
        # self.clinical_blocks.append(layers.Dropout(rate=dropout_rate))
        self.clinical_blocks.append(layers.Activation("relu", name="block0_conv0_act"))
        # self.clinical_blocks.append(layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), name=f'MaxPool0'))
        # self.clinical_blocks.append(layers.Conv2D(filters=64,
        #                                           kernel_size=5,
        #                                           padding=padding,
        #                                           activation='relu',
        #                                           name=f'Conv1'))
        self.clinical_blocks.append(
            layers.SeparableConv2D(filters=64, kernel_size=7, padding='same'))
        # self.clinical_blocks.append(layers.BatchNormalization(axis=-1, name="block0_conv1_bn"))
        # self.clinical_blocks.append(layers.LayerNormalization(name="block0_conv1_ln"))
        # self.clinical_blocks.append(layers.Dropout(rate=dropout_rate))
        self.clinical_blocks.append(layers.Activation("relu", name="block0_conv1_act"))
        # self.clinical_blocks.append(layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), name=f'MaxPool1'))

        # for i in range(nb_stacks):
        self.residual_layer1 = layers.Conv2D(64, (1, 1), padding="same", use_bias=False)
        self.residual_layer1_act = layers.Activation("relu", name="residual_act1")
        # self.residual_layers.append(layers.BatchNormalization(axis=1))
        # self.residual_layers.append(layers.LayerNormalization(name="residual_ln"))

        self.clinical_block2 = []
        self.clinical_block2.append(layers.Conv2D(filters=64,
                                                  kernel_size=7,
                                                  padding=padding,
                                                  name=f'block2_conv1'))
        # self.clinical_block2.append(layers.BatchNormalization(axis=1, name="block2_conv1_bn"))
        # self.clinical_block2.append(layers.LayerNormalization(name="block2_conv1_ln"))
        self.clinical_block2.append(layers.Activation("relu", name="block2_conv2_act0"))
        self.clinical_block2.append(layers.MaxPool2D(pool_size=3, strides=2, name=f'MaxPool2'))
        self.clinical_block2.append(layers.Conv2D(64,
                                                  7,
                                                  padding="same",
                                                  use_bias=False, name="block2_conv2"
                                                  ))
        # self.clinical_block2.append(layers.BatchNormalization(axis=1, name="block2_conv2_bn"))
        # self.clinical_block2.append(layers.LayerNormalization(name="block2_conv2_ln"))
        self.clinical_block2.append(layers.Activation("relu", name="block2_conv2_act1"))
        self.clinical_block2.append(layers.MaxPool2D(pool_size=2, strides=2, name=f'MaxPool3'))

        self.residual_layer2 = layers.Conv2D(32, (1, 1), padding="same", use_bias=False)
        self.residual_layer2_act = layers.Activation("relu", name="residual_act2")

        self.exit_flow = []
        self.exit_flow.append(layers.Activation("relu", name="block2_conv2_act"))
        self.exit_flow.append(layers.Conv2D(filters=32,
                                            kernel_size=5,
                                            padding=padding,
                                            activation='relu',
                                            name=f'block3_conv1'))
        # self.exit_flow.append(layers.BatchNormalization(axis=-1, name="block3_conv1_bn"))
        # self.exit_flow.append(layers.LayerNormalization(name="block3_conv1_ln"))
        self.exit_flow.append(layers.Activation("relu", name="block3_conv1_act"))

        self.flatten = layers.Flatten()
        self.final_ac = layers.Dense(50, activation='relu', name='Act_Final')
        self.dense = layers.Dense(units=target_size)

    def call(self, inputs, training=True):
        self.skip_connections = [inputs]
        x = inputs

        for clinical_block in self.clinical_blocks:
            x = clinical_block(x)

        # residual = inputs
        # residual = self.residual_layer1(residual)
        # x = layers.add([x, residual])
        # # x = self.residual_layer1_act(x)
        # residual = x

        for clinical_block in self.clinical_block2:
            x = clinical_block(x)

        # residual = self.residual_layer2(residual)
        # x = layers.add([x, residual])
        # x = self.residual_layer2_act(x)

        for exit_flow in self.exit_flow:
            x = exit_flow(x)

        # self.skip_connections.append(x)

        # if self.use_skip_connections and len(self.clinical_blocks) > 0:
        #     # x = layers.add(self.skip_connections, name='Add_Skip_Connections')
        #     x = layers.add([inputs, x], name='Skip_Connections')
        #     x = layers.GlobalAvgPool2D()(x)
        #     x = self.final_ac(x)
        x = self.flatten(x)
        # x = self.final_ac(x)
        x = self.dense(x)
        return x

    def summary(self, x):
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name="ChronoPatternNet")
        return model.summary()
