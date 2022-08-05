"""Main script for training Neural network model

Reads parameters from the file passed as a command line argument
"""

import sys
import json
import tensorflow as tf

from data_generator import DynamicalSystemDataGenerator
from dynamical_system import DoubleWellPotentialSystem, TwoParticleSystem
from nn_models import (
    SingleParticleNNLagrangian,
    TwoParticleNNLagrangian,
    LagrangianModel,
)
from initializer import (
    SingleParticleConstantInitializer,
    TwoParticleConstantInitializer,
)

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv} PARAMETERFILE")
    sys.exit(-1)

print("==========================================")
print("     training neural network model")
print("==========================================")

parameter_filename = sys.argv[1]
with open(parameter_filename, "r", encoding="utf-8") as json_data:
    parameters = json.load(json_data)
    json_data.close()


def pretty_print(d, indent):
    """Print out parameters with indents"""
    for key, value in d.items():
        if isinstance(value, dict):
            print(indent * " " + str(key))
            pretty_print(value, indent + 2)
        print(indent * " " + str(key) + " : " + str(value))


pretty_print(parameters, 2)

# Set training parameters passed from file
EPOCHS = parameters["epochs"]
STEPS_PER_EPOCH = parameters["steps_per_epoch"]
BATCH_SIZE = parameters["batch_size"]
SHUFFLE_BUFFER_SIZE = 32 * BATCH_SIZE

rotation_invariant = bool(parameters["rotation_invariant"])
translation_invariant = bool(parameters["translation_invariant"])
reflection_invariant = bool(parameters["reflection_invariant"])

# Select system
if parameters["system"] == "TwoParticle":
    dim_space = parameters["two_particle"]["dim_space"]
    dim = 2 * dim_space
    initializer = TwoParticleConstantInitializer(dim)
    dynamical_system = TwoParticleSystem(dim_space)
    nn_lagrangian = TwoParticleNNLagrangian(
        dim_space,
        rotation_invariant=rotation_invariant,
        translation_invariant=translation_invariant,
        reflection_invariant=reflection_invariant,
    )
elif parameters["system"] == "DoubleWell":
    dim = parameters["double_well"]["dim"]
    initializer = SingleParticleConstantInitializer(dim)
    dynamical_system = DoubleWellPotentialSystem(dim)
    nn_lagrangian = SingleParticleNNLagrangian(
        dim,
        rotation_invariant=rotation_invariant,
        reflection_invariant=reflection_invariant,
    )
else:
    print("ERROR: unknown system :" + parameters["system"])

# Create data generator
data_generator = DynamicalSystemDataGenerator(
    dynamical_system,
    initializer,
    re_initialize=False,
    sigma=parameters["sigma"],
    tinterval=0.1,
)
train_batches = data_generator.dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

# Construct NN model
model = LagrangianModel(nn_lagrangian)

learning_rate = tf.keras.optimizers.schedules.CosineDecay(
    parameters["initial_learning_rate"],
    EPOCHS * STEPS_PER_EPOCH,
    alpha=1.0e-2,
)

# Compile model
model.compile(
    loss="mse",
    metrics=[],
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
)
log_dir = "./tb_logs/"
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# Fit model
result = model.fit(
    train_batches,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=[tensorboard_cb],
)

#
if parameters["saved_model_filename"] != "":
    nn_lagrangian.save(parameters["saved_model_filename"])
