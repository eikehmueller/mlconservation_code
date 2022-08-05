"""Main script for training Neural network model

Parameters are set via a json file which is passed as a command line argument:

python train_model.py --parameterfile=PARAMETERFILE

If no parameter filename is given, it defaults to training_parameters.py.
"""

import json
import argparse
import tensorflow as tf

from data_generator import DynamicalSystemDataGenerator, KeplerDataGenerator
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
from kepler import KeplerSolution

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--parameterfile",
    dest="parameterfile",
    action="store",
    default="training_parameters.json",
    help="name of json file with training parameters",
)
cmdline_args = parser.parse_args()

print("==========================================")
print("     training neural network model")
print("==========================================")

with open(cmdline_args.parameterfile, "r", encoding="utf-8") as json_data:
    parameters = json.load(json_data)
    json_data.close()


def pretty_print(d, indent=0):
    """Print out dictionary with indents

    :arg d: dictionary to print
    :arg indent: indent to use
    """
    for key, value in d.items():
        if isinstance(value, dict):
            print(indent * " " + str(key))
            pretty_print(value, indent + 4)
        else:
            print(indent * " " + str(key) + " : " + str(value))


print("==== parameters ====")
print("reading parameters from " + cmdline_args.parameterfile)
pretty_print(parameters)
print("")

# Set training parameters passed from file
EPOCHS = parameters["epochs"]
STEPS_PER_EPOCH = parameters["steps_per_epoch"]
BATCH_SIZE = parameters["batch_size"]
SHUFFLE_BUFFER_SIZE = 4 * BATCH_SIZE

rotation_invariant = bool(parameters["rotation_invariant"])
translation_invariant = bool(parameters["translation_invariant"])
reflection_invariant = bool(parameters["reflection_invariant"])

# ---- Select system ----
if parameters["system"] == "TwoParticle":
    dim_space = parameters["two_particle"]["dim_space"]
    dim = 2 * dim_space
    initializer = TwoParticleConstantInitializer(dim)
    dynamical_system = TwoParticleSystem(
        dim_space,
        mass1=parameters["two_particle"]["mass"][0],
        mass2=parameters["two_particle"]["mass"][1],
        mu=parameters["two_particle"]["mu"],
        kappa=parameters["two_particle"]["kappa"],
    )
    nn_lagrangian = TwoParticleNNLagrangian(
        dim_space,
        rotation_invariant=rotation_invariant,
        translation_invariant=translation_invariant,
        reflection_invariant=reflection_invariant,
    )
elif parameters["system"] == "DoubleWell":
    dim = parameters["double_well"]["dim"]
    initializer = SingleParticleConstantInitializer(dim)
    dynamical_system = DoubleWellPotentialSystem(
        dim=dim,
        mass=parameters["double_well"]["mass"],
        mu=parameters["double_well"]["mu"],
        kappa=parameters["double_well"]["kappa"],
    )
    nn_lagrangian = SingleParticleNNLagrangian(
        dim,
        rotation_invariant=rotation_invariant,
        reflection_invariant=reflection_invariant,
    )
elif parameters["system"] == "Kepler":
    nn_lagrangian = SingleParticleNNLagrangian(
        3,
        rotation_invariant=rotation_invariant,
        reflection_invariant=reflection_invariant,
    )
else:
    print("ERROR: unknown system :" + parameters["system"])

# ---- Create data generator ----
if parameters["system"] in ["DoubleWell", "TwoParticle"]:
    data_generator = DynamicalSystemDataGenerator(
        dynamical_system,
        initializer,
        re_initialize=False,
        sigma=parameters["sigma"],
        tinterval=0.1,
    )
else:
    kepler_solution = KeplerSolution(
        mass=parameters["kepler"]["mass"],
        alpha=parameters["kepler"]["alpha"],
        excentricity=parameters["kepler"]["excentricity"],
        energy=parameters["kepler"]["energy"],
    )
    data_generator = KeplerDataGenerator(kepler_solution, sigma=parameters["sigma"])

train_batches = data_generator.dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

# ---- Construct NN model ----
model = LagrangianModel(nn_lagrangian)

learning_rate = tf.keras.optimizers.schedules.CosineDecay(
    parameters["initial_learning_rate"],
    EPOCHS * STEPS_PER_EPOCH,
    alpha=1.0e-2,
)

# ---- Compile model ----
model.compile(
    loss="mse",
    metrics=[],
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
)
log_dir = "./tb_logs/"
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# ---- Fit model ----
result = model.fit(
    train_batches,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=[tensorboard_cb],
)

#
if parameters["saved_model_filename"] != "":
    nn_lagrangian.save(parameters["saved_model_filename"])
