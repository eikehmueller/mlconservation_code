"""Main script for training Neural network model

Parameters are set via a toml file which is passed as a command line argument:

python train_model.py --parameterfile=PARAMETERFILE

If no parameter filename is given, it defaults to training_parameters.toml
"""

import argparse
import toml
import tensorflow as tf

from data_generator import DynamicalSystemDataGenerator, KeplerDataGenerator
from dynamical_system import (
    DoubleWellPotentialSystem,
    TwoParticleSystem,
    SchwarzschildSystem,
)
from nn_models import (
    SingleParticleNNLagrangian,
    TwoParticleNNLagrangian,
    SchwarzschildNNLagrangian,
    LagrangianModel,
)
from initializer import (
    SingleParticleConstantInitializer,
    TwoParticleConstantInitializer,
    SchwarzschildConstantInitializer,
)
from kepler import KeplerSolution

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--parameterfile",
    dest="parameterfile",
    action="store",
    default="training_parameters.toml",
    help="name of toml file with training parameters",
)
cmdline_args = parser.parse_args()

print("==========================================")
print("     training neural network model")
print("==========================================")
print("")
with open(cmdline_args.parameterfile, "r", encoding="utf8") as toml_data:
    parameters = toml.load(toml_data)

print("==== parameters ====")
print(cmdline_args.parameterfile)
print("----------------- begin ----------------------")
print(toml.dumps(parameters))
print("----------------- end ------------------------")

# Set training parameters passed from file
EPOCHS = parameters["training"]["epochs"]
STEPS_PER_EPOCH = parameters["training"]["steps_per_epoch"]
BATCH_SIZE = parameters["training"]["batch_size"]
SHUFFLE_BUFFER_SIZE = 4 * BATCH_SIZE

rotation_invariant = bool(parameters["symmetry"]["rotation_invariant"])
translation_invariant = bool(parameters["symmetry"]["translation_invariant"])
reflection_invariant = bool(parameters["symmetry"]["reflection_invariant"])

# The intermediate dense layers to be used in the NNs
dense_layers = [
    tf.keras.layers.Dense(128, activation="softplus"),
    tf.keras.layers.Dense(128, activation="softplus"),
]


# ---- Select system ----
if parameters["system"]["name"] == "TwoParticle":
    dim_space = parameters["system_specific"]["two_particle"]["dim_space"]
    dim = 2 * dim_space
    initializer = TwoParticleConstantInitializer(dim)
    dynamical_system = TwoParticleSystem(
        dim_space,
        mass1=parameters["system_specific"]["two_particle"]["mass"][0],
        mass2=parameters["system_specific"]["two_particle"]["mass"][1],
        mu=parameters["system_specific"]["two_particle"]["mu"],
        kappa=parameters["system_specific"]["two_particle"]["kappa"],
    )
    nn_lagrangian = TwoParticleNNLagrangian(
        dim_space,
        dense_layers,
        rotation_invariant=rotation_invariant,
        translation_invariant=translation_invariant,
        reflection_invariant=reflection_invariant,
    )
elif parameters["system"]["name"] == "DoubleWell":
    dim = parameters["system_specific"]["double_well"]["dim"]
    initializer = SingleParticleConstantInitializer(dim)
    dynamical_system = DoubleWellPotentialSystem(
        dim=dim,
        mass=parameters["system_specific"]["double_well"]["mass"],
        mu=parameters["system_specific"]["double_well"]["mu"],
        kappa=parameters["system_specific"]["double_well"]["kappa"],
    )
    nn_lagrangian = SingleParticleNNLagrangian(
        dim,
        dense_layers,
        rotation_invariant=rotation_invariant,
        reflection_invariant=reflection_invariant,
    )
elif parameters["system"]["name"] == "Kepler":
    nn_lagrangian = SingleParticleNNLagrangian(
        3,
        dense_layers,
        rotation_invariant=rotation_invariant,
    )
elif parameters["system"]["name"] == "Schwarzschild":
    initializer = SchwarzschildConstantInitializer(
        kappa=parameters["system_specific"]["schwarzschild"]["r_s"]
    )
    dynamical_system = SchwarzschildSystem(
        kappa=parameters["system_specific"]["schwarzschild"]["r_s"]
    )
    nn_lagrangian = SchwarzschildNNLagrangian(
        dense_layers,
        rotation_invariant=rotation_invariant,
    )
else:
    print("ERROR: unknown system :" + parameters["system"])

# ---- Create data generator ----
if parameters["system"]["name"] == "Kepler":
    kepler_solution = KeplerSolution(
        mass=parameters["system_specific"]["kepler"]["mass"],
        alpha=parameters["system_specific"]["kepler"]["alpha"],
        excentricity=parameters["system_specific"]["kepler"]["excentricity"],
        energy=parameters["system_specific"]["kepler"]["energy"],
    )
    data_generator = KeplerDataGenerator(
        kepler_solution, sigma=parameters["system"]["sigma"]
    )
else:
    data_generator = DynamicalSystemDataGenerator(
        dynamical_system,
        initializer,
        re_initialize=False,
        sigma=parameters["sigma"],
        tinterval=0.1,
    )

train_batches = data_generator.dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

# ---- Construct NN model ----
model = LagrangianModel(nn_lagrangian)

learning_rate = tf.keras.optimizers.schedules.CosineDecay(
    parameters["training"]["initial_learning_rate"],
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
if parameters["saved_model"]["filename"] != "":
    nn_lagrangian.save(parameters["saved_model"]["filename"])
