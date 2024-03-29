"""Main script for training Neural network model

The models are trained on (noisy) synthetic data, which is obtained by solving the true
dynamical system with a four order Runge Kutta (RK4) integrator; for the Kepler system (motion
in a 1/r potential) the exact solution is known, so this is used instead in this case.
During training, the mean square error between the predicted acceleration and the true
acceleration is minimised.

Parameters are set via a toml file which is passed as a command line argument:

python train_model.py --parameterfile=PARAMETERFILE

If no parameter filename is given, it defaults to training_parameters.toml
"""

import os
import argparse
from shutil import copyfile
import toml
import numpy as np
import tensorflow as tf

from conservative_nn.data_generator import (
    DynamicalSystemDataGenerator,
    KeplerDataGenerator,
)
from conservative_nn.dynamical_system import (
    DoubleWellPotentialSystem,
    TwoParticleSystem,
    SchwarzschildSystem,
)
from conservative_nn.nn_lagrangian import (
    SingleParticleNNLagrangian,
    TwoParticleNNLagrangian,
    SchwarzschildNNLagrangian,
)
from conservative_nn.nn_lagrangian_model import NNLagrangianModel
from conservative_nn.initializer import (
    SingleParticleConstantInitializer,
    TwoParticleConstantInitializer,
    SchwarzschildConstantInitializer,
)
from conservative_nn.kepler import KeplerSolution

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--parameterfile",
    dest="parameterfile",
    action="store",
    default="training_parameters.toml",
    help="name of toml file with training parameters",
)
parser.add_argument(
    "--tblogdir",
    dest="tblogdir",
    action="store",
    default="tb_logs",
    help="name of directory used for tensorboard logging",
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
print(f"writing tensorboard logs to {cmdline_args.tblogdir}")

# Set training parameters passed from file
EPOCHS = parameters["training"]["epochs"]
STEPS_PER_EPOCH = parameters["training"]["steps_per_epoch"]
BATCH_SIZE = parameters["training"]["batch_size"]
SHUFFLE_BUFFER_SIZE = 128 * BATCH_SIZE

rotation_invariant = bool(parameters["symmetry"]["rotation_invariant"])
translation_invariant = bool(parameters["symmetry"]["translation_invariant"])
reflection_invariant = bool(parameters["symmetry"]["reflection_invariant"])

# The intermediate dense layers to be used in the NNs
n_units = 128
dense_layers = [
    tf.keras.layers.Dense(
        n_units,
        activation="softplus",
        kernel_initializer=tf.keras.initializers.RandomNormal(
            stddev=2.0 / np.sqrt(n_units), seed=214127
        ),
    ),
    tf.keras.layers.Dense(
        n_units,
        activation="softplus",
        kernel_initializer=tf.keras.initializers.RandomNormal(
            stddev=1.0 / np.sqrt(n_units), seed=775411
        ),
    ),
]


# ---- Select system ----
if parameters["system"]["name"] == "TwoParticle":
    dim_space = parameters["system_specific"]["two_particle"]["dim_space"]
    dim = 2 * dim_space
    initializer = TwoParticleConstantInitializer(
        dim,
        mass1=parameters["system_specific"]["two_particle"]["mass"][0],
        mass2=parameters["system_specific"]["two_particle"]["mass"][1],
    )
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
        parameters["system_specific"]["schwarzschild"]["r_s"]
    )
    dynamical_system = SchwarzschildSystem(
        parameters["system_specific"]["schwarzschild"]["r_s"]
    )
    nn_lagrangian = SchwarzschildNNLagrangian(
        dense_layers,
        rotation_invariant=rotation_invariant,
        time_independent=parameters["system_specific"]["schwarzschild"][
            "time_independent"
        ],
    )
else:
    print("ERROR: unknown system :" + parameters["system"])

# ---- Create data generator ----
if parameters["system"]["name"] == "Kepler":
    kepler_solution = KeplerSolution(
        mass=parameters["system_specific"]["kepler"]["mass"],
        alpha=parameters["system_specific"]["kepler"]["alpha"],
        excentricity=parameters["system_specific"]["kepler"]["excentricity"],
        angular_momentum=parameters["system_specific"]["kepler"]["angular_momentum"],
    )
    data_generator = KeplerDataGenerator(
        kepler_solution, sigma=parameters["system"]["sigma"]
    )
else:
    data_generator = DynamicalSystemDataGenerator(
        dynamical_system,
        initializer,
        re_initialize=False,
        sigma=parameters["system"]["sigma"],
        tinterval=0.1,
    )

train_batches = (
    data_generator.dataset.shuffle(SHUFFLE_BUFFER_SIZE, seed=21845)
    .prefetch(tf.data.AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=True)
)

# ---- Construct NN model ----
model = NNLagrangianModel(nn_lagrangian)

if parameters["training"]["schedule"] == "Constant":
    learning_rate = parameters["training"]["learning_rate"]
elif parameters["training"]["schedule"] == "CosineDecay":
    learning_rate = tf.keras.optimizers.schedules.CosineDecay(
        parameters["training"]["learning_rate"],
        EPOCHS * STEPS_PER_EPOCH,
        alpha=parameters["training"]["alpha"],
    )
else:
    schedule = parameters["training"]["schedule"]
    raise RuntimeError(f"Unknown traning schedule {schedule}")


# ---- Compile model ----
model.compile(
    loss="mse",
    metrics=[],
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
)

tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=cmdline_args.tblogdir)

# ---- Fit model ----
result = model.fit(
    train_batches,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=[tensorboard_cb],
)

#
if parameters["saved_model"]["filename"] != "":
    # Save model
    nn_lagrangian.save(parameters["saved_model"]["filename"])
    # Copy parameter file to directory with saved model
    copyfile(
        cmdline_args.parameterfile,
        os.path.join(parameters["saved_model"]["filename"], "training_parameters.toml"),
    )
