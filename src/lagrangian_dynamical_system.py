import tensorflow as tf


class LagrangianDynamicalSystem(tf.keras.layers.Layer):
    """

    The time evolution of the variable y = (q, dot{q})
    can be written as

    dy/dt = f(y)

    This class implement the function f for a given Lagrangian L.

    In this case the function can be written as

    f = (dot{q}, g)

    with

    g_k = ( (dL/dq)_j - dot{q}_i*(J_{q,dot{q}})_{ij} )*(J_{dot{q},dot{q}}^{-1})_{jk}

    and the matrices (J_{q,dot{q}})_{ij} = d^2 L / (dq_i ddot{q}_j),
    (J_{dot{q},dot{q}})_{ij} = d^2 L / (ddot{q}_i ddot{q}_j)

    """

    def __init__(self, lagrangian):
        super().__init__()
        self.dim = lagrangian.dim
        self.lagrangian = lagrangian

    @tf.function
    def _hessian(self, y, j, k):
        """Helper function for computing Hessian d^2 L / (dy_j dy_k)"""
        d_y_j = tf.unstack(tf.gradients(self.lagrangian(y), y)[0], axis=1)[j]
        return tf.unstack(tf.gradients(d_y_j, y)[0], axis=1)[k]

    @tf.function
    def div_L(self, y):
        """Gradient of Lagrangian dL/dq"""
        dL = tf.unstack(tf.gradients(self.lagrangian(y), y)[0], axis=1)
        return tf.stack(dL[: self.dim], axis=1)

    @tf.function
    def J_qdotqdot(self, y):
        """d x d matrix J_{dot{q},dot{q}}"""
        rows = []
        for j in range(self.dim, 2 * self.dim):
            row = []
            for k in range(self.dim, 2 * self.dim):
                row.append(self._hessian(y, j, k))
            rows.append(tf.stack(row, axis=1))
        return tf.stack(rows, axis=1)

    @tf.function
    def J_qqdot(self, y):
        """d x d matrix J_{q,dot{q}}"""
        rows = []
        for j in range(0, self.dim):
            row = []
            for k in range(self.dim, 2 * self.dim):
                row.append(self._hessian(y, j, k))
            rows.append(tf.stack(row, axis=1))
        return tf.stack(rows, axis=1)

    @tf.function
    def J_qdotqdot_inv(self, y):
        """d x d matrix (J_{dot{q},dot{q}})^{-1}"""
        return tf.linalg.pinv(self.J_qdotqdot(y))

    @tf.function
    def call(self, y):
        """Return value of the acceleration d^2q/dt^2(q,qdot) for a given input y

        :arg y: Phase space vector y = (q,qdot)
        """
        q_qdot = tf.unstack(y, axis=1)
        qdot = tf.stack(q_qdot[self.dim :], axis=1)
        qdotdot = tf.einsum(
            "ai,aij->aj",
            self.div_L(y) - tf.einsum("ai,aij->aj", qdot, self.J_qqdot(y)),
            self.J_qdotqdot_inv(y),
        )
        return qdotdot


class RelativisticChargedParticleLagrangianDynamicalSystem(tf.keras.layers.Layer):
    """Dynamical system derived from a relativistic Lagrangian with vector potential.

    Given (contravariant) coordinates x^mu, velocities u^mu and a vector potential
    A^mu (which depends on the position x), the time evolution of the
    state y = (x,u) = (q,dot{q}) can be written as

      dy/dt = f(q,dot{q})

    for some function

      f = (dot{q}, g)

    The components of the function g are given by:

    g^mu = ( dL/dx^nu + dA^rho/dx^nu * dL/dA^rho
             - u^rho* ((J_{x,u})_{rho,nu} + dA^sigma/dx^rho * (J_{A,u})_{sigma,nu}) )
                * (J^{-1}_{u,u})^{nu,mu}

    where the matrices J_{x,u}, J_{u,u} and J_{A,u} are defined as

    (J_{x,u})_{mu,nu} = d^2L/(dx^mu du^nu)
    (J_{u,u})_{mu,nu} = (d^L/(du^mu du^nu))
    (J_{A,u})_{mu,nu} = d^2L/(dA^mu du^nu)

    It is assumed that the functional dependence of the vector potential
    is known and passed as a tensorflow function.

    :arg lagrangian: Lagrangian L(x,u,A)
    :arg A_vec_func: Vector potential function A(x)
    """

    def __init__(self, lagrangian, A_vec_func):
        super().__init__()
        self.dim = 4
        self.lagrangian = lagrangian
        self.A_vec_func = A_vec_func

    @tf.function
    def _hessian(self, y, mu, nu):
        """Helper function for computing Hessian d^2 L / (dy^mu dy^nu)"""
        d_y_mu = tf.unstack(tf.gradients(self.lagrangian(y), y)[0], axis=1)[mu]
        return tf.unstack(tf.gradients(d_y_mu, y)[0], axis=1)[nu]

    @tf.function
    def dL_dx(self, y):
        """Gradient of Lagrangian dL/dx"""
        dL = tf.unstack(tf.gradients(self.lagrangian(y), y)[0], axis=1)
        return tf.stack(dL[:4], axis=1)

    @tf.function
    def dL_dA(self, y):
        """Partial derivative of Lagrangian with respect to A, dL/dA"""
        dL = tf.unstack(tf.gradients(self.lagrangian(y), y)[0], axis=1)
        return tf.stack(dL[8:], axis=1)

    @tf.function
    def J_xu(self, y):
        """4 x 4 matrix J_{x,u}"""
        rows = []
        for mu in range(0, 4):
            row = []
            for nu in range(4, 8):
                row.append(self._hessian(y, mu, nu))
            rows.append(tf.stack(row, axis=1))
        return tf.stack(rows, axis=1)

    @tf.function
    def J_uu(self, y):
        """4 x 4 matrix J_{u,u}"""
        rows = []
        for mu in range(4, 8):
            row = []
            for nu in range(4, 8):
                row.append(self._hessian(y, mu, nu))
            rows.append(tf.stack(row, axis=1))
        return tf.stack(rows, axis=1)

    @tf.function
    def J_Au(self, y):
        """4 x 4 matrix J_{A,u}"""
        rows = []
        for mu in range(8, 12):
            row = []
            for nu in range(4, 8):
                row.append(self._hessian(y, mu, nu))
            rows.append(tf.stack(row, axis=1))
        return tf.stack(rows, axis=1)

    @tf.function
    def dA_dx(self, x):
        """Compute the derivative of A^mu with respect to x^nu"""
        A_vec = tf.unstack(self.A_vec_func(x), axis=1)
        dA_dx_list = []
        for i in range(4):
            dA_dx_list.append(tf.gradients(A_vec[i], x, stop_gradients=[x])[0])
        return tf.stack(dA_dx_list, axis=2)

    @tf.function
    def J_uu_inv(self, y):
        """4 x 4 matrix (J_{uu})^{-1}"""
        return tf.linalg.pinv(self.J_uu(y))

    @tf.function
    def call(self, y):
        """Return value of the acceleration d^2u/dt^2(x,u,A) for a given input y = (x,u,A)

        :arg y: Phase space vector y = (q,qdot) = (x,u)
        """
        # Compute x, u and A(x) which can then be fed to the Lagrangian
        x_u = tf.unstack(y, axis=1)
        x = tf.stack(x_u[0:4], axis=1)
        u = tf.stack(x_u[4:8], axis=1)
        A_vec = tf.unstack(self.A_vec_func(x), axis=1)
        x_u_A = tf.stack(x_u + A_vec, axis=1)
        acceleration = tf.einsum(
            "ai,aij->aj",
            self.dL_dx(x_u_A)
            + tf.einsum("ajk,ak->aj", self.dA_dx(x), self.dL_dA(x_u_A))
            - tf.einsum(
                "ai,aij->aj",
                u,
                self.J_xu(x_u_A)
                + tf.einsum("ajk,akm->ajm", self.dA_dx(x), self.J_Au(x_u_A)),
            ),
            self.J_uu_inv(x_u_A),
        )
        return acceleration
