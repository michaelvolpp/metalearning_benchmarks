import numpy as np
import copy

from metalearning_benchmarks.benchmarks.base_benchmark import (
    MetaLearningBenchmark,
    MetaLearningTask,
)

from scipy.stats import truncnorm


def euler(f, dt, n_steps, x0):
    x = x0.copy()
    for step in range(n_steps):
        x += dt * f(x)
    return x


def furuta_rhs(
    theta_arm,  # unused, as system is invariant w.r.t. theta_arm
    theta_pendulum,
    theta_arm_dot,
    theta_pendulum_dot,
    tau1,
    tau2,
    mass_arm,
    mass_pendulum,
    length_arm,
    dist_center_of_mass_arm,
    dist_center_of_mass_pendulum,
    moment_of_inertia_com_arm,
    moment_of_inertia_com_pendulum,
    damping_arm,
    damping_pendulum,
    g,
):
    """
    Returns the equations of motion for a Furuta pendulum, taken from
    http://downloads.hindawi.com/journals/jcse/2011/528341.pdf
    A summary of this paper can be found on wikipedia:
    https://en.wikipedia.org/wiki/Furuta_pendulum

    Parameters
    ----------
    - The current state: theta_arm, theta_pendulum, theta_arm_dot,
      theta_pendulum_dot
    - The current torques: tau1, tau2
    - Additional parameters of the pendulum, cf. links above.
    g : float
        Gravitational acceleration

    Returns
    -------
    numpy.ndarray
        The temporal derivative of the state.
    """

    assert (
        theta_arm.size
        == theta_pendulum.size
        == theta_arm_dot.size
        == theta_pendulum_dot.size
        == tau1.size
        == tau2.size
    )
    n_pts = theta_arm.size

    # transform to nomenclature of paper
    # state
    theta2 = theta_pendulum
    theta1_dot = theta_arm_dot
    theta2_dot = theta_pendulum_dot

    # parameters
    m1 = mass_arm
    m2 = mass_pendulum
    L1 = length_arm
    l1 = dist_center_of_mass_arm
    l2 = dist_center_of_mass_pendulum
    J1 = moment_of_inertia_com_arm
    J2 = moment_of_inertia_com_pendulum
    b1 = damping_arm
    b2 = damping_pendulum

    # moments of inertia around axis of rotation
    J1_h = J1 + m1 * l1 ** 2
    J2_h = J2 + m2 * l2 ** 2
    J0_h = J1_h + m2 * L1 ** 2

    # equations of motion
    v1 = np.array(
        [
            theta1_dot,
            theta2_dot,
            theta1_dot * theta2_dot,
            theta1_dot ** 2,
            theta2_dot ** 2,
        ]
    )
    v2 = np.array([tau1, tau2, np.repeat(g, n_pts)])
    c = (
        J0_h * J2_h
        + J2_h ** 2 * np.sin(theta2) ** 2
        - m2 ** 2 * L1 ** 2 * l2 ** 2 * np.cos(theta2) ** 2
    )
    # acceleration of arm
    w11 = np.array(
        [
            np.repeat(-J2_h * b1, n_pts),
            m2 * L1 * l2 * np.cos(theta2) * b2,
            -(J2_h ** 2) * np.sin(2 * theta2),
            -0.5 * J2_h * m2 * L1 * l2 * np.cos(theta2) * np.sin(2 * theta2),
            J2_h * m2 * L1 * l2 * np.sin(theta2),
        ]
    )
    w12 = np.array(
        [
            np.repeat(J2_h, n_pts),
            -m2 * L1 * l2 * np.cos(theta2),
            0.5 * m2 ** 2 * l2 ** 2 * L1 * np.sin(2 * theta2),
        ]
    )
    # theta1_dotdot = (v1.dot(w11) + v2.dot(w12)) / c
    theta1_dotdot = (np.sum(v1 * w11, axis=0) + np.sum(v2 * w12, axis=0)) / c
    assert theta1_dotdot.size == n_pts

    # acceleration of pendulum
    w21 = np.array(
        [
            m2 * L1 * l2 * np.cos(theta2) * b1,
            -b2 * (J0_h + J2_h * np.sin(theta2) ** 2),
            m2 * L1 * l2 * J2_h * np.cos(theta2) * np.sin(2 * theta2),
            -0.5 * np.sin(2 * theta2) * (J0_h * J2_h + J2_h ** 2 * np.sin(theta2) ** 2),
            -0.5 * m2 ** 2 * L1 ** 2 * l2 ** 2 * np.sin(2 * theta2),
        ]
    )
    w22 = np.array(
        [
            -m2 * L1 * l2 * np.cos(theta2),
            J0_h + J2_h * np.sin(theta2) ** 2,
            -m2 * l2 * np.sin(theta2) * (J0_h + J2_h * np.sin(theta2) ** 2),
        ]
    )
    theta2_dotdot = (np.sum(v1 * w21, axis=0) + np.sum(v2 * w22, axis=0)) / c
    assert theta2_dotdot.size == n_pts

    state_dot = np.array([theta1_dot, theta2_dot, theta1_dotdot, theta2_dotdot]).T
    assert not np.isnan(state_dot).any()  # catch overflows

    return state_dot


class FurutaBenchmark(MetaLearningBenchmark):
    # cf. ABLR paper
    d_x = 4
    d_y = 4

    # fixed parameters
    std_tau1 = 0.5
    std_tau2 = 0.5
    dt = 100 * 1e-3
    sim_dt = 1e-3

    # this non-determinism breaks the noise test
    torque_noise = 0.0

    # parameter spaces
    # from http://downloads.hindawi.com/journals/jcse/2011/528341.pdf Sec. 8
    # note: we cannot make bounds dependent on sampled parameters yet, so the
    #  arm-mass can actually be located outside of the arm
    true_mass_arm = 0.300
    true_length_arm = 0.278
    true_dist_link_mass_arm = 0.150
    true_mass_pendulum = 0.075
    true_dist_link_mass_pendulum = 0.148
    true_damping_arm = 1.0e-4
    true_damping_pendulum = 2.8e-4
    low = 0.2
    high = 2.0
    low_damping = 0.2
    high_damping = 20.0

    # descriptors
    descriptors = {}
    # mass arm
    descriptors["mass_arm"] = [low * true_mass_arm, high * true_mass_arm]

    # length arm
    descriptors["length_arm"] = [low * true_length_arm, high * true_length_arm]

    # mass pendulum
    descriptors["dist_link_mass_arm"] = [0.1, high * true_dist_link_mass_arm]

    descriptors["mass_pendulum"] = [low * true_mass_pendulum, high * true_mass_pendulum]

    # length pendulum
    descriptors["dist_link_mass_pendulum"] = [
        0.1,
        high * true_dist_link_mass_pendulum,
    ]

    descriptors["damping_arm"] = [
        low_damping * true_damping_arm,
        high_damping * true_damping_arm,
    ]
    descriptors["damping_pendulum"] = [
        low_damping * true_damping_pendulum,
        high_damping * true_damping_pendulum,
    ]

    # search space
    search_space = {}

    # # theta arm
    search_space["theta_arm"] = [0.0, 2 * np.pi]

    # theta_pendulum
    search_space["theta_pendulum"] = [0.0, 2 * np.pi]

    # theta_arm_dot
    search_space["theta_arm_dot"] = [-2 * np.pi / 0.5, 2 * np.pi / 0.5]

    # theta_pendulum_dot
    search_space["theta_pendulum_dot"] = [-2 * np.pi / 0.5, 2 * np.pi / 0.5]

    d_param = 7

    param_bounds = np.array(list(descriptors.values()))
    x_bounds = np.array(list(search_space.values()))

    def __init__(
        self, n_task, n_datapoints_per_task, output_noise, seed_task, seed_x, seed_noise
    ):
        super().__init__(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise,
            seed_task=seed_task,
            seed_x=seed_x,
            seed_noise=seed_noise,
        )

        # sample from truncated normal
        std_trunc_norm = 1
        interval_centers = (self.param_bounds[:, 1] + self.param_bounds[:, 0]) / 2

        a, b = (self.param_bounds[:, 0]), (self.param_bounds[:, 1])

        # scale determines how narrow the truncnorm is
        scales = 0.1 * (b - a) / 2
        truncnorm.random_state = self.rng_param

        a_, b_ = (a - interval_centers) / scales, (b - interval_centers) / scales
        # x = [np.linspace(a, b, 1000) for a, b in zip(a, b)]
        # , size=n_task
        # self.params = np.array([truncnorm.pdf(x, a_, b_, loc=interval_center, scale=scale) for x, a_, b_, interval_center, scale in zip(x, a_, b_, interval_centers, scales)]).T
        self.params = np.array(
            [
                truncnorm.rvs(a_, b_, loc=interval_center, scale=scale, size=n_task)
                for a_, b_, interval_center, scale in zip(
                    a_, b_, interval_centers, scales
                )
            ]
        ).T

        # sample uniformly
        self.x = (
            self.rng_x.rand(n_task, n_datapoints_per_task, self.d_x)
            * (self.x_bounds[:, 1] - self.x_bounds[:, 0])
            + self.x_bounds[:, 0]
        )
        self.y = np.zeros((self.x.shape[0], self.x.shape[1], self.d_y))
        self.param_dict = []
        for i in range(self.n_task):
            # get the params as a dict, again
            self.param_dict.append(
                {k: v for k, v in zip(self.descriptors.keys(), self.params[i])}
            )
            self.param_dict[i]["std_tau1"] = self.std_tau1
            self.param_dict[i]["std_tau2"] = self.std_tau2

            self.y[i] = self(param=self.param_dict[i], x=self.x[i])

    def _get_task_by_index_without_noise(self, task_index: int) -> MetaLearningTask:
        return MetaLearningTask(
            x=self.x[task_index], y=self.y[task_index], param=self.params[task_index]
        )

    def __call__(self, param: dict, x: np.ndarray) -> np.ndarray:

        """
        Integrates the equations of motion of a Furuta pendulum for dt using
        an Euler-timestep of sim_dt. Applies random actions in each step to
        obtain physically sensible noisy behaviour.

        Parameters
        ----------
        - The current state: theta_arm, theta_pendulum, theta_arm_dot,
          theta_pendulum_dot
        - The current torques: tau1, tau2
        - Parameters of the pendulum, cf. links above.
        g : float
            Gravitational acceleration
        std_tau1, std_tau2: floats
            Standard deviations of the Gaussians from which the actions are sampled.
        prng:  numpy.random.RandomState
            Pseudo random number generator used to sample the actions.
        dt : float
            Time step
        sim_dt: float
            Time step used for Euler integration

        Returns
        -------
        numpy.ndarray
            The difference between the current state and the next state.
        float
            The query cost of evaluating the benchmark.
        """

        param = copy.deepcopy(param)
        std_tau1, std_tau2 = param.get("std_tau1"), param.get("std_tau2")

        del param["std_tau1"]
        del param["std_tau2"]

        param["dist_center_of_mass_arm"] = param["dist_link_mass_arm"]
        del param["dist_link_mass_arm"]
        param["dist_center_of_mass_pendulum"] = param["dist_link_mass_pendulum"]
        del param["dist_link_mass_pendulum"]

        param["moment_of_inertia_com_arm"] = 0.0
        param["moment_of_inertia_com_pendulum"] = 0.0

        param["g"] = 9.81

        f = lambda state: furuta_rhs(
            theta_arm=state[:, 0],
            theta_pendulum=state[:, 1],
            theta_arm_dot=state[:, 2],
            theta_pendulum_dot=state[:, 3],
            tau1=std_tau1 * self.rng_noise.randn(state.shape[0]) * self.torque_noise,
            tau2=std_tau2 * self.rng_noise.randn(state.shape[0]) * self.torque_noise,
            **param
        )
        n_steps = int(self.dt / self.sim_dt)
        theta_arm, theta_pendulum, theta_arm_dot, theta_pendulum_dot = x.T

        initial_state = np.stack(
            [theta_arm, theta_pendulum, theta_arm_dot, theta_pendulum_dot], axis=-1
        )

        next_state = euler(f=f, dt=self.sim_dt, n_steps=n_steps, x0=initial_state)
        dstate = next_state - initial_state

        return dstate


class FreeFuruta(MetaLearningBenchmark):
    def __call__(
        self,
        theta_arm,
        theta_pendulum,
        theta_arm_dot,
        theta_pendulum_dot,
        mass_arm,
        mass_pendulum,
        length_arm,
        dist_center_of_mass_arm,
        dist_center_of_mass_pendulum,
        moment_of_inertia_com_arm,
        moment_of_inertia_com_pendulum,
        damping_arm,
        damping_pendulum,
        g,
        std_tau1,
        std_tau2,
        prng,
        dt,
        sim_dt=1e-3,
        fidelity=1,
    ):
        """
        Integrates the equations of motion of a Furuta pendulum for dt using
        an Euler-timestep of sim_dt. Applies random actions in each step to
        obtain physically sensible noisy behaviour.

        Parameters
        ----------
        - The current state: theta_arm, theta_pendulum, theta_arm_dot,
          theta_pendulum_dot
        - The current torques: tau1, tau2
        - Parameters of the pendulum, cf. links above.
        g : float
            Gravitational acceleration
        std_tau1, std_tau2: floats
            Standard deviations of the Gaussians from which the actions are sampled.
        prng:  numpy.random.RandomState
            Pseudo random number generator used to sample the actions.
        dt : float
            Time step
        sim_dt: float
            Time step used for Euler integration

        Returns
        -------
        numpy.ndarray
            The difference between the current state and the next state.
        float
            The query cost of evaluating the benchmark.
        """

        f = lambda state: furuta_rhs(
            theta_arm=state[:, 0],
            theta_pendulum=state[:, 1],
            theta_arm_dot=state[:, 2],
            theta_pendulum_dot=state[:, 3],
            tau1=std_tau1 * prng.randn(state.shape[0]),
            tau2=std_tau2 * prng.randn(state.shape[0]),
            mass_arm=mass_arm,
            mass_pendulum=mass_pendulum,
            length_arm=length_arm,
            dist_center_of_mass_arm=dist_center_of_mass_arm,
            dist_center_of_mass_pendulum=dist_center_of_mass_pendulum,
            moment_of_inertia_com_arm=moment_of_inertia_com_arm,
            moment_of_inertia_com_pendulum=moment_of_inertia_com_pendulum,
            damping_arm=damping_arm,
            damping_pendulum=damping_pendulum,
            g=g,
        )
        n_steps = int(dt / sim_dt)
        theta_arm, theta_pendulum, theta_arm_dot, theta_pendulum_dot = (
            np.atleast_2d(theta_arm),
            np.atleast_2d(theta_pendulum),
            np.atleast_2d(theta_arm_dot),
            np.atleast_2d(theta_pendulum_dot),
        )
        initial_state = np.concatenate(
            [theta_arm, theta_pendulum, theta_arm_dot, theta_pendulum_dot], axis=1
        )

        next_state = euler(f=f, dt=sim_dt, n_steps=n_steps, x0=initial_state)
        dstate = next_state - initial_state

        return dstate


class FreePointmassFuruta(FurutaBenchmark):
    d_x = 4
    d_y = 4
    d_param = 7

    def __call__(self, param: dict, x: np.ndarray) -> np.ndarray:

        """
        Integrates the equations of motion of a Furuta pendulum with point masses
        placed on given distances from the links on massless rods.
        Uses action noise with std_dev_i = std_tau_i * moment_of_inertia_arm_i.
        """

        param = copy.deepcopy(param)

        mass_arm = param["mass_arm"]
        dist_link_mass_arm = param["dist_link_mass_arm"]
        mass_pendulum = param["mass_pendulum"]
        length_arm = param["length_arm"]
        dist_link_mass_pendulum = param["dist_link_mass_pendulum"]

        param["std_tau1"] = self.std_tau1 * (
            mass_arm * dist_link_mass_arm ** 2 + mass_pendulum * length_arm ** 2
        )
        param["std_tau2"] = (
            self.std_tau2 * mass_pendulum * dist_link_mass_pendulum ** 2,
        )

        return FurutaBenchmark.__call__(
            self,
            param,
            x,
        )


class FreePointmassFurutaVelocities(FreePointmassFuruta):
    d_x = 4
    d_y = 2
    d_param = 7

    def __call__(self, param: dict, x: np.ndarray) -> np.ndarray:
        """
        The FreePointmassFuruta function with only the changes in the angular velocities
        returned.
        """
        dstate = FreePointmassFuruta.__call__(
            self,
            param,
            x,
        )

        # return only the d_velocities
        return dstate[:, 2:]


class FreePointmassFurutaThetaArm(FreePointmassFuruta):
    d_x = 4
    d_y = 1
    d_param = 7

    def __call__(self, param: dict, x: np.ndarray) -> np.ndarray:
        """
        The FreePointmassFuruta function with only the changes in theta_arm
        returned.
        """
        dstate = FreePointmassFuruta.__call__(
            self,
            param,
            x,
        )

        # return only d_theta_arm
        return dstate[:, 0:1]


class FreePointmassFurutaThetaPendulum(FreePointmassFuruta):
    d_x = 4
    d_y = 1
    d_param = 7

    def __call__(self, param: dict, x: np.ndarray) -> np.ndarray:
        """
        The FreePointmassFuruta function with only the changes in theta_pendulum
        returned.
        """
        dstate = FreePointmassFuruta.__call__(
            self,
            param,
            x,
        )

        # return only d_theta_pendulum
        return dstate[:, 1:2]


class FreePointmassFurutaThetaArmDot(FreePointmassFuruta):
    d_x = 4
    d_y = 1
    d_param = 7

    def __call__(self, param: dict, x: np.ndarray) -> np.ndarray:
        """
        The FreePointmassFuruta function with only the changes in theta_arm_dot
        returned.
        """
        dstate = FreePointmassFuruta.__call__(
            self,
            param,
            x,
        )

        # return only d_theta_arm_dot
        return dstate[:, 2:3]


class FreePointmassFurutaThetaPendulumDot(FreePointmassFuruta):
    d_x = 4
    d_y = 1
    d_param = 7

    def __call__(self, param: dict, x: np.ndarray) -> np.ndarray:
        """
        The FreePointmassFuruta function with only the changes in theta_pendulum_dot
        returned.
        """
        dstate = FreePointmassFuruta.__call__(
            self,
            param,
            x,
        )

        # return only d_theta_pendulum_dot
        return dstate[:, 3:4]
