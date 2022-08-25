import copy

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import truncnorm
from tqdm import tqdm

from metalearning_benchmarks import MetaLearningBenchmark, MetaLearningTask


def euler(f, dt, n_steps, x0):
    x = x0.copy()
    for _ in range(n_steps):
        x += dt * f(None, x)  # does not implement time dependent f
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
    - g : float
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
        == 1
    )

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
    J1_h = J1 + m1 * l1**2
    J2_h = J2 + m2 * l2**2
    J0_h = J1_h + m2 * L1**2

    # equations of motion
    v1 = np.array(
        [
            theta1_dot,
            theta2_dot,
            theta1_dot * theta2_dot,
            theta1_dot**2,
            theta2_dot**2,
        ]
    )
    v2 = np.array([tau1, tau2, g])
    c = (
        J0_h * J2_h
        + J2_h**2 * np.sin(theta2) ** 2
        - m2**2 * L1**2 * l2**2 * np.cos(theta2) ** 2
    )
    # acceleration of arm
    w11 = np.array(
        [
            -J2_h * b1,
            m2 * L1 * l2 * np.cos(theta2) * b2,
            -(J2_h**2) * np.sin(2 * theta2),
            -0.5 * J2_h * m2 * L1 * l2 * np.cos(theta2) * np.sin(2 * theta2),
            J2_h * m2 * L1 * l2 * np.sin(theta2),
        ]
    )
    w12 = np.array(
        [
            J2_h,
            -m2 * L1 * l2 * np.cos(theta2),
            0.5 * m2**2 * l2**2 * L1 * np.sin(2 * theta2),
        ]
    )
    theta1_dotdot = (np.sum(v1 * w11, axis=0) + np.sum(v2 * w12, axis=0)) / c

    # acceleration of pendulum
    w21 = np.array(
        [
            m2 * L1 * l2 * np.cos(theta2) * b1,
            -b2 * (J0_h + J2_h * np.sin(theta2) ** 2),
            m2 * L1 * l2 * J2_h * np.cos(theta2) * np.sin(2 * theta2),
            -0.5 * np.sin(2 * theta2) * (J0_h * J2_h + J2_h**2 * np.sin(theta2) ** 2),
            -0.5 * m2**2 * L1**2 * l2**2 * np.sin(2 * theta2),
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

    state_dot = np.array([theta1_dot, theta2_dot, theta1_dotdot, theta2_dotdot]).T
    assert not np.isnan(state_dot).any()  # catch overflows

    return state_dot


class FreePointMassFurutaBenchmark(MetaLearningBenchmark):
    """
    Furuta pendulum with point masses.
    - Generates full trajectories with dt = 0.1s, sim_dt = 0.001s
    - Applies torque noise in each step proportional to arm/pendulum moment of inertia
    - Normalizes x:
        - x_norm = s * x -> x_dot_norm = s * x_dot
        - solves x_dot_norm = s * x_dot = s * f(x) = s * f(x_norm / s)
    """

    ### For testing ###
    # Dynamical systems ...
    # - generate noise not via output noise
    # - store trajectories, so the x's (= states) are not sampled independently, but
    #   depend on the chosen parameters, i.e., only the same param-seed together with
    #   the same x-seed generates the same x's
    is_dynamical_system = True

    ### Input-output dimensions ###
    d_x = 4
    d_y = 4

    ### The parameter space ###
    ## Define a set of sensible parameters around which we sample the systems
    # From http://downloads.hindawi.com/journals/jcse/2011/528341.pdf Sec. 8
    # Note: we cannot make bounds dependent on previously sampled parameters yet, so the
    #  arm-mass can actually be located outside of the arm
    d_param = 7
    true_mass_arm = 0.300
    true_length_arm = 0.278
    true_dist_link_mass_arm = 0.150
    true_mass_pendulum = 0.075
    true_dist_link_mass_pendulum = 0.148
    true_damping_arm = 2 * 1.0e-4  # doubled in comparison to paper
    true_damping_pendulum = 2 * 2.8e-4  # doubled in comparison to paper
    ## Define the parameter bounds from which the systems are sampled
    low = 0.5
    high = 2.0
    param_bounds = dict()
    param_bounds["mass_arm"] = [low * true_mass_arm, high * true_mass_arm]
    param_bounds["length_arm"] = [low * true_length_arm, high * true_length_arm]
    param_bounds["dist_link_mass_arm"] = [0.1, high * true_dist_link_mass_arm]
    param_bounds["mass_pendulum"] = [low * true_mass_pendulum, high * true_mass_pendulum]  # fmt:skip
    param_bounds["dist_link_mass_pendulum"] = [0.1, high * true_dist_link_mass_pendulum]
    param_bounds["damping_arm"] = [low * true_damping_arm, high * true_damping_arm]  # fmt:skip
    param_bounds["damping_pendulum"] = [low * true_damping_pendulum, high * true_damping_pendulum]  # fmt:skip

    ### Define the search space ###
    search_space = dict()
    search_space["theta_arm"] = [0.0, 2 * np.pi]
    search_space["theta_pendulum"] = [0.0, 2 * np.pi]
    search_space["theta_arm_dot"] = [-2 * np.pi / 0.5, 2 * np.pi / 0.5]
    search_space["theta_pendulum_dot"] = [-2 * np.pi / 0.5, 2 * np.pi / 0.5]
    x_bounds = np.array(list(search_space.values()))

    ### Define the action noise ###
    # TODO: this non-determinism breaks the noise test
    std_tau1_base = 0.5
    std_tau2_base = 0.5
    torque_noise = 1.0

    ### Define state normalizer ###
    norm_scale = 1 / np.array([0.39981845, 3.297136, 1.4318603, 7.853314])
    # norm_scale = np.ones((d_x,))

    ### Integrator parameters ###
    sim_dt = 1e-3
    dt = 1e-1

    def __init__(
        self, n_task, n_datapoints_per_task, output_noise, seed_task, seed_x, seed_noise
    ):
        assert (
            output_noise == 0.0
        ), "Do not specify output noise! Noise is generated through random torques!"
        super().__init__(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise,
            seed_task=seed_task,
            seed_x=seed_x,
            seed_noise=seed_noise,
        )

        ### Sample the parameters (from uniform) ###
        param_bounds = np.array(list(self.param_bounds.values()))
        params = self.rng_param.rand(n_task, self.d_param)
        params = params * (param_bounds[:, 1] - param_bounds[:, 0]) + param_bounds[:, 0]
        self.params = dict(zip(self.param_bounds.keys(), params.T))

        ### Define the initial state: pendulum balances in upward position ###
        self.theta_arm_0 = 0.0
        self.theta_pendulum_0 = np.pi
        self.theta_arm_dot_0 = 0.0
        self.theta_pendulum_dot_0 = 0.0

        ### Integrate the EOMs ###
        self.x = np.zeros((n_task, n_datapoints_per_task, self.d_x))
        self.x[:, 0, :] = self.norm_scale * [
            self.theta_arm_0,
            self.theta_pendulum_0,
            self.theta_arm_dot_0,
            self.theta_pendulum_dot_0,
        ]
        self.y = np.zeros((n_task, n_datapoints_per_task, self.d_y))
        for l in (pbar := tqdm(range(self.n_task), desc="Generating tasks")):
            for t in range(self.n_datapoints_per_task):
                dx = self._compute_next_state(
                    theta_arm=self.x[l, t, 0],
                    theta_pendulum=self.x[l, t, 1],
                    theta_arm_dot=self.x[l, t, 2],
                    theta_pendulum_dot=self.x[l, t, 3],
                    mass_arm=self.params["mass_arm"][l],
                    mass_pendulum=self.params["mass_pendulum"][l],
                    length_arm=self.params["length_arm"][l],
                    dist_link_mass_arm=self.params["dist_link_mass_arm"][l],
                    dist_link_mass_pendulum=self.params["dist_link_mass_pendulum"][l],
                    damping_arm=self.params["damping_arm"][l],
                    damping_pendulum=self.params["damping_pendulum"][l],
                )
                self.y[l, t] = dx
                if t < self.n_datapoints_per_task - 1:
                    self.x[l, t + 1] = self.x[l, t] + dx

    def _get_task_by_index_without_noise(self, task_index: int) -> MetaLearningTask:
        return MetaLearningTask(x=self.x[task_index], y=self.y[task_index])

    def _compute_next_state(
        self,
        theta_arm: float,
        theta_pendulum: float,
        theta_arm_dot: float,
        theta_pendulum_dot: float,
        mass_arm: float,
        mass_pendulum: float,
        length_arm: float,
        dist_link_mass_arm: float,
        dist_link_mass_pendulum: float,
        damping_arm: float,
        damping_pendulum: float,
    ) -> np.ndarray:
        """
        Integrates the equations of motion of a Furuta pendulum for dt using
        an Euler-timestep of sim_dt.
        The pendulum masses are modeled as point masses placed on given distances
        from the links on massless rods.
        Uses action noise with std_dev_i = std_tau_i * moment_of_inertia_arm_i.

        Parameters
        ----------
        The current state, action noise std deviations, and the system parameters.

        Returns
        -------
        numpy.ndarray
            The difference between the current state and the next state.
        """

        ### Define the action noise std dev propto of the moment of inertia ###
        moi_arm = mass_arm * dist_link_mass_arm**2 + mass_pendulum * length_arm**2
        moi_pendulum = mass_pendulum * dist_link_mass_pendulum**2
        std_tau1 = self.std_tau1_base * moi_arm
        std_tau2 = self.std_tau2_base * moi_pendulum

        ### The integrator parameters ###
        n_steps = int(self.dt / self.sim_dt)
        initial_state = np.array(
            [theta_arm, theta_pendulum, theta_arm_dot, theta_pendulum_dot]
        )

        ### Integrate ###
        f = lambda t, state: self.norm_scale * furuta_rhs(
            theta_arm=state[0] / self.norm_scale[0],
            theta_pendulum=state[1] / self.norm_scale[1],
            theta_arm_dot=state[2] / self.norm_scale[2],
            theta_pendulum_dot=state[3] / self.norm_scale[3],
            # use rng_x (not rng_noise) as the torque-noise determines the states
            # (i.e., the x's) of the trajectories
            tau1=std_tau1 * self.rng_x.randn() * self.torque_noise,
            tau2=std_tau2 * self.rng_x.randn() * self.torque_noise,
            mass_arm=mass_arm,
            mass_pendulum=mass_pendulum,
            length_arm=length_arm,
            dist_center_of_mass_arm=dist_link_mass_arm,
            dist_center_of_mass_pendulum=dist_link_mass_pendulum,
            moment_of_inertia_com_arm=0.0,
            moment_of_inertia_com_pendulum=0.0,
            damping_arm=damping_arm,
            damping_pendulum=damping_pendulum,
            g=9.81,
        )
        next_state = euler(f=f, dt=self.sim_dt, n_steps=n_steps, x0=initial_state)
        # next_state = solve_ivp(fun=f, t_span=(0.0, self.sim_dt), y0=initial_state.squeeze()).y[:, -1] # fmt: skip

        dstate = next_state - initial_state

        return dstate


def compute_feature_normalizers(bm):
    from metalearning_benchmarks.util import collate_benchmark

    x, _ = collate_benchmark(bm)

    return (
        x.mean(axis=(0, 1)),
        x.std(axis=(0, 1)),
    )


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    n_task = 4
    n_datapoints_per_task = 128
    bm = FreePointMassFurutaBenchmark(
        n_task=n_task,
        n_datapoints_per_task=n_datapoints_per_task,
        output_noise=0.0,
        seed_task=1234,
        seed_x=2234,
        seed_noise=3234,
    )

    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, squeeze=False)
    t = np.arange(n_datapoints_per_task)
    for l in range(n_task):
        task = bm.get_task_by_index(l)
        for i, ax in enumerate(axes[:, 0]):
            ax.plot(t, task.x[:, i])
    plt.show()
