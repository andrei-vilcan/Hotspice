from hotspice.active_inference import ActiveInferenceSimulation


setup = dict(
    env='tracking',
    env_params=dict(
        V = 3.0,
        alpha=0.3,
        x_0=0.0,
        dx_0=0.0
    ),
    n=3
)


sim = ActiveInferenceSimulation(**setup)
sim.run_simulation()