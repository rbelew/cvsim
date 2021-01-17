import pyabc
import tempfile
import os
import matplotlib.pyplot as plt

import sciris as sc
import covasim as cv


default_pars = dict(
    start_day = '2020-02-01',
    end_day   = '2020-04-11',
    beta      = 0.015,
    verbose = 0,
)


def model(parameters):
    print('*', end=None)
    pars = sc.dcp(default_pars)
    pars.update(parameters)
    sim = cv.Sim(pars=pars, datafile='example_data.csv', interventions=cv.test_num(daily_tests='data'))
    sim.run()
    fit = sim.compute_fit()
    return {"mismatch": fit.mismatch}

prior = pyabc.Distribution(rel_death_prob=pyabc.RV("uniform", 0, 3))


def distance(x, y):
    return x["mismatch"]

abc = pyabc.ABCSMC(model, prior, distance)

db_path = ("sqlite:///" + os.path.join(tempfile.gettempdir(), "test.db"))
observation = 0 # No observed mismatch!
abc.new(db_path, {"mismatch": observation})


#%% Run
history = abc.run(minimum_epsilon=.1, max_nr_populations=10)


#%% Plotting
fig, ax = plt.subplots()
for t in range(history.max_t+1):
    df, w = history.get_distribution(m=0, t=t)
    pyabc.visualization.plot_kde_1d(
        df, w,
        xmin=0, xmax=3,
        x="rel_death_prob", ax=ax,
        label="PDF t={}".format(t))
ax.axvline(observation, color="k", linestyle="dashed");
ax.legend()

_, arr_ax = plt.subplots(2, 2)

pyabc.visualization.plot_sample_numbers(history, ax=arr_ax[0][0])
pyabc.visualization.plot_epsilons(history, ax=arr_ax[0][1])
pyabc.visualization.plot_credible_intervals(
    history, levels=[0.95, 0.9, 0.5], ts=[0, 1, 2, 3, 4],
    show_mean=True, show_kde_max_1d=True,
    refval={'rel_death_prob': 2.0}, arr_ax=arr_ax[1][0])
pyabc.visualization.plot_effective_sample_sizes(history, ax=arr_ax[1][1])

plt.gcf().set_size_inches((12, 8))
plt.gcf().tight_layout()