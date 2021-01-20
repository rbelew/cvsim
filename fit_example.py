import covasim as cv
sim = cv.Sim(datafile='testODE-noise.csv')

sim.run()
sim.plot(to_plot=['new_deaths', 'cum_deaths', 'n_infectious'])

fit = sim.compute_fit(keys=['cum_deaths', 'n_infectious'], weights=dict(cum_deaths=1.0, n_infectious=0.5))
fit.plot()