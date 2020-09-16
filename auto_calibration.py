'''
Illustration of an automatic calibration to NY state data
'''

import numpy as np
import pylab as pl
import sciris as sc
import covasim as cv
import scipy as sp
import optuna as op

cv.check_version('1.6.1', die=False) # Ensure Covasim version is correct


class Calibration:

    def __init__(self, storage):

        # Settings
        self.pop_size = 100e3 # Number of agents
        self.start_day = '2020-02-01'
        self.end_day = '2020-07-30' # Change final day here
        self.state = 'NY'
        self.datafile = 'NY.csv'
        self.total_pop = 19453561 # Population of NY from census, nst-est2019-alldata.csv

        # Saving and running
        self.n_trials  = 20 # Number of sequential Optuna trials
        self.n_workers = 4 # Number of parallel Optuna threads -- incompatible with n_runs > 1
        self.n_runs    = 1 # Number of sims being averaged together in a single trial -- incompatible with n_workers > 1
        self.storage   = storage # Database location
        self.name      = 'covasim' # Optuna study name -- not important but required

        assert self.n_workers == 1 or self.n_runs == 1, f'Since daemons cannot spawn, you cannot parallelize both workers ({self.n_workers}) and sims per worker ({self.n_runs})'

        # Control verbosity
        self.to_plot = ['cum_infections', 'new_infections', 'cum_tests', 'new_tests', 'cum_diagnoses', 'new_diagnoses', 'cum_deaths', 'new_deaths']


    def create_sim(self, x, verbose=0):
        ''' Create the simulation from the parameters '''

        if isinstance(x, dict):
            pars, pkeys = self.get_bounds() # Get parameter guesses
            x = [x[k] for k in pkeys]

        # Define and load the data
        self.calibration_parameters = x

        # Convert parameters
        pop_infected = x[0]
        beta         = x[1]
        beta_day     = x[2]
        beta_change  = x[3]
        symp_test    = x[4]

        # Create parameters
        pars = dict(
            pop_size     = self.pop_size,
            pop_scale    = self.total_pop/self.pop_size,
            pop_infected = pop_infected,
            beta         = beta,
            start_day    = self.start_day,
            end_day      = self.end_day,
            rescale      = True,
            verbose      = verbose,
        )

        # Create the sim
        sim = cv.Sim(pars, datafile=self.datafile)

        # Add interventions
        interventions = [
            cv.change_beta(days=beta_day, changes=beta_change),
            cv.test_num(daily_tests=sim.data['new_tests'].dropna(), symp_test=symp_test),
            ]

        # Update
        sim.update_pars(interventions=interventions)

        self.sim = sim

        return sim


    def get_bounds(self):
        ''' Set parameter starting points and bounds -- NB, only lower and upper bounds used for fitting '''
        pdict = sc.objdict(
            pop_infected = dict(best=10000, lb=1000,  ub=50000),
            beta         = dict(best=0.015, lb=0.007, ub=0.020),
            beta_day     = dict(best=50,    lb=30,    ub=90),
            beta_change  = dict(best=0.5,   lb=0.2,   ub=0.9),
            symp_test    = dict(best=30,    lb=5,     ub=200),
        )

        # Convert from dicts to arrays
        pars = sc.objdict()
        for key in ['best', 'lb', 'ub']:
            pars[key] = np.array([v[key] for v in pdict.values()])

        return pars, pdict.keys()


    def smooth(self, y, sigma=3):
        ''' Optional smoothing if using daily death data '''
        return sp.ndimage.gaussian_filter1d(y, sigma=sigma)


    def run_msim(self, new_deaths=False):
        ''' Run the simulation -- if new_deaths, fit to daily deaths rather than cumulative cases + deaths '''
        if self.n_runs == 1:
            sim = self.sim
            sim.run()
        else:
            msim = cv.MultiSim(base_sim=self.sim)
            msim.run(n_runs=self.n_runs)
            sim = msim.reduce(output=True)
        if new_deaths:
            offset = cv.daydiff(sim['start_day'], sim.data['date'][0])
            d_data = self.smooth(sim.data['new_deaths'].values)
            d_sim  = self.smooth(sim.results['new_deaths'].values[offset:])
            minlen = min(len(d_data), len(d_sim))
            d_data = d_data[:minlen]
            d_sim = d_sim[:minlen]
            deaths = {'deaths':dict(data=d_data, sim=d_sim, weights=1)}
            sim.compute_fit(custom=deaths, keys=['cum_diagnoses', 'cum_deaths'], weights={'cum_diagnoses':0.2, 'cum_deaths':0.2}, output=False)
        else:
            sim.compute_fit(output=False)

        self.sim = sim
        self.mismatch = sim.results.fit.mismatch

        return sim


    def objective(self, x):
        ''' Define the objective function we are trying to minimize '''
        self.create_sim(x=x)
        self.run_msim()
        return self.mismatch


    def op_objective(self, trial):
        ''' Define the objective for Optuna '''
        pars, pkeys = self.get_bounds() # Get parameter guesses
        x = np.zeros(len(pkeys))
        for k,key in enumerate(pkeys):
            x[k] = trial.suggest_uniform(key, pars.lb[k], pars.ub[k])

        return self.objective(x)

    def worker(self):
        ''' Run a single Optuna worker '''
        study = op.load_study(storage=self.storage, study_name=self.name)
        return study.optimize(self.op_objective, n_trials=self.n_trials)


    def run_workers(self):
        ''' Run allworkers -- parallelized if each sim is not parallelized '''
        if self.n_workers == 1:
            self.worker()
        else:
            sc.parallelize(self.worker, self.n_workers)
        return


    def make_study(self):
        try: op.delete_study(storage=self.storage, study_name=self.name)
        except: pass
        return op.create_study(storage=self.storage, study_name=self.name)


    def load_study(self):
        return op.load_study(storage=self.storage, study_name=self.name)


    def get_best_pars(self, print_mismatch=True):
        ''' Get the outcomes of a calibration '''
        study = self.load_study()
        output = study.best_params
        if print_mismatch:
            print(f'Mismatch: {study.best_value}')
        return output


    def calibrate(self):
        ''' Perform the calibration '''
        self.make_study()
        self.run_workers()
        output = self.get_best_pars()
        return output


    def save(self):
        pars_calib = self.get_best_pars()
        sc.savejson(f'calibrated_parameters_{self.until}_{self.state}.json', pars_calib)


if __name__ == '__main__':

    recalibrate = True # Whether to run the calibration
    do_plot     = True # Whether to plot results
    storage = f'sqlite:///example_calibration.db' # Optuna database location
    verbose = 0.1 # How much detail to print

    cal = Calibration(storage)

    # Plot initial
    if do_plot:
        print('Running initial uncalibrated simulation...')
        pars, pkeys = cal.get_bounds() # Get parameter guesses
        sim = cal.create_sim(pars.best, verbose=verbose)
        sim.run()
        sim.plot(to_plot=cal.to_plot)
        pl.gcf().suptitle('Initial parameter values')
        cal.objective(pars.best)
        pl.pause(1.0) # Ensure it has time to render

    # Calibrate
    if recalibrate:
        print(f'Starting calibration for {cal.state}...')
        T = sc.tic()
        pars_calib = cal.calibrate()
        sc.toc(T)
    else:
        pars_calib = cal.get_best_pars()

    # Plot result
    if do_plot:
        print('Plotting result...')
        x = [pars_calib[k] for k in pkeys]
        sim = cal.create_sim(x, verbose=verbose)
        sim.run()
        sim.plot(to_plot=cal.to_plot)
        pl.gcf().suptitle('Calibrated parameter values')




print('Done.')
