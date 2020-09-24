''' cmaesTst: test frame for CMAES optimization of covasim

Created on Aug 16, 2020

@author: rik
'''

import math
from multiprocessing import Pool, cpu_count
import socket

import numpy as np
import optuna as op
import sciris as sc
import covasim as cv

import cmaes
import cma

from optuna.samplers import CmaEsSampler

class Calibration:

	def __init__(self, storage):

		# Settings
		self.pop_size = 100e3 # Number of agents
		self.start_day = '2020-02-01'
		self.end_day = '2020-07-30' # Change final day here
		self.state = 'NY'
		self.datafile = dataDir + 'data/NY.csv'
		self.total_pop = 19453561 # Population of NY from census, nst-est2019-alldata.csv

		# Saving and running
		self.n_trials  = 5 # Number of sequential Optuna trials
		self.n_workers = 4 # Number of parallel Optuna threads -- incompatible with n_runs > 1
		self.n_runs	= 1 # Number of sims being averaged together in a single trial -- incompatible with n_workers > 1
		self.storage   = storage # Database location
		self.name	  = 'cmaesTst' # Optuna study name -- not important but required

		assert self.n_workers == 1 or self.n_runs == 1, f'Since daemons cannot spawn, you cannot parallelize both workers ({self.n_workers}) and sims per worker ({self.n_runs})'

		# Control verbosity
		self.to_plot = ['cum_infections', 'new_infections', 'cum_tests', 'new_tests', 'cum_diagnoses', 'new_diagnoses', 'cum_deaths', 'new_deaths']


	def create_sim(self, x, verbose=0):
		''' Create the simulation from the parameters '''

		if isinstance(x, dict):
			pars, pkeys = self.get_bounds() # Get parameter guesses
			x = [x[k] for k in pkeys]

		self.calibration_parameters = x

		pop_infected = math.exp(x[0])
		beta		 = math.exp(x[1])
		
		# NB: optimization over only two parameters
# 		beta_day	 = x[2]
# 		beta_change  = x[3]
# 		symp_test	= x[4]
		beta_day	 = 50
		beta_change  = 0.5
		symp_test	= 30


		# Create parameters
		pars = dict(
			pop_size	 = self.pop_size,
			pop_scale	= self.total_pop/self.pop_size,
			pop_infected = pop_infected,
			beta		 = beta,
			start_day	= self.start_day,
			end_day	  = self.end_day,
			rescale	  = True,
			verbose	  = verbose,
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
			# 200920: ASSUME log scaling
			pop_infected = dict(best=math.log(10000), lb=math.log(1000),  ub=math.log(50000)),
			beta		 = dict(best=math.log(0.015), lb=math.log(0.007), ub=math.log(0.020)),
			# NB: optimization over only two parameters
# 			beta_day	 = dict(best=50,	lb=30,	ub=90),
# 			beta_change  = dict(best=0.5,   lb=0.2,   ub=0.9),
# 			symp_test	= dict(best=30,	lb=5,	 ub=200),
		)

		# Convert from dicts to arrays
		pars = sc.objdict()
		for key in ['best', 'lb', 'ub']:
			pars[key] = np.array([v[key] for v in pdict.values()])

		return pars, pdict.keys()


	def smooth(self, y, sigma=3):
		''' Optional smoothing if using daily death data '''
		return sp.ndimage.gaussian_filter1d(y, sigma=sigma)


	def objective(self, x):
		''' Define the objective function we are trying to minimize '''
		self.create_sim(x)
		
		sim = self.sim
		sim.run()
		
		tranTree = sim.compute_fit()

		self.sim = sim
		self.mismatch = tranTree.mismatch
		
		# print(f'objective: {x} {self.mismatch}')
		
		return self.mismatch

def objectiveQuad(trial):
	x = trial[0]
	y = trial[1]
	return x ** 2 + y



if __name__ == '__main__':
	
	global dataDir
	hostname = socket.gethostname()
	if hostname == 'hancock':
		dataDir = '/System/Volumes/Data/rikData/coviData/'
	elif hostname == 'mjq':
		dataDir = '/home/Data/covid/'

	
	storage	 = f'sqlite:///{dataDir}db/cmaesTst.db'
	cal = Calibration(storage)
	pars, pkeys = cal.get_bounds() # Get parameter guesses
	
	# CMA
	
	cmaesPopSize = 20

	print('# Gen,I,value,infect,beta')

	es = cma.CMAEvolutionStrategy(
            x0=np.array(pars['best']),
            
            sigma0=0.1, # 	(pars['ub']-pars['lb'])/4
			
			# Argument bounds can be None or bounds[0] and bounds[1] are lower and
			# upper domain boundaries, each is either None or a scalar or a list
			# or array of appropriate size.
            
            inopts={ "popsize": cmaesPopSize, \
				'CMA_elitist': True,
				# 'bounds': [np.array(pars['lb']), np.array(pars['ub']) ] 
				} 
        )	

	for gen in range(50):
		solutions = []
		# values = []
		currPop = es.ask()
					
		# NB: fixed number=4 processes
		with Pool(4) as pool:
			
			values = pool.map(cal.objective, currPop)
			
			for i,x in enumerate(currPop):
				infect = math.exp(x[0])
				beta = math.exp(x[1])
				# NB:  print full precision, for sampling database
				print(f"# {gen},{i},{values[i]:10.6e},{infect:10.6e},{beta:10.6e}")
		
			es.tell(currPop, values)
			es.disp()			
			
	es.result_pretty()
	
