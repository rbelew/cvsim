''' cvsim2pyABC:  using pyABC sampler with covasim

following https://pyabc.readthedocs.io/en/latest/examples/parameter_inference.html
		  https://github.com/ICB-DCM/pyABC/blob/master/doc/examples/multiscale_agent_based.ipynb

Created on Nov 18, 2020

@author: rik
'''

import csv
import os
from string import capwords
import sys
import datetime
from time import time
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.stats as st
from scipy.integrate import solve_ivp

import pandas as pd
import matplotlib.pyplot as plt

import pyabc
from pyabc.sampler import ConcurrentFutureSampler, MulticoreEvalParallelSampler
from pyabc.visualization import *
from pyabc import MedianEpsilon, LocalTransition

import covasim as cv
import covasim.interventions as cvintrv
import covasim.misc as cvm
import covasim.data as cvdata

# %matplotlib inline

def distDiff(simulation, data):
	'''absolute difference between sim and data, weighting death 2x
	'''

	return 2 * np.absolute(data["new_deaths"] - simulation["new_deaths"]).sum() + \
			   np.absolute(data["n_infectious"] - simulation["n_infectious"]).sum()

def outbreak(x,verbose=False,allParam=False):
	'''https://github.com/JesperLM/SEIRD_Model_COVID-19
		https://raw.githubusercontent.com/JesperLM/SEIRD_Model_COVID-19/master/SEIRD_model.py
	'''
	
	ndays = NDays
	beta  = x['beta']
	n_infectious = x['n_infectious']
	
	def SEIRD_ODE(t, u):
		S, E, I, R, D = u
		# NB: elide social_distancing(t)
		return [-beta*S*I/(N-D), \
				 beta*S*I/(N-D) - alpha*E, \
				 alpha*E - gamma*I - mu*I, \
				 gamma*I, \
				 mu*I]

	incubation_time = 5	 		# Time before an individual is infected
	alpha = 1 / incubation_time # inverse of average incubation period
	time_sick = 15		  		# Time an individual is sick

	gamma = 1 / time_sick  		# mean recovery rate
	death_rate = 0.01	   		# death rate of disease
	mu = death_rate*gamma

	if verbose:
		print(f'\n* outbreak: beta: {beta} n_infectious:{n_infectious}')
		print(f'outbreak: R: {alpha/(alpha+mu)*beta/(mu+gamma)}')		
		start_time = time()

	# print('outbreak: TESTING popsize=1000')
	N = 1000000
	dt =   1 # one day = 24 h
	# NDays = 30				

	I0 = n_infectious   			# Amount of initally infected
	E0 = float(n_infectious)/2   	# Amount of initally exposed
	R0 = 0   					# Amount of initally recovered
	D0 = 0   					# Amount of initally dead
	S0 = N - E0 - I0 - R0 - D0

	SEIRD_0 = [S0, E0, I0 ,R0, D0]	

	N_t = int(ndays/dt)		# Corresponding no of time steps
	t_ivp = np.linspace(0, ndays, N_t+1)

	soln = solve_ivp(SEIRD_ODE, [0,ndays], SEIRD_0, t_eval=t_ivp)

	S = soln.y[0]
	E = soln.y[1]
	I = soln.y[2]
	R = soln.y[3]
	D = soln.y[4]

	if verbose:
		print(f'outbreak: after {NDays}: Death={D[-1]} Suseptible={S[-1]/N}')	
		print(f"ODE run took {time() - start_time:.2f}s")
	
	if allParam:
		allp = {'S':S, 'E':E, 'I':I, 'R':R, 'D': D}
		return allp

	return {"new_deaths": D, "n_infectious": I}

def plot_ODE(data,outf):
	'''ASSUME data is allParam from outbreak()
	'''
	
	k0 = 'D'
	ndays = len(data[k0])
	timeVec = np.linspace(0,ndays,ndays)

	f, ax = plt.subplots()
	ax.set_yscale('log')
	
	for k in ['S', 'E', 'I', 'R', 'D']:
		if k=='S':
			# NB: drop typically huge S to better show others
 			S0 = data[k][0]       
 			continue
		mrk = '.' if k in ['I','D'] else ','
		ax.plot(timeVec, data[k], marker=mrk, label=k)
	ax.legend()
	ax.set_title(f'ODE S0={S0:5.0e}')
	
	plt.savefig(outf)
	
def load_CVData(inf):

	reader = csv.DictReader(open(inf))
	diagVec = []
	deathVec = []
	prevDiag = 0
	prevDeath = 0
	for i,entry in enumerate(reader):
		# date,cum_diagnoses,cum_deaths
		cummDiag = int(entry['cum_diagnoses'])
		cummDeath = int(entry['cum_deaths'])
		diagVec.append(cummDiag-prevDiag)
		deathVec.append(cummDeath-prevDeath)
		prevDiag = cummDiag
		prevDeath = cummDeath
	
	return {'new_deaths': np.array(deathVec), 'n_infectious': np.array(diagVec) }

def load_TstData(inf):

	reader = csv.DictReader(open(inf))
	diagVec = []
	deathVec = []
	for i,entry in enumerate(reader):
		# T,Death,Diag
		diagVec.append(int(float(entry['Diag'])))
		deathVec.append(int(float(entry['Death'])))
	
	return {'new_deaths': np.array(deathVec), 'n_infectious': np.array(diagVec) }
NDays = 0

def plot_history(history,outf):
	fig, ax = plt.subplots()
	for t in range(history.max_t + 1):
		df, w = history.get_distribution(m=0, t=t)
		pyabc.visualization.plot_kde_1d(df, w, xmin=0, xmax=10,
									   x='theta', ax=ax,
									   label=f"t={t}")
	ax.legend()
	plt.savefig(outf)


def addNoise(pars,sigma):
	npars = {}
	for k in pars:
		nvec = pars[k] + sigma * np.random.randn(len(pars[k]))
		npars[k] = np.round(nvec)
		
	return npars

def plot_noise(data,noiseData,outf):
	
	k0 = data.keys()[0]
	ndays = len(data[k0])
	timeVec = np.linspace(0,ndays,ndays)

	f, (ax1, ax2) = plt.subplots(ncols=2)
	
	ax1.plot(timeVec, data['new_deaths'], color='r', label='Simulation')
	ax1.scatter(timeVec, noiseData['new_deaths'], marker='.', color='b', label='Noised')
	ax1.set_title('new_deaths')

	ax2.plot(timeVec, data['n_infectious'],color='r', label='Simulation')
	ax2.scatter(timeVec, noiseData['n_infectious'], marker='.', color='b', label='Noised')
	ax2.set_title('n_infectious')
	
	plt.savefig(outf)

def rptResult(result,outf,addDate=False):
	
	outs = open(outf,'w')
	# NB: labels to match covasim.Fit 
	hdr = 'T,new_deaths,n_infectious'
	if addDate:
		hdr += ',date'
	outs.write(hdr+'\n')
	
	startDate = datetime.date(2020,1,1)
	for i in range(NDays+1):
		line = f'{i},{result["new_deaths"][i]},{result["n_infectious"][i]}'
		if addDate:
			date = startDate + datetime.timedelta(days=i)
			line += f',{date.strftime("%y%m%d")}'
		outs.write(line+'\n')
	outs.close()

def addDateSeries(ndataDF):
	startDate = datetime.date(2020,1,1)
	dates = [f'{(startDate + datetime.timedelta(days=i)).strftime("%y%m%d")}' for i in range(NDays+1)]
	ndataDF['date'] = pd.Series(dates)
	return ndataDF

def runCovasimModel(pars,lbl,datafile):
	sim = cv.Sim(pars,label=lbl,datafile=datafile)
	sim.run()
	fit = sim.compute_fit()
	return fit.mismatches[:]
	
if __name__ == '__main__':
	
	lbl = 'cvABCTst'

	dataDir = '/System/Volumes/Data/rikData/coviData/'
	localDir = dataDir + 'pyABC/'
	
	db_path = "sqlite:///" + localDir +  f"{lbl}.db"
	
	tstRunParam = {'n_infectious':500,'beta': 0.07}
	NDays = 300
	
	maxABCPop = 10
	ncore = 4		
	
	# 210113: using testODE results as "data"
	result = outbreak(tstRunParam)
	noisedResult = addNoise(result,20)
	
	noiseFile = dataDir + 'testODE-noise.csv'
	rptResult(noisedResult,noiseFile,addDate=True)
	
	ndataDF = pd.DataFrame(noisedResult)
	ndataDatedDF = addDateSeries(ndataDF)	

	# Use default MulticoreEvalParallelSampler sampler
	# pool = ThreadPoolExecutor(max_workers=nworkers)
	# sampler = ConcurrentFutureSampler(pool)
	
	sampler = MulticoreEvalParallelSampler(ncore)
	
	bnds0 = dict(n_infectious= (10,1000),  beta=(1e-3, 1e-1))
	
	doPlots = True

	exptList = []
	exptIdx = 0
	print('main: TESTING one experiment 3_3')
	for maxPop in [3]: #[3,5,10,20,40]:
		for maxNR in [3]: # [3,5,10,20,40]:
			expt = {'maxABCPop': maxPop,'maxNRPop': maxNR, 'bounds': bnds0}
			expt['idx'] = exptIdx
			exptIdx += 1
			expt['name'] = f'{maxPop:02}_{maxNR:02}'
			exptList.append(expt)

	for ie,expt in enumerate(exptList):
		exptIdx = expt['idx']

		bounds = expt['bounds']
		maxABCPop = expt['maxABCPop']
		maxNRPop = expt['maxNRPop']

		# NB: pyabc.Distribution inherits from scipy.stats the convention of the two argument being
		#     lowerBound and RANGE (vs. lower bound, upper bound)
		iiRange = bounds['n_infectious'][1] - bounds['n_infectious'][0]
		betaRange = bounds['beta'][1] - bounds['beta'][0]
		prior = pyabc.Distribution(pop_infected=pyabc.RV("uniform", bounds['n_infectious'][0],  iiRange),
								   beta = pyabc.RV("uniform", bounds['beta'][0], betaRange))

		# modelRun = outbreak
		
		pars = dict(
		    pop_size = 50e3,
		    n_days = NDays,
		    verbose = 'brief',
		)

		# model = runCovasimModel(pars,lbl,datafile=ndataDF)
		
# 		import netabc.covasim
# 		netabc = netabc.covasim.CovaSimModel()
				
		abc = pyabc.ABCSMC(runCovasimModel, prior, distDiff,
					 population_size=maxABCPop,
					 sampler=sampler,
	# 				 transitions=LocalTransition(k_fraction=.3),
	# 				 eps=MedianEpsilon(500, median_multiplier=0.7)
					 )
		
		# new method returned an integer. This is the id of the ABC-SMC run. 
		history = abc.new(db_path, ndataDatedDF)
		runID = history.id
		
		print(f'main: runID={runID} exptIdx={exptIdx}: {expt}')
		
		start_time = time()
	
		history = abc.run(minimum_epsilon=.1, max_nr_populations=maxNRPop)
		
		print(f"main: runID={runID} exptIdx={exptIdx} ABC time {time() - start_time:.2f}s")

		if doPlots:
			
			plotDir = localDir + f'{lbl}_plots/'
			if not os.path.exists(plotDir):
				print(f'creating {plotDir}')
				os.mkdir(plotDir)
			
			fig, axes = plt.subplots(ncols=2)
			for t in range(history.max_t+1):
				particles = history.get_distribution(m=0, t=t)
				plot_kde_1d(*particles, "pop_infected",
							label=f"t={t}", ax=axes[0],
							xmin=bounds['n_infectious'][0], xmax=bounds['n_infectious'][1], numx=300)
				plot_kde_1d(*particles, "beta",
							label=f"t={t}", ax=axes[1],
							xmin=bounds['beta'][0], xmax=bounds['beta'][1], numx=300)	
	
			exptName = expt['name']
			plt.savefig(plotDir + f'{exptIdx:02}_{exptName}_kdeTime.png')
			print('kdeTime plotted')
	
			fig, ax = plt.subplots()
			df, w = history.get_distribution(m=0)
			plot_kde_matrix(df, w, limits=bounds)
			
			plt.savefig(plotDir + f'{exptIdx:02}_{exptName}_kdeMatrix.png')
			print('kdeMatrix plotted')
			
# 			histPlotFile = plotDir + f'{exptIdx:02}_{exptName}_history.png'
# 			plot_history(history, histPlotFile)
			
# 			fig, axes = plt.subplots(ncols=2)
# 			plot_sample_numbers(history, ax=axes[0])
# 			pyabc.visualization.plot_epsilons(history, ax=axes[1])
# 			pyabc.visualization.plot_credible_intervals(
# 				history, levels=[0.95, 0.9, 0.5], ts=[0, 1, 2, 3, 4],
# 				show_mean=True, show_kde_max_1d=True,
# 				refval={'mean': 2.5}, arr_ax=axes[2])
# 			pyabc.visualization.plot_effective_sample_sizes(history, ax=axes[3])
# 	
# 			plt.savefig(plotDir + f'{exptIdx:02}_{exptName}_sample.png')
# 			print('kdeMatrix plotted')


