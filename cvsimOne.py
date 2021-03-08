''' cvsimOne:  run covasim on ONE country
Created on Aug 11, 2020

@version: 0.3
Feb 15 2021

@author: rbelew@ucsd.edu
'''

from collections import defaultdict
import csv
from datetime import date
import datetime
from time import time
import glob
from itertools import repeat
import math
import pickle
import os
import socket
import sys

import numpy as np

import matplotlib 
matplotlib.use('Agg')

import pylab as pl

# from sklearn import metrics as skmetric

import sqlite3 as sqlite

import covasim as cv
import covasim.interventions as cvintrv
import covasim.misc as cvm
import covasim.data as cvdata

# from covasim.data import country_age_data as cad

import boto3

import sciris as sc

from multiprocessing import Pool, cpu_count

import pyabc
from pyabc.populationstrategy import AdaptivePopulationSize
from pyabc.sampler import MulticoreEvalParallelSampler, MulticoreParticleParallelSampler
from pyabc.visualization import *
from pyabc.transition import Transition, MultivariateNormalTransition

import cma
import multiprocessing.pool as mpp
from covasim.utils import false

def basicStats(l):
	"Returns avg and stdev"
	if len(l) == 0:
		return(0.,0.)

	sum = 0
	for n in l:
		sum += n
	avg = float(sum) / len(l)

	sumDiffSq = 0.
	for n in l:
		sumDiffSq += (n-avg)*(n-avg)

	stdev = math.sqrt(sumDiffSq) / float(len(l))
	return (avg,stdev)

CurrCtnyParams = {}

DecadeKeys = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
TypicalTestRate = 3.3e-4 # 0.01

ModelStartDate = '2020-01-21'
ModelEndDate =  '2020-07-31'
NormCountry = {"Cote dIvoire": 'Côte d’Ivoire',
				'Timor Leste': 'Timor-Leste'}

def initDB(currDB):
	curs = currDB.cursor()
	cmd = '''CREATE TABLE trial (
					idx	INTEGER,
					gen	INTEGER,
					indiv	INTEGER,
					value	REAL,
					infect	INTEGER,
					beta	REAL,
					testrate	REAL,
					cumm_diagnoses REAL,
					cumm_deaths	REAL,
					cumm_tests	REAL,
					cum_diagnoses_mismatch	REAL,
					cum_deaths_mismatch	REAL,
					cum_tests_mismatch	REAL,
					PRIMARY KEY(idx) ) '''
	curs.execute(cmd)

	cmd2 = '''CREATE TABLE generation (
					idx	INTEGER,
					gen	INTEGER,
					valavg	REAL,
					valsd	REAL,
					infectavg	REAL,
					infectsd	REAL,
					betaavg	REAL,
					betasd	REAL,
					testrateavg	REAL,
					testratesd	REAL,

					PRIMARY KEY(idx) ) '''
	curs.execute(cmd2)

	return currDB

def loadECDCGeog(inf):
	
	ecdcCountry = defaultdict(lambda: defaultdict(list)) # cname -> {:cname :iso3 :continent :pop19 }
	
	nmissPop = 0
	reader = csv.DictReader(open(inf))
	for i,entry in enumerate(reader):
		# dateRep,day,month,year,cases,deaths,countriesAndTerritories,geoId,countryterritoryCode,popData2019,continentExp,Cumulative_number_for_14_days_of_COVID-19_cases_per_100000
		
		cname = entry['countriesAndTerritories'].strip()
		iso3 = entry['countryterritoryCode'].strip()
		if cname not in ecdcCountry:
			pop19 = entry['popData2019'].strip()
			if pop19 == '':
				print('loadECDC: No pop19? %s %s' % (iso3,entry['countriesAndTerritories']))
				nmissPop += 1
				pop19 = 0
			else:
				pop19 = int(pop19)
			ecdcCountry[cname] = {'cname': cname,
								 'iso3': iso3,
								 'continent': entry['continentExp'].strip(),
								 'pop19': pop19}
	print('loadECDC: NCountry=%d NMissPop=%d' % (len(ecdcCountry),nmissPop))
	return ecdcCountry

def loadGeogInfo(inf):
	
	countryInfo = loadECDCGeog(inf) # # cname -> {:cname :iso3 :continent :pop19 }
	# NB: TWN and HKG missing from ECDC ?!
	countryInfo['Taiwan'] = {'cname': 'Taiwan', 'iso3': 'TWN', 'continent': 'Asia',
							 'pop19': 23773876 } # https://www.worldometers.info/world-population/taiwan-population/
	countryInfo['Hong_Kong'] = {'cname': 'Hong_Kong', 'iso3': 'HKG', 'continent': 'Asia',
							 'pop19': 7436154 } # https://www.worldometers.info/world-population/taiwan-population/	
	cname2iso = {cname: countryInfo[cname]['iso3'] for cname in countryInfo.keys() }
	iso2cname = {countryInfo[cname]['iso3']: cname for cname in countryInfo.keys() }	

	return countryInfo,cname2iso,iso2cname

def loadConsensusEducIntrvn(inf):
	'''load consensus educational closures from consensusIntrvn.csv
	built by util.bldConsensusEducIntrvn()
	'''
	minDuration = 14
	reader = csv.DictReader(open(inf))
	interveneTbl = defaultdict(list) # iso3 -> [(sdate,edate), ...]
	for i,entry in enumerate(reader):
		# ISO3,SDate,EDate,NDays
		
		ndays = int(entry['NDays'])
		if ndays < minDuration:
			continue
		
		iso3 = entry['ISO3']
		sdate = datetime.datetime.strptime(entry['SDate'],'%y/%m/%d')
		edate = datetime.datetime.strptime(entry['EDate'],'%y/%m/%d')
		
		interveneTbl[iso3].append( (sdate,edate) )
		
	return interveneTbl
	
def loadFinalTestRate(inf):	
	reader = csv.DictReader(open(inf))
	testRate = {} # country -> testRate
	
	for i,entry in enumerate(reader):
		# Country,UseTest,testPerDiag,testPerDeath,indivPerDiag,indivPerDeath,lastNDiag,lastNDeath,diagEst,deathEst,Pop19,NTest,TestRate
		country = entry['Country']
		trate = float(entry['TestRate'])
		testRate[country] = trate
		
	return testRate

# 201110: refer to  GLOBAL var ala jpg_CalibUK
# def create_sim(x):
def create_sim(x,currCtyPar):

	# 201110: refer to  GLOBAL var ala jpg_CalibUK
	# if LogTransform:
	if currCtyPar['LogTransform']:
		pop_infected = math.exp(x[0])
		beta		 = math.exp(x[1])
	else:
		pop_infected = x[0]
		beta		 = x[1]

	# cliffckerr, https://github.com/InstituteforDiseaseModeling/covasim/issues/269
	# The data file and location are properties of the simulation, not model
	# parameters, so they should only be supplied as arguments to Sim() and
	# should not be present in the parameters object. Storage only pertains
	# to Optuna, and should also not be part of the pars object.

	pars = sc.objdict(
		pop_size	 = currCtyPar['pop_size'],
		pop_scale	= currCtyPar['pop_scale'],
		pop_type	 = currCtyPar['pop_type'],
		start_day	= currCtyPar['start_day'],
		end_day	  = currCtyPar['end_day'],
		asymp_factor = currCtyPar['asymp_factor'],
		# contacts	 = currCtyPar['contacts'],
		rescale	  = currCtyPar['rescale'],
		verbose	  = currCtyPar['verbose'],
		interventions = currCtyPar['interventions'],
		# 

		beta		 = beta,
		pop_infected = pop_infected,
	)

	# Create the baseline simulation
	sim = cv.Sim(pars=pars,datafile=currCtyPar['datafile'],location=currCtyPar['location'], \
				 datacols=currCtyPar['datacols'] )
		# age_dist = currCtyPar['age_dist'],

	
# 	beta_layer  = dict(h=3.0, w=0.6, c=0.3, e=0.6)		
# 	dynam_layer = dict(h=0,   w=0,   c=0, e=0)
# 	iso_factor  = dict(h=0.3, w=0.1, c=0.1, e=0.1)
# 	quar_factor = dict(h=0.6, w=0.2, c=0.2, e=0.2)
# 		
# 	sim['beta_layer'] = beta_layer
# 	sim['dynam_layer'] = dynam_layer
# 	sim['iso_factor'] = iso_factor
# 	sim['quar_factor'] = quar_factor
	
	# from JPG's Calibration_UK
	# what do they do?!
# 	sim['prognoses']['sus_ORs'][0] = 1.0 # ages 0-10
# 	sim['prognoses']['sus_ORs'][1] = 1.0 # ages 10-20


	# NB: UseTestRate==constant test interventions already added in _main
	if currCtyPar['UseTestRate']=='data':
		tn = cvintrv.test_num(daily_tests=sim.data['new_tests'])
		tn.do_plot = False
		pars['interventions'].append(tn)
		sim.update_pars(interventions=pars['interventions'])
		
	elif currCtyPar['UseTestRate']=='srch':
		# NB: no LOG transform on test_rate
		test_rate = x[2]

		testIntrvn = cvintrv.test_prob(symp_prob=test_rate, asymp_prob=test_rate)
		testIntrvn.do_plot = False
		
		pars['interventions'].append(testIntrvn)		
		sim.update_pars(interventions=pars['interventions'])

	return sim

def objective(x,currCtyPar):
	''' Define the objective function we are trying to minimize '''

	# Create and run the sim
	# 201110: refer to  GLOBAL var ala jpg_CalibUK
	# sim = create_sim(x)
	sim = create_sim(x,currCtyPar)
	
	sim.run()
	summary2get = ['cum_diagnoses', 'cum_deaths', 'cum_tests']
	results = {}
	for k in summary2get:
		results[k] = sim.summary[k]
	fit = sim.compute_fit()
	for k,v in fit.mismatches.items():
		rk = f'{k}_mismatch'
		results[rk] = v
	results['fit'] = fit.mismatch
	return results
	
def get_bounds():
	''' Set parameter starting points and bounds 
		NB: uses ORDERED dictionary => specifiy bounds IN ORDER!'''

	pdict = sc.objdict()

	pdict['pop_infected'] = dict(best=20000,lb=1000,   ub=50000)
	pdict['beta']         = dict(best=.005, lb=0.001, ub=0.01)
	
	# X[2] = test_rate
	if UseTestRate=='srch':
		pdict['test_rate'] = dict(best=3.3e-4,  lb=2e-6,   ub=3e-3)
		
	if LogTransform:
		# NB: only ['pop_infected','beta'] LOG transformed
		for param in ParamLogTransformed:
			
			for key in ['best', 'lb', 'ub']:
				pdict[param][key] = math.log(pdict[param][key])

	# Convert from dicts to arrays
	pars = sc.objdict()
	for key in ['best', 'lb', 'ub']:
		pars[key] = np.array([v[key] for v in pdict.values()])

	return pars, pdict.keys()

def calibrate_CMA(currDB):
	'''calibrate parameters using CMA library
	'''
	
	# print('NB: TESTING only 3 generations!')
	# ngen =  3
	ngen =  25

	# print('NB: TESTING only 1 trial/worker !')
	# cmaesPopSize = n_workers
	cmaesPopSize = n_trials * n_workers
    
	trialAttribNames = 'gen,indiv,value,infect,beta,testrate,' + \
						'cumm_diagnoses,cumm_deaths,cumm_tests,' + \
						'cum_diagnoses_mismatch,cum_deaths_mismatch,cum_tests_mismatch'
						
	generationAttribNames = 'gen,valavg,valsd,infectavg,infectsd,betaavg,betasd,testrateavg,testratesd'

	pars, pkeys = get_bounds() # Get parameter guesses

	print(f'calibrate2: cmaesPopSize={cmaesPopSize} ngen={ngen}')
	# print(f'calibrate2: SchoolClose={SchoolClose} UseTestRate={UseTestRate} LogTransform={LogTransform}')
	print(f'calibrate2:  pars=\n{pars}')
	esopts = { "popsize": cmaesPopSize, \
				# 'CMA_elitist': True,

				# Argument bounds can be None or bounds[0] and bounds[1] are lower and
				# upper domain boundaries, each is either None or a scalar or a list
				# or array of appropriate size.	            
				'bounds': [np.array(pars['lb']), np.array(pars['ub']) ],
				# 'bounds': [ [1.e+03, 1.e-03],[5.e+04, 8.e-03] ],
				
				'verbose': 1,
				} 
	
	if not LogTransform:
		
# 		if 'test_rate' in pkeys: 
# 			esopts['CMA_stds'] = [1e4,0.1,1.]
# 		else:
# 			esopts['CMA_stds'] = [1e4,0.1]

		# cf. cma.fmin()
		qtrRange = (pars['ub'] - pars['lb'])/4
		esopts['CMA_stds'] = qtrRange
		
	es = cma.CMAEvolutionStrategy(
			sigma0 = 1e-1,
            x0=np.array(pars['best']),
            inopts=esopts
        )	

	if RptCMAGen:
		hdr = '# Gen,I,' + ','.join(pkeys)
		print(hdr)
	
	initDB(currDB)
	cursor = currDB.cursor()

	for gen in range(ngen):
		solutions = []
		# values = []
		currPop = es.ask()
					
		# NB: fixed number=4 processes
		with Pool(n_workers) as pool:
			
			zipObj = zip(currPop, repeat(CurrCtnyParams))
			argList = list(zipObj)
			
			############
			try: 
				allResults = pool.starmap(objective,argList)
			except Exception as e:
				print(f'calibrate_CMA: gen={gen} {e}')
				break
				# import pdb; pdb.set_trace
			############
			
			valuesOnly = []
			
			for i,x in enumerate(currPop):
				if LogTransform:
					infect = math.exp(x[0])
					beta = math.exp(x[1])
				else:
					infect = x[0]
					beta = x[1]

				# NB: testrate NOT subject to LOG transform
				if 'test_rate' in pkeys: 
					test_rate = x[2]
				else:
					test_rate = 0.
	
				result = allResults[i]
				
				value = result['fit']
				valuesOnly.append(value)
				
				valList = [ gen,i,value,infect,beta,test_rate ]
				
				for rk in ['cum_diagnoses','cum_deaths','cum_tests']:
					valList.append(result[rk])
					
				for rk in ['cum_diagnoses_mismatch','cum_deaths_mismatch']:
					valList.append(result[rk])
				if UseTestRate=='data': 
					valList.append(result['cum_tests_mismatch'])
				else:
					valList.append(0.0)
			
				qms = ','.join(len(valList)*'?')
				sql = 'insert into trial (' + trialAttribNames + ') values (%s)' % (qms)
				cursor.execute(sql,tuple(valList))

				if RptCMAGen:
					# NB:  print full precision, for sampling database
					line = f"# {gen},{i},{value:.60g},{infect:.60g},{beta:.60g}"
					if 'test_rate' in pkeys:
						line += f',{test_rate:.60g}'	
					print(line)
				
			# eo-Pool context

		currDB.commit() # commit trials
				
		es.tell(currPop, valuesOnly)
		es.disp()
		stats = {'avg':{}, "std": {}}
		
		stats['avg']['values'] = np.mean(valuesOnly)
		stats['std']['values'] = np.std(valuesOnly)
		
		for i,k in enumerate(pkeys):
			vals = []
			for indiv in currPop:
				if LogTransform and k in ParamLogTransformed:
					vals.append( math.exp(indiv[i]) )
				else:
					vals.append( indiv[i] )							
			stats['avg'][k] = np.mean( vals )
			stats['std'][k] = np.std(  vals )
			
		valList2 = [gen] 
		for k in ['values'] + pkeys:
			print(f"> {gen} {k} {stats['avg'][k]:.6e} {stats['std'][k]:.6e}")
			valList2 += [ stats['avg'][k], stats['std'][k] ]
			
		# NB: generation table has testrate avg, sd in any case
		if 'test_rate' not in pkeys:	
			valList2 += [0.,0.]
		
		qms = ','.join(len(valList2)*'?')
		sql = 'insert into generation (' + generationAttribNames + ') values (%s)' % (qms)
		cursor.execute(sql,tuple(valList2))

		currDB.commit() # commit generation								
	
	es.result_pretty()
	bestX = es.best.x
	
	# NB: return results in same form as get_bounds(), as expected by create_sim()
	# NB: it will do exp() on these values if LogTransform!
	
	return bestX

def ppFit(fit,outf):
	'''get details on fit
	'''
	fitVars = ['cum_deaths','cum_diagnoses']
	fitComponents = ['diffs','gofs','losses','mismatches']
	
	outs = open(outf,'w')
	
	vlen = None
	for compon in fitComponents:
		fcompon = fit.__getattribute__(compon)
		
		for fvar in fitVars:
			vec = fcompon[fvar]
			try:
				currVLen = len(vec)
			except:
			# if compon=='mismatches':
				lineList = [compon,fvar,str(vec)]
				line = ','.join(lineList)
				outs.write(line+'\n')
				continue
			if vlen==None:
				vlen = currVLen
				# print(f'vlen={vlen}')
				hdrList = ['compon','fvar'] + [f't_{i}' for i in range(vlen)]
				hdr = ','.join(hdrList)
				outs.write(hdr+'\n')
			if currVLen != vlen:
				print(f'bad vlen?! {compon} {fvar}')
				
			lineList = [compon,fvar] 
			for v in vec:
				lineList.append(str(v))
			line = ','.join(lineList)
			outs.write(line+'\n')
	
	line = f'mismatch, ,{fit.mismatch}'
	outs.write(line+'\n')
	outs.close()
	
def cvModel(simDict):
	'''construct Sim using same create_sim() shared with calibrate_CMA
	'''
	
	ppSimDict = {f'{k}': f'{v:.4e}' for k,v in simDict.items()}

	x = [ simDict['pop_infected'], simDict['beta'] ]
	if UseTestRate=='srch':
		x.append(simDict['test_rate'])
		
	if CurrCtnyParams['LogTransform']:
		pop_infected = math.exp(x[0])
		beta		 = math.exp(x[1])
		if UseTestRate=='srch':
			testRate = math.exp(x[2])
	else:
		pop_infected = x[0]
		beta		 = x[1]
		if UseTestRate=='srch':
			testRate = x[2]

	sim = create_sim(x,CurrCtnyParams)
	
	startTime = datetime.datetime.now()
	# print(f'cvModel: start: "{startTime}","{ppSimDict}","{x}"')
	
	sim.run()
	
	global infectStats
	cummInfect = sim.results['cum_infections']
	infectStats.append(cummInfect)
	
	fit = sim.compute_fit(keys=['cum_deaths', 'cum_diagnoses'], 
						weights=dict(cum_deaths=1.0, pop_infected=0.5))
	
	endTime = datetime.datetime.now()
	elapTime = endTime - startTime
	print(f'cvModel: "{startTime}",{elapTime.seconds} sec,"{ppSimDict}",{fit.mismatch:e}')
	return {"mismatch": fit.mismatch}

def getABC_FinalPopStats(dbfile):
	'''Use parameter mean of final generation
	'''
	
	conn = sqlite.connect(dbfile)
	curs = conn.cursor()
	paramVec = defaultdict(list)
	
	# popIdx = max(pop.time)
	sql = 'select id from populations order by t desc'
	curs.execute(sql)
	popIdx = curs.fetchone()[0]
	
	# modelIdx = model with population_id = popIdx
	valList = [popIdx]
	sql = 'select id from models where population_id=?'
	curs.execute(sql,tuple(valList))
	modelIdx = curs.fetchone()[0]
	
	# particleList with model_id = modelIdx
	valList = [modelIdx]
	sql = 'select id from particles where model_id=?'
	curs.execute(sql,tuple(valList))
	allParticles = curs.fetchall()
	for particle in allParticles:
		partIdx = particle[0]
	
		# parameter name, value for parameters with particle_id in particleList
		valList2 = [partIdx]
		sql = 'select name,value from parameters where particle_id=?'
		curs.execute(sql,tuple(valList2))
		allParam = curs.fetchall()
		for name,value in allParam:
			paramVec[name].append(value)
	
	stats = {}
	for k in paramVec:
		info = {}
		avg,sd = basicStats(paramVec[k])
		info['avg'] = avg
		info['sd'] = sd
		stats[k] = info
	return stats

def getABC_maxKDE(history,confidence=0.95):
	'''after pyABC.visualization.credible.plot_credible_intervals()
	'''
	
	n_run = 1
	levels = [confidence]
	df, w = history.get_distribution(0)
	par_names = list(df.columns.values)
	n_par = len(par_names)
	n_confidence = len(levels)
    
	# prepare matrices
	median = np.empty((n_par, n_run))
	kde_max = np.empty((n_par, n_run))
	kde_max_1d = np.empty((n_par, n_run))
	kde = MultivariateNormalTransition()
	kde_1d = MultivariateNormalTransition()

	# fill matrices
	# normalize weights to be sure
	w /= w.sum()
	# fit kde
	_kde_max_pnt = pyabc.visualization.credible.compute_kde_max(kde, df, w)
	# iterate over parameters
	paramKDE = {}
	for i_par, par in enumerate(par_names):
		info = {'i': i_par}
		# as numpy array
		vals = np.array(df[par])
		# median
		median[i_par] = pyabc.visualization.credible.compute_quantile(vals, w, 0.5)
		info['median'] = median[i_par]
		# kde max
		kde_max[i_par] = _kde_max_pnt[par]
		info['kde_max'] = median[i_par]
		
		_kde_max_1d_pnt = pyabc.visualization.credible.compute_kde_max(kde_1d, df[[par]], w)
		kde_max_1d[i_par] = _kde_max_1d_pnt[par]
		info['kde_max_1d'] = kde_max_1d[i_par]

		lb, ub = pyabc.visualization.credible.compute_credible_interval(vals, w, confidence)
		info['ci_lb'] = lb
		info['ci_ub'] = ub
		paramKDE[par] = info
		
	return paramKDE

def calibrate_ABC(abcDB):
	'''calibrate parameters using pyABC library
	'''

	# print('TESTING: ncore=4 for hancock')
	ncore = n_workers # 4
	# sampler = MulticoreParticleParallelSampler(ncore) # DYN sampling strategy
	# abcSampler = 'part//'
	sampler = MulticoreEvalParallelSampler(ncore,check_max_eval=True) # STAT sampling strategy
	abcSampler = 'eval//'
	maxABCPop = ncore
	maxNGen = 50
	maxABCWallTime = datetime.timedelta(minutes=10)
	MinABCAccept = 0.2 
	MinABCEps = 100
	# print('TESTING: minEps=200')
	# MinABCEps = 100
	# NB: ASSUME > MinABCAccept are accepted each gen
	# print('TESTING: limit abc evals=1000')
	# maxABCEval = 1000 # maxNGen * ncore / MinABCAccept
	maxABCEval = maxNGen * ncore / MinABCAccept / 2
	
	print(f'calibrate_ABC: {abcSampler} maxNGen={maxNGen} maxABCEval={maxABCEval}  maxABCWallTime={maxABCWallTime} MinABCAccept={MinABCAccept} MinABCEps={MinABCEps}')
	bounds, pkeys = get_bounds()
	print(f'calibrate_ABC:  bounds={bounds}')

	# NB: pyabc.Distribution inherits from scipy.stats the convention of the two argument being
	#	 lowerBound and RANGE (vs. lower bound, upper bound)
	iiRange = bounds['ub'][0] - bounds['lb'][0]
	betaRange = bounds['ub'][1] - bounds['lb'][1]
	if UseTestRate=='srch':
		testRateRange = bounds['ub'][1] - bounds['lb'][2]
		
	cvPrior =  pyabc.Distribution(
			pop_infected=     pyabc.RV("uniform", bounds['lb'][0], iiRange),
			beta=             pyabc.RV("uniform", bounds['lb'][1], betaRange))
	
	# NB: treat cvPrior just like a dict even tho its a ParameterStructure
	if UseTestRate=='srch':
		cvPrior['test_rate']= pyabc.RV("uniform", bounds['lb'][2], testRateRange)

	abc = pyabc.ABCSMC(cvModel, cvPrior, None, # 210222: Use default PNorm()
					# population_size=AdaptivePopulationSize(ncore, 0.15,max_population_size=4*ncore)
					population_size=maxABCPop,
					sampler=sampler)
	
	# No observed mismatch!
	history = abc.new(abcDB, {"mismatch": 0})
	
	runID = history.id
	
	print(f'calibrate_ABC: runID={runID}')

	start_time = time()

	history = abc.run(min_acceptance_rate=MinABCAccept, minimum_epsilon=MinABCEps, 
					max_total_nr_simulations=maxABCEval, max_walltime=maxABCWallTime,
					max_nr_populations=50)
	
	print(f"calibrate_ABC: runID={runID} ABC time {time() - start_time:.2f}s")
		
	return history

def plot_history(history,bounds,outf):
	
	fig = None
	for t in range(history.max_t+1):
		df,w = history.get_distribution(m=0, t=t)
		# HACK: need particular time's dataframe to know columns!
		if fig==None:
			paramList = list(df.columns)		
			fig, axes = pl.subplots(ncols=len(paramList))

		for i,param in enumerate(paramList):
			xmin = bounds['lb'][i]
			xmax = bounds['ub'][i]

			plot_kde_1d(df, w, x=param,label=f"t={t}", numx=300, xmin=xmin, xmax=xmax, ax=axes[i])
			axes[i].legend()

	pl.savefig(outf)

def plot_credible(history,outf):
	
	fig = None
	for t in range(history.max_t+1):
		df,w = history.get_distribution(m=0, t=t)
		# HACK: need particular time's dataframe to know columns!
		if fig==None:
			paramList = list(df.columns)		
			fig, axes = pl.subplots(ncols=len(paramList))

		for i,param in enumerate(paramList):

			plot_credible_intervals(history, levels=[0.95],show_kde_max=True, par_names = [param], arr_ax=axes[i])

	pl.savefig(outf)
	
# ASSUME school closings START and then assumed to be in effect until SchoolOut date
	
# Following jpg_Calibration_UK
ClosedSchoolBeta = {'s': 0.02,
					'h': 1.29,
					'w': 0.2,
					'c': 0.2}

def bldEducLevelBeta(intrvList):
	'''convert list of (datestr,elev,endDate) into per-level beta changes
		return list of per-layer changes
	'''
	
	levIntrv = {}
	for idate,edate in intrvList:
		bdates = [idate, edate]
		for lev in ClosedSchoolBeta.keys():
			newBeta = ClosedSchoolBeta[lev]
			bvec = [newBeta, 1.0]
			levIntrv[lev] = cv.interventions.change_beta(days=bdates, changes=bvec, layers=lev)
	
	return [levIntrv[elayer] for elayer in levIntrv.keys()]

def getCountryInfoDataFrame(datafile,missTest,minInfect=50):
	infoDict = {'start_date': None,
				'tot_pop': None}
	
	dataTbl = cvm.load_data(datafile)
	
	popVec = dataTbl['population']	
	# ASSUME total population doesn't change
	# NB: make tot_pop an integer
	infoDict['tot_pop'] = int(popVec[0])
	
	cummDiagVec = dataTbl['cum_diagnoses']
	for date in cummDiagVec.keys():
		if cummDiagVec[date] > minInfect:
			infoDict['start_date'] = date
			break
		
	if missTest:
		infoDict['ntestDays'] = 0
	else:
		if 'cum_tests' not in dataTbl:
			# print(f'getCountryInfoDataFrame: trying to get tests from file without?: {datafile}')
			infoDict['ntestDays'] = 0
		else:
			testVec = dataTbl['cum_tests']
			infoDict['ntestDays'] = sum(1 for v in testVec.notna() if v==True)

	return infoDict

def loadAllCountryData(dataDir,iso2cname,missTestList=[]):
	ecdcFiles = glob.glob(dataDir+'*.csv')
	allCnty = {}
	for ecdcf in sorted(ecdcFiles):
		spos = ecdcf.rfind('/')
		ppos = ecdcf.rfind('.')
		iso3 = ecdcf[spos+1:ppos]
		if iso3 not in iso2cname:
			print(f'loadAllCountryData: no country for ISO3?! {iso3}')
			continue
		country = iso2cname[iso3]
		
		# NB empty missTestList implies ASSUME country has testing data
		if missTestList == []:
			missTest = False
		else:
			missTest = iso3 in missTestList
		cntyInfo = getCountryInfoDataFrame(ecdcf,missTest,minInfect=50)
		# NB: add country to info
		cntyInfo['country'] = country
		
		allCnty[iso3] = cntyInfo

	print(f'loadAllCountryData: NCountry={len(allCnty)}')
	return allCnty

if __name__ == '__main__':
	
	global UseUNAgeData
	global UNAgeData
	global MinInfect
	
	global PlotFitMeasures
	global PlotDir
	global RptCMAGen
		
	AWS_Host = False
	AWS_S3Client = None
	AWS_S3Bucket = ''
	popSize = 32
			
	hostname = socket.gethostname()
	if hostname == 'hancock':
		dataDir = '/System/Volumes/Data/rikData/coviData/localData/'
		n_workers = 4
		n_trials  = 4 
		print(f'NB: hancock TESTING: n_worker={n_workers} ntrials={n_trials}')
	elif hostname == 'mjq':
		dataDir = '/home/Data/covid/data/'
		n_workers = 4
		n_trials  = 8
	elif hostname == 'covaguest':
		dataDir = '/home/rbelew/data/'
		n_workers = 120
		n_trials  = 1
		popSize = 120

	else:
		print(f'odd host!? {hostname}')
		sys.exit(0) 

	argv = sys.argv
	runPrefix = argv[1]
	print(argv,len(argv))
	# NB: second argument if restart required
	if len(argv) < 3:
		print(f'main: cvsimOne requires ISO3 argument!')
		sys.exit()
		
	iso3 = argv[2]

	SchoolClose = False	
	UseTestRate = 'data' #  'srch' or 'fix' or 'data' 
	LogTransform = False
	ParamLogTransformed = ['pop_infected','beta']
	RunName = f'{runPrefix}_close-{SchoolClose:d}_test-{UseTestRate}'
	
	Calibrate = True
	Calibrator = 'abc'
	
	print(f'{RunName} on {hostname} popSize={(n_workers*n_trials)}/{popSize} n_workers={n_workers} n_trials={n_trials}')
	print(f'SchoolClose={SchoolClose} UseTestRate={UseTestRate} Calibrate={Calibrate} LogTransform={LogTransform}')
	
	ECDCDir = dataDir + 'ecdc+test/'

	runDir = dataDir + f'runs/{RunName}/'
	if not os.path.exists(runDir):
		print( 'creating runDir',runDir)
		os.mkdir(runDir)
	
	DBDir = runDir + 'db/'
	if not os.path.exists(DBDir):
		print( 'creating DBDir',DBDir)
		os.mkdir(DBDir)
		
	PlotDir = runDir + 'plots/'
	if not os.path.exists(PlotDir):
		print( 'creating PlotDir',PlotDir)
		os.mkdir(PlotDir)
		
	PlotInitial = False
	PlotPeople = False
	PlotFitMeasures = False
	RptCMAGen = False
	doABCPlots = True
		
	MinInfect = 50
	
	datacols = ['date','population', 'cum_diagnoses', 'cum_deaths']
	if UseTestRate == 'data':
		datacols.append('new_tests')
	
	to_plot =  ['cum_diagnoses', 'cum_deaths', 'cum_tests']

	commonParam = {  'start_day': ModelStartDate,
					'end_day':  ModelEndDate,
					'pop_size':  1e5, 
		    		'rand_seed': 1, 
		   		 	'pop_type': 'hybrid', # 'educlevel', 
				   	'asymp_factor': 2,
					# 'contacts': educLevelSize,
					'rescale': True,
					'verbose': 0., # 0.1
					'datacols': datacols,
					
					# NB: need to pass these global variables into create_sim()
					'LogTransform': LogTransform,
					'UseTestRate': UseTestRate,

		}

	ecdcGeogFile = dataDir +  'COVID-19-geographic-disbtribution-worldwide.csv'	
	countryInfo,cname2iso,iso2cname = loadGeogInfo(ecdcGeogFile)
	
	countryDataDir = dataDir + 'ecdc+test/'
	# NB: missingTestList defaults to empty [] 
	#	OK because will be ignored below when UseTestRate == 'data'
	countryData = loadAllCountryData(countryDataDir,iso2cname)
			
	interveneFile = dataDir + 'consensusIntrvn.csv'
	allEducIntrvn = loadConsensusEducIntrvn(interveneFile)
	ncountry = len(allEducIntrvn)
	
	# begin ONE country
	
	country = iso2cname[iso3]
	print(f'main: {iso3} - {country}: begin')

	cntyInfo = countryData[iso3]
	if UseTestRate == 'data' and cntyInfo['ntestDays'] == 0:
		print(f'main: skipping {iso3} without test data')
		sys.exit(-1)
	
	educIntrvn = allEducIntrvn[iso3]
	
	# separate database for cvsim trials data
	cmaDBfile = f'{DBDir}{iso3}-cma.db'
	if os.path.exists(cmaDBfile):
		print('cmaDB %s exists; DELETING!' % cmaDBfile)
		os.remove(cmaDBfile)

	abcDBfile = f'{DBDir}{iso3}.db'
	if os.path.exists(abcDBfile):
		# print('DB %s exists; skipping' % dbfile)
		# continue
		print('abcDB %s exists; DELETING!' % abcDBfile)
		os.remove(abcDBfile)

	pars = commonParam.copy()
			
	pars['start_day'] = cntyInfo['start_date']
	pars['tot_pop'] = cntyInfo['tot_pop']
		
	loc = country.replace('_',' ')
	if loc in NormCountry:
		loc = NormCountry[loc]
	pars['location'] = loc
	
	pars['country'] = country
		
	pars['datafile'] = countryDataDir + f'{iso3}.csv'
	# 201110: refer to  GLOBAL var ala jpg_CalibUK
	cvStorage   = f'sqlite:///{cmaDBfile}'
	print(f'main: cvStorage={cvStorage}')
	pars['storage'] = cvStorage
	
	pop_scale = int(pars['tot_pop']/pars['pop_size'])
	pars['pop_scale'] = pop_scale
	# pars['age_dist'] = ageData
	pars['interventions'] = []
		
	# NB UseTestRate == data or search: interventions built in create_sim
	if UseTestRate == 'fix':	
		testIntrvn = cvintrv.test_prob(symp_prob=TypicalTestRate, asymp_prob=TypicalTestRate)
		testIntrvn.do_plot = False
		pars['interventions'].append(testIntrvn)
	
	if SchoolClose:			
		
		beta_changes = bldEducLevelBeta(educIntrvn)
		
		for intrvn in beta_changes:
			intrvn.do_plot = True

		pars['interventions'] += beta_changes		

	CurrCtnyParams = pars	
	
	if PlotInitial:
		# initial run
		print(f'Running initial for {iso3}...')
		pars, pkeys = get_bounds() # Get parameter guesses
		sim = create_sim(pars.best,CurrCtnyParams)
		sim.run()
		
		sim.plot(to_plot=to_plot,do_save=True,fig_path=PlotDir + iso3 + '-initial.png')
		pl.gcf().axes[0].set_title('Initial parameter values')
		objective(pars.best)
		pl.pause(1.0) # Ensure it has time to render
	
	if PlotPeople:
		peopleFig = sim.people.plot()
		peoplePlotFile = PlotDir + iso3 + '-people.png'
		peopleFig.savefig(peoplePlotFile)
	
	if Calibrate:
		# Calibrate
		print(f'main: Starting calibration for {iso3}...')
		
		T = sc.tic()

		try:
			
			if Calibrator == 'cmaes':
				cmaDB = sqlite.connect(cmaDBfile)		
				pars_calib = calibrate_CMA(cmaDB)
				
			elif Calibrator == 'abc':
				abcDBFile = f'{DBDir}{iso3}-abc.db'
				print(f'main: abcDBFile={abcDBFile}')
				
				abcDB = "sqlite:///" + abcDBFile
				abcHistory = calibrate_ABC(abcDB)
			
		except Exception as e:
			sc.toc(T)
			print(f'main: EXCEPTION during calibration? {iso3} {e}')
			sys.exit(-1)

		sc.toc(T)

		if Calibrator == 'cmaes':
			x = pars_calib
		elif Calibrator == 'abc':

			# HACK: abcHistory.db is just the sqlite string; use sqlite directly
			paramStats = getABC_FinalPopStats(abcDBFile)
			print(f'main: popMean="{paramStats}"')
			
			paramKDE = getABC_maxKDE(abcHistory)
			print(f'main: paramKDE="{paramKDE}"')

			# convert to best,lb,ub format of params
			covaSimParamOrder = ['pop_infected','beta']
			if UseTestRate=='srch':
				covaSimParamOrder.append('test_rate')
				
	
			x = [ paramKDE[param]['kde_max'][0] for param in covaSimParamOrder]
	
		# Plot result
		if doABCPlots:
			
			bounds, pkeys = get_bounds()
			
			_, axes = pl.subplots(2, 2)
			plot_sample_numbers(abcHistory, ax=axes[0][0])
			plot_epsilons(abcHistory, ax=axes[0][1])
			plot_credible_intervals(abcHistory, arr_ax=axes[1][0])
			plot_effective_sample_sizes(abcHistory, ax=axes[1][1])
			pl.gcf().set_size_inches((12, 8))
			pl.gcf().tight_layout()
			
			pl.savefig(PlotDir + f'{iso3}-pyABC.png')
			print('pyABC plotted')
					
			fig, ax = pl.subplots()
			df, w = abcHistory.get_distribution(m=0)
			plot_kde_matrix(df, w,limits=bounds)
			
			pl.savefig(PlotDir + f'{iso3}-kdeMatrix.png')
			print('kdeMatrix plotted')

			crediblePlotFile = PlotDir + f'{iso3}-credible.png'
			plot_credible(abcHistory, crediblePlotFile)
			print('credible plotted')
			
		
			histPlotFile = PlotDir + f'{iso3}-history.png'
			plot_history(abcHistory, bounds, histPlotFile)
			print('history plotted')
		
		print('Plotting result...')

		# x provided by either CMAES or pyABC
		
		sim = create_sim(x,CurrCtnyParams)
			
		sim.run()
		
		fig_path = PlotDir + f'{iso3}-fit.png'
		sim.plot(to_plot=to_plot,do_save=True,fig_path=fig_path)
			
	print(f'main: {iso3} - {country}: end')
