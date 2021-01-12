''' cvsimTst:  testing of covasim
Created on Aug 11, 2020

@version: 0.2.2
Oct 2 2020

@author: rbelew@ucsd.edu
'''

from collections import defaultdict
import csv
from datetime import date
import datetime
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
# import optuna as op
# from optuna.samplers import CmaEsSampler

from multiprocessing import Pool, cpu_count

import cma
# import astroabc

# https://stackoverflow.com/a/57364423
# istarmap.py for Python 3.8+
import multiprocessing.pool as mpp
from covasim.utils import false

def istarmap(self, func, iterable, chunksize=1):
	"""starmap-version of imap
	"""
	self._check_running()
	if chunksize < 1:
		raise ValueError(
			"Chunksize must be 1+, not {0:n}".format(
				chunksize))

	task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
	result = mpp.IMapIterator(self)
	self._taskqueue.put(
		(
			self._guarded_task_generation(result._job,
										  mpp.starmapstar,
										  task_batches),
			result._set_length
		))
	return (item for chunk in result for item in chunk)
mpp.Pool.istarmap = istarmap

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

	
def get_bounds():
	''' Set parameter starting points and bounds 
		NB: uses ORDERED dictionary => specifiy bounds IN ORDER!'''

	pdict = sc.objdict()

	# Cliff's auto_calibrate
# 	pop_infected = dict(best=10000, lb=1000,  ub=50000),
# 	beta		 = dict(best=0.015, lb=0.007, ub=0.020),

	# JPG UK calibration
# 	pop_infected = dict(best=4500,  lb=1000,   ub=10000),
# 	beta		 = dict(best=0.00522, lb=0.003, ub=0.008),

	# 201103
	pdict['pop_infected'] = dict(best=21000,lb=16000,   ub=26000)
	pdict['beta']         = dict(best=.005, lb=0.001, ub=0.01)

# 	pdict['pop_infected'] = dict(best=10000,lb=1000, ub=50000)
# 	pdict['beta']         = dict(best=.005, lb=0.003, ub=0.02)
	
	# X[2] = test_rate
	if UseTestRate=='srch':
		# 201215: using bounds from searchData_201213.ods
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

# 201110: refer to  GLOBAL var ala jpg_CalibUK
# def objective(x):
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

def op_objective(trial):

	pars, pkeys = get_bounds() # Get parameter guesses
	x = np.zeros(len(pkeys))
	for k,key in enumerate(pkeys):
		x[k] = trial.suggest_uniform(key, pars.lb[k], pars.ub[k])

	return objective(x)

def worker(CurrCtnyParams):
	# 201110: refer to  GLOBAL var ala jpg_CalibUK
	storage = CurrCtnyParams['storage']
	name = RunName + CurrCtnyParams['location']
	study = op.load_study(storage=storage, study_name=name)
	return study.optimize(op_objective, n_trials=n_trials)

def run_workers(CurrCtnyParams):
	# return sc.parallelize(worker, n_workers, kwargs={'CurrCtnyParams':CurrCtnyParams}, ncpus=4)
	return sc.parallelize(worker, n_workers, kwargs={'CurrCtnyParams':CurrCtnyParams}, ncpus=n_cpus)

def make_study():

	# 201110: refer to  GLOBAL var ala jpg_CalibUK
	# storage = CurrCtnyParams['storage']
	# name = CurrCtnyParams['location']
	
	try: 
		op.delete_study(storage=storage, study_name=name)
	except: 
		pass
	if OptunaSampler == None:
		return op.create_study(storage=storage, study_name=name)
	else:
		return op.create_study(storage=storage, study_name=name,sampler=OptunaSampler)

# def calibrate(cntyParams):
def calibrate():
	''' Perform the calibration wrt/ GLOBAL CurrCtnyParams
	'''
	
	# 201110: refer to  GLOBAL var ala jpg_CalibUK
# 	storage = CurrCtnyParams['storage']
# 	name = CurrCtnyParams['location']

	make_study()
	run_workers(CurrCtnyParams)
	study = op.load_study(storage=storage, study_name=name)
	output = study.best_params
	return output, study

def calibrate_CMA(currDB):
	'''calibrate parameters using CMA library
	'''
	
	print('NB: TESTING only 3 generations!')
	ngen = 3 # 25

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
			levIntrv[lev] = cv.change_beta(days=bdates, changes=bvec, layers=lev)
	
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


	
# 201110: refer to  GLOBAL var ala jpg_CalibUK
# storage   = 'sqlite:////System/Volumes/Data/rikData/coviData/localData/db/Austria-cvsim.db'
# name = 'tstOptuna'
# n_trials = 5
# n_workers = 1
# n_cpus = 1
# UseTestRate = 'srch' #  'srch' or 'fix' or 'data' 
# LogTransform = False
# ParamLogTransformed = ['pop_infected','beta']
# RunName = 'runName'

if __name__ == '__main__':
	
	global UseUNAgeData
	global UNAgeData
	global MinInfect
	
	global PlotFitMeasures
	global PlotDir
	global RptCMAGen
	
	# 201110: refer to  GLOBAL var ala jpg_CalibUK
# 	global n_trials
# 	global n_workers
# 	global UseTestRate
# 	global LogTransform
# 	global ParamLogTransformed

# 	global CurrCtnyParams
			
# 	global AWS_S3Client
# 	global AWS_S3Bucket
	
	AWS_Host = False
	AWS_S3Client = None
	AWS_S3Bucket = ''
	popSize = 32
			
	hostname = socket.gethostname()
	if hostname == 'hancock':
		dataDir = '/System/Volumes/Data/rikData/coviData/localData/'
		print('NB: TESTING: n_worker=1 ntrials=3')
		n_workers = 1 # 4
		n_trials  = 3 # 8 
	elif hostname == 'mjq':
		dataDir = '/home/Data/covid/data/'
		n_workers = 4
		n_trials  = 8
	elif hostname.startswith('ip-'):
		AWS_S3Client = boto3.client('s3')
		AWS_S3Bucket = 'cvsim'
		AWS_Host = True

		dataDir = '/home/ubuntu/data/'
		
		n_workers = cpu_count()
		n_trials  =  int(popSize / n_workers)
	else:
		print(f'odd host!? {hostname}')
		sys.exit(0) 

	argv = sys.argv
	runPrefix = argv[1]
	# NB: second argument if restart required
	if len(argv) > 2:
		restartISO3 = argv[2]
		print(f'main: restarting after {restartISO3}')
	else:
		restartISO3 = 'AAA'
		print(f'main: starting at beginning')	

	SchoolClose = True	
	UseTestRate = 'data' #  'srch' or 'fix' or 'data' 
	LogTransform = False
	ParamLogTransformed = ['pop_infected','beta']
	RunName = f'{runPrefix}_close-{SchoolClose:d}_test-{UseTestRate}'
	
	Calibrate = True
	
	print(f'{RunName} on {hostname} popSize={(n_workers*n_trials)}/{popSize} n_workers={n_workers} n_trials={n_trials}')
	print(f'SchoolClose={SchoolClose} UseTestRate={UseTestRate} Calibrate={Calibrate} LogTransform={LogTransform}')
	
	# dataDir = 'PATH_TO_CVSIM_DATA'

	# local directory from https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide-2020-08-04.xlsx
	# ASSUME simplifyECDCData(ECDCDirOrig,ECDCDir) has been run
	# ECDCDirOrig = dataDir + 'data/ecdc-orig/'
	# ASSUME mrgOWIDTestData() has added testing data
	# 	ECDCDir = dataDir + 'data/ecdc-simple_201002/'
	# 	OWIDTestFile = dataDir + 'data/OWID-daily-tests-per-thousand-people-smoothed-7-day_201001.csv'
	# 	mergeDir = dataDir + 'data/ecdc+test/'
	# 	mrgOWIDTestData(ECDCDir,OWIDTestFile,mergeDir)

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
	

# n_trials = 5
# n_workers = 1
# n_cpus = 1
# UseTestRate = 'srch' #  'srch' or 'fix' or 'data' 
# LogTransform = False
# ParamLogTransformed = ['pop_infected','beta']
# RunName = 'runName'

	###### Run options
	
# 	cmaesSampler = CmaEsSampler()
# 	OptunaSampler = cmaesSampler # None
	
	PlotInitial = False
	PlotPeople = False
	PlotFitMeasures = False
	RptCMAGen = False
		
	MinInfect = 50
	
	# all data columns
	# ['Unnamed: 0', 'key', 'population', 'aggregate', 'cum_diagnoses', 'cum_deaths', 'cum_recovered', 
	# 'cum_active', 'cum_tests', 'cum_hospitalized', 'hospitalized_current', 'cum_discharged', 'icu', 
	# 'icu_current', 'date', 'day', 'diagnoses', 'deaths', 'tests', 'hospitalized', 'discharged', 
	# 'recovered', 'active']

	datacols = ['date','population', 'cum_diagnoses', 'cum_deaths']
	if UseTestRate == 'data':
		datacols.append('new_tests')
	
	to_plot =  ['cum_diagnoses', 'cum_deaths', 'cum_tests']

	parsCommon = {  'start_day': ModelStartDate,
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
	
	allISO3 = sorted(list(allEducIntrvn.keys()))
	
	print('NB: TESTING 1 COUNTRIES ONLY!',['ARG'])
	for ci,iso3 in enumerate(['ARG']):
	# for ci,iso3 in enumerate(allISO3):
		if iso3 <= restartISO3:
			print(f'main: skipping {iso3}')
			continue
		
		country = iso2cname[iso3]
		print(f'main: {ci}/{ncountry} {iso3} - {country}: begin')

		cntyInfo = countryData[iso3]
		if UseTestRate == 'data' and cntyInfo['ntestDays'] == 0:
			print(f'main: skipping {iso3} without test data')
			continue
		
		educIntrvn = allEducIntrvn[iso3]
		
		# separate database for cvsim trials data
		mydbfile = f'{DBDir}{iso3}-cvsim.db'
		if os.path.exists(mydbfile):
			print('DB %s exists; DELETING!' % mydbfile)
			os.remove(mydbfile)
		currDB = sqlite.connect(mydbfile)		
		
		dbfile = f'{DBDir}{iso3}.db'
		if os.path.exists(dbfile):
			# print('DB %s exists; skipping' % dbfile)
			# continue
			print('DB %s exists; DELETING!' % dbfile)
			os.remove(dbfile)

		pars = parsCommon.copy()
		
		# 201110: refer to  GLOBAL var ala jpg_CalibUK
		storage   = f'sqlite:///{dbfile}'
		
		pars['start_day'] = cntyInfo['start_date']
		pars['tot_pop'] = cntyInfo['tot_pop']
			
		loc = country.replace('_',' ')
		if loc in NormCountry:
			loc = NormCountry[loc]
		pars['location'] = loc
		
		pars['country'] = country
			
		pars['datafile'] = countryDataDir + f'{iso3}.csv'
		# 201110: refer to  GLOBAL var ala jpg_CalibUK
		pars['storage'] = storage
		
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
			print(f'Starting calibration for {iso3}...')
			
			T = sc.tic()
			
			try:
				# print(f'** Sampler={OptunaSampler}')
				# pars_calib, study = calibrate()
				
				pars_calib = calibrate_CMA(currDB)
				
			except Exception as e:
				sc.toc(T)
				print(f'main: EXCEPTION {iso3} {e}')
				continue
	
			sc.toc(T)
		
			# Plot result
			print('Plotting result...')
			sim = create_sim(pars_calib,CurrCtnyParams)
				
			sim.run()
			
			fig_path = PlotDir + f'{iso3}-fit.png'
			sim.plot(to_plot=to_plot,do_save=True,fig_path=fig_path)
			
		if AWS_Host:
			#   An Amazon S3 bucket has no directory hierarchy...
			#   You can, however, create a
			#   logical hierarchy by using object key names that imply a folder
			#   structure. 
			
			print('Saving db,plot.log to S3...')
			
			# mydbfile = f'{DBDir}{iso3}-cvsim.db'
			dbPathAtS3 = f'{RunName}/db/{iso3}-cvsim.db'
			AWS_S3Client.upload_file(mydbfile, AWS_S3Bucket, Key=dbPathAtS3)
			
			figPathAtS3 = f'{RunName}/plot/{iso3}-fit.png'
			AWS_S3Client.upload_file(fig_path, AWS_S3Bucket, Key=figPathAtS3)
			
			logFilePath = f'/home/ubuntu/src/{runPrefix}.log'
			# It is not possible to append to an existing S3 object.  
			# NB: REPLACE log file with updated version
			logPathAtS3 = f'{RunName}/{runPrefix}.log'
			AWS_S3Client.upload_file(logFilePath, AWS_S3Bucket, Key=logPathAtS3)
		
		print(f'main: {ci}/{ncountry} {iso3} - {country}: end')
