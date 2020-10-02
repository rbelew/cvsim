''' cvsimTst:  testing of covasim
Created on Aug 11, 2020

@version: 0.2.2
Oct 2 2020

@author: rbelew@ucsd.edu
'''

from collections import defaultdict
import csv
from datetime import date
from itertools import repeat
import math
import pickle
import os
import socket
import sys

import numpy as np
import pylab as pl

from sklearn import metrics as skmetric

import sqlite3 as sqlite

import covasim as cv
import covasim.interventions as cvintrv
import covasim.misc as cvm

# from covasim.data import country_age_data as cad

import sciris as sc
import optuna as op
from optuna.samplers import CmaEsSampler

from multiprocessing import Pool, cpu_count
import cma

# https://stackoverflow.com/a/57364423
# istarmap.py for Python 3.8+
import multiprocessing.pool as mpp
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
n_trials  = 10
n_workers = 4

# Countries WITH education level interventions AND at least a week of data before this
EducCountryNames = ['Austria', 'Belgium', 'Switzerland', 'Czechia', 'Germany',
					'Denmark', 'Ecuador', 'Spain', 'Estonia', 'Finland', 'France',
					'Greece', 'Croatia', 'Hungary', 'India',
					'Ireland', 'Italy', 'Japan', 'Kazakhstan', 'Kuwait', 'Lithuania',
					'Mexico', 'North Macedonia', 'Malaysia', 'Netherlands',
					'Norway', 'New Zealand', 'Poland', 'Portugal', 'Romania', 'Senegal',
					'Singapore', 'Serbia', 'Slovenia', 'Taiwan']

DecadeKeys = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
EducLevel_ages = {'pre': [0,3],'sk': [4,5], 'sp': [6,10], 'ss': [11,18], 'su': [19,24],'work': [25, 65],'old': [66,100]}
EducLevelKeys = ['pre','sk', 'sp', 'ss', 'su','work','old']
TypicalTestRate = 0.01


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
					PRIMARY KEY(idx) ) '''
	curs.execute(cmd)

	return currDB

def loadTotalPop(inf):	
	reader = csv.DictReader(open(inf))
	popTbl = {}
	for i,entry in enumerate(reader):
		# Country,popData2019
		cname = entry['Country']
		try:
			pop19 = int(entry['popData2019'])
		except Exception as e:
			print('bad totPop',cname)
			pop19 = 0
		popTbl[cname] = pop19
		
	return popTbl

def loadEducIntervn(inf):
	reader = csv.DictReader(open(inf))
	interveneTbl = defaultdict(list) # country -> [(date,educLevelSet), ...]
	for i,entry in enumerate(reader):
		# ISO3,CName,Date,Levels
		cname = entry['CName']
		date = entry['Date']
		levStr = entry['Levels']
		elevSet = eval(levStr)
		interveneTbl[cname].append( (date,elevSet) )
		
	return interveneTbl

def loadEndEducDates(inf):
	reader = csv.DictReader(open(inf))
	interveneTbl = defaultdict(list) # country -> [(date,educLevelSet), ...]
	for i,entry in enumerate(reader):
		# ISO3,CName,Date,Levels
		cname = entry['CName']
		date = entry['Date']
		levStr = entry['Levels']
		elevSet = eval(levStr)
		interveneTbl[cname].append( (date,elevSet) )
		
	return interveneTbl

def loadAgeDistrib(inf):
	'''capture age stratification suitable to educLevel interventions
	also builds decade stratification for comparison to covid19_scenarios ages
	return allAgeStrat: country->educLevelKey->n, allDecStrat: country->decadeKey->n
	'''

	maxAge = 100
	decTbl = {}
	for dkey in DecadeKeys:
		if dkey.find('+') != -1:
			minAge = int(dkey[:-1])
			decTbl[dkey] = [minAge,maxAge]
		else:
			bits = dkey.split('-')
			decTbl[dkey] = [int(bits[0]),int(bits[1])]
			
	reader = csv.DictReader(open(inf))
	allAgeStrat = {}
	allDecStrat = {}
	for i,entry in enumerate(reader):
		# Index,Variant,Country,Notes,Country code,Type,Parent code,RefDate,,0,1,2,3,4,5,6,7,...,98,99,100
		country = entry['Country'].strip()
		if country not in EducCountryNames:
			continue

		ageStrat = defaultdict(int) # ageRangeStr -> n
		decStrat = defaultdict(int)
		currAgeBinIdx = 0 
		currAgeBin = EducLevel_ages[ EducLevelKeys[currAgeBinIdx] ]
		currDecBinIdx = 0 
		currDecBin = decTbl[ DecadeKeys[currDecBinIdx] ]
		for age in range(101):
			if age > currAgeBin[1]:
				currAgeBinIdx += 1
				currAgeBin = EducLevel_ages[ EducLevelKeys[currAgeBinIdx] ]
			if age > currDecBin[1]:
				currDecBinIdx += 1
				currDecBin = decTbl[ DecadeKeys[currDecBinIdx] ]
			ageStr = str(age)
			# NB: UN-WPP stats are in 1000's
			nthousand = int(entry[ageStr])
			nage = nthousand * 1000
			ageStrat[ EducLevelKeys[currAgeBinIdx] ] += nage 
			decStrat[ DecadeKeys[currDecBinIdx] ] += nage
			
		allAgeStrat[country] = dict(ageStrat)
		allDecStrat[country] = dict(decStrat) 
		
	return (allAgeStrat,allDecStrat)

def educAgeToNPArr(entries):

	# after data.loaders.get_age_distribution()
	
	# UNAgeDistFile = dataDir + 'data/ageDist/UN-WPP-fullAge2.csv'
	# NB: python dictionaries returned
	 #allAgeStratDict,allDecStratDict = loadAgeDistrib(UNAgeDistFile)

	
	result = {}
	for loc,age_distribution in entries.items():
		total_pop = sum(list(age_distribution.values()))
		local_pop = []

		for ageStrat, age_pop in age_distribution.items():
			ageRange = EducLevel_ages[ageStrat]
			val = [ageRange[0], ageRange[1], age_pop/total_pop]
			local_pop.append(val)
		result[loc] = np.array(local_pop)
			
	return result
	
def loadFinalTestRate(inf):	
	reader = csv.DictReader(open(inf))
	testRate = {} # country -> testRate
	
	for i,entry in enumerate(reader):
		# Country,UseTest,testPerDiag,testPerDeath,indivPerDiag,indivPerDeath,lastNDiag,lastNDeath,diagEst,deathEst,Pop19,NTest,TestRate
		country = entry['Country']
		trate = float(entry['TestRate'])
		testRate[country] = trate
		
	return testRate

def create_sim(x,currCtyPar):

	if currCtyPar['useLogParam']:
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
		contacts	 = currCtyPar['contacts'],
		rescale	  = currCtyPar['rescale'],
		verbose	  = currCtyPar['verbose'],
		interventions = currCtyPar['interventions'],
		# 

		beta		 = beta,
		pop_infected = pop_infected,
	)

	# Create the baseline simulation
	sim = cv.Sim(pars=pars,datafile=currCtyPar['datafile'],location=currCtyPar['location'], \
				age_dist = currCtyPar['age_dist'], datacols=currCtyPar['datacols'] )

	# NB: contacts set via pars, but parallel layer-dependent dicts are not!


	# NB: contacts getting 's' layer added during initialization, somewhere?
	del sim['contacts']['s']
	
	beta_layer  = dict(h=3.0, w=0.6, c=0.3)
	for elev in EducLevelLayers:
		beta_layer[elev] = 0.6	
	dynam_layer = dict(h=0,   w=0,   c=0)
	for elev in EducLevelLayers:
		dynam_layer[elev] = 0
	iso_factor  = dict(h=0.3, w=0.1, c=0.1)
	for elev in EducLevelLayers:
		iso_factor[elev] = 0.1
	quar_factor = dict(h=0.6, w=0.2, c=0.2)
	for elev in EducLevelLayers:
		quar_factor[elev] = 0.2
		
	sim['beta_layer'] = beta_layer
	sim['dynam_layer'] = dynam_layer
	sim['iso_factor'] = iso_factor
	sim['quar_factor'] = quar_factor
	
	# from JPG's Calibration_UK
	# what do they do?!
	sim['prognoses']['sus_ORs'][0] = 1.0 # ages 0-10
	sim['prognoses']['sus_ORs'][1] = 1.0 # ages 10-20


	# NB: UseTestRate==constant or ==data: test interventions already added in _main
	if currCtyPar['useTestRate']=='search':
		# NB: no LOG transform on test_rate
		test_rate = x[2]

		testIntrvn = cvintrv.test_prob(symp_prob=test_rate, asymp_prob=test_rate)
		testIntrvn.do_plot = False
		
		pars['interventions'].append(testIntrvn)
		
		sim.update_pars(interventions=pars['interventions'])

	return sim

	
def get_bounds():
	''' Set parameter starting points and bounds '''

	pdict = sc.objdict(
		pop_infected = dict(best=10000,  lb=1000,   ub=50000),
		beta		 = dict(best=0.015, lb=0.007, ub=0.020),
	)

	if UseTestRate=='search':
		trBest = TypicalTestRate
		trLB = 0. # trBest * 0.1
		trUB = 1. # trBest * 10
		pdict['test_rate'] = dict(best=trBest,  lb=trLB,   ub=trUB)
		
	if UseLogParam:
		# NB: only ['pop_infected','beta'] LOG transformed
		for param in ParamLogTransformed:
			
			for key in ['best', 'lb', 'ub']:
				pdict[param][key] = math.log(pdict[param][key])
		
		
	# JPG calibration
# 	pdict = sc.objdict(
# 		beta		 = dict(best=0.00522, lb=0.003, ub=0.008),
# 		pop_infected = dict(best=4500,  lb=1000,   ub=10000),
# 	)

	# Cliff's auto_calibration
# 	pdict = sc.objdict(
# 		pop_infected = dict(best=10000,  lb=1000,   ub=50000),
# 		beta		 = dict(best=0.015, lb=0.007, ub=0.020),
# 		beta_day	 = dict(best=20,	lb=5,	 ub=60),
# 		beta_change  = dict(best=0.5,   lb=0.2,   ub=0.9),
# 		symp_test	= dict(best=30,   lb=5,	ub=200),
# 	)

	# Convert from dicts to arrays
	pars = sc.objdict()
	for key in ['best', 'lb', 'ub']:
		pars[key] = np.array([v[key] for v in pdict.values()])

	return pars, pdict.keys()

def objective(x,currCtyPar):
	''' Define the objective function we are trying to minimize '''

	# Create and run the sim
	sim = create_sim(x,currCtyPar)
	sim.run()
	fit = sim.compute_fit()
	
	return fit.mismatch

def op_objective(trial):

	pars, pkeys = get_bounds() # Get parameter guesses
	x = np.zeros(len(pkeys))
	for k,key in enumerate(pkeys):
		x[k] = trial.suggest_uniform(key, pars.lb[k], pars.ub[k])

	return objective(x)

def worker(CurrCtnyParams):
	storage = CurrCtnyParams['storage']
	name = '200930_' + CurrCtnyParams['location']
	study = op.load_study(storage=storage, study_name=name)
	return study.optimize(op_objective, n_trials=n_trials)

def run_workers(CurrCtnyParams):
	return sc.parallelize(worker, n_workers, kwargs={'CurrCtnyParams':CurrCtnyParams}, ncpus=4)

def make_study():

	storage = CurrCtnyParams['storage']
	name = CurrCtnyParams['location']
	
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
	
	storage = CurrCtnyParams['storage']
	name = CurrCtnyParams['location']

	make_study()
	run_workers(CurrCtnyParams)
	study = op.load_study(storage=storage, study_name=name)
	output = study.best_params
	return output, study

def calibrate2(currDB):
	
	ngen = 25
	cmaesPopSize = n_trials * n_workers  
	print(f'calibrate2: cmaesPopSize={cmaesPopSize} ngen={ngen}')

	pars, pkeys = get_bounds() # Get parameter guesses
	print(f'calibrate2: pars={pars}')

	
	hdr = '# Gen,I,' + ','.join(pkeys)
	print(hdr)
	
	initDB(currDB)
	cursor = currDB.cursor()
	trialAttrbNames = 'gen,indiv,value,infect,beta,testrate'
	
	esopts = { "popsize": cmaesPopSize, \
				# 'CMA_elitist': True,

				# Argument bounds can be None or bounds[0] and bounds[1] are lower and
				# upper domain boundaries, each is either None or a scalar or a list
				# or array of appropriate size.	            
				'bounds': [np.array(pars['lb']), np.array(pars['ub']) ],
				# 'bounds': [ [1.e+03, 1.e-03],[5.e+04, 8.e-03] ],
				
				'verbose': 1,
				} 
	if not UseLogParam:
		esopts['CMA_stds'] = [1e4,0.01]

	es = cma.CMAEvolutionStrategy(
            x0=np.array(pars['best']),
            sigma0=1e-1, 
            inopts=esopts
        )	

	for gen in range(25):
		solutions = []
		# values = []
		currPop = es.ask()
					
		# NB: fixed number=4 processes
		with Pool(4) as pool:
			
			zipObj = zip(currPop, repeat(CurrCtnyParams))
			argList = list(zipObj)
			
			values = pool.starmap(objective,argList)
			
			for i,x in enumerate(currPop):
				if UseLogParam:
					infect = math.exp(x[0])
					beta = math.exp(x[1])
				else:
					infect = x[0]
					beta = x[1]

				# NB: testrate NOT subject to LOG transform
				if 'test_rate' in pkeys: 
					test_rate = x[2]
				else:
					# NB: for inclusion in trial DB
					test_rate = 0.
	
				# NB:  print full precision, for sampling database
				line = f"# {gen},{i},{values[i]:.60g},{infect:.60g},{beta:.60g}"
				if 'test_rate' in pkeys:
					line += f',{test_rate:.60g}'	
				print(line)
				
				valList = [ gen,i,values[i],infect,beta,test_rate ]
				qms = ','.join(len(valList)*'?')
				sql = 'insert into trial (' + trialAttrbNames + ') values (%s)' % (qms)
				cursor.execute(sql,tuple(valList))

			# eo-Pool context
				
		currDB.commit()
				
		es.tell(currPop, values)
		es.disp()
		stats = {'avg':{}, "std": {}}
		for i,k in enumerate(pkeys):
			vals = []
			for indiv in currPop:
				if UseLogParam and k in ParamLogTransformed:
					vals.append( math.exp(indiv[i]) )
				else:
					vals.append( indiv[i] )							
			stats['avg'][k] = np.mean( vals )
			stats['std'][k] = np.std(  vals )
		for i,k in enumerate(pkeys):
			print(f"> {gen} {i} {k} {stats['avg'][k]} {stats['std'][k]}")					
	
	es.result_pretty()
	bestX = es.best.x
	
	# NB: return results in same form as get_bounds(), as expected by create_sim()
	# NB: it will do exp() on these values if UseLogParam!
	
	return bestX
	


# ASSUME school closings START and then assumed to be in effect until SchoolOut date
SchoolOutDate = '2020-06-01'
ClosedSchoolBeta = 0.02
EducLevelLayers = ['sk', 'sp', 'ss', 'su']

def bldEducLevelBeta(intrvList,endDateSpec):
	'''convert list of (date, {elev}) into per-level beta changes
		ASSUME all intervention dates are  < SchoolOutDate
		ASSUME all closings stay in effect until SchoolOutDate
		return list of per-layer changes
	'''

	# cf. population.make_educLevel_contacts()
	# elevIntrvn = ['k', 'p', 's', 'u']
	
	# ASSUME school closings START and then assumed to be in effect
	bdays = sorted([date for date,elevSet in intrvList])
	
	bchange = {}
	levIntrv = {}
	if endDateSpec != None:
		# NB:  endDateSpec is LIST of (date,elevSet) tuples, ala standard interventions
		#      ASSUME end date is unique/country; use first
		specEndDate = endDateSpec[0][0]
		endLevSet = endDateSpec[0][1]
	for date,elevSet in intrvList:
		for elev in elevSet:
			elayer = 's' + elev
			bvec =   [ClosedSchoolBeta, 1.0]

			# 200828: Check for specified SchoolOutDate in 
			bdays.append(SchoolOutDate)
			if endDateSpec == None:
				endDate = SchoolOutDate
			else:
				if elev in endLevSet:
					endDate = specEndDate
				else:
					endDate = SchoolOutDate
				
			bdates = [date, endDate]
			levIntrv[elayer] = cv.change_beta(days=bdates, changes=bvec, layers=elayer)
	
	# 2DO: make changes based on school interventions!
# 	for lev in ['h','w','c']:
# 		for date in bdays:
# 			bchange[lev] = [1.0 for date in bdays]

	return [levIntrv[elayer] for elayer in levIntrv.keys()]


def getCountryInfo(datafile,minInfect=50):
	infoDict = {'start_date': None,
				'tot_pop': None}
	
	dataTbl = cvm.load_data(datafile)
	
	popVec = dataTbl['population']	
	# ASSUME total population doesn't change
	# NB: make tot_pop an integer
	infoDict['tot_pop'] = int(popVec[0])
	infoDict['ndays'] = len(popVec)
	
	cummDiagVec = dataTbl['cum_diagnoses']
	for date in cummDiagVec.keys():
		if cummDiagVec[date] > minInfect:
			infoDict['start_date'] = date
			break
		
	testVec = dataTbl['tests']
	infoDict['ntests'] = sum(1 for v in testVec.notna() if v==True)
	
	return infoDict

def rptECDPSumm():
	print('Country,Pop,StartDay,Ndays,Ntests')
	for country in sorted(EducCountryNames):
		datafile = ECDPDir + country + '.csv'									
		cntyInfo = getCountryInfo(datafile,minInfect=50)
		print(f'{country},{cntyInfo["tot_pop"]},{cntyInfo["start_date"]},{cntyInfo["ndays"]},{cntyInfo["ntests"]}')	
	
if __name__ == '__main__':
	
	global UseTestRate
	global UseUNAgeData
	global UNAgeData
	global UseLogParam
	global ParamLogTransformed
	
	global PlotFitMeasures
	global PlotDir
	
# 	global n_trials
# 	global n_workers	
# 	global CurrCtnyParams
			
	hostname = socket.gethostname()
	if hostname == 'hancock':
		dataDir = '/System/Volumes/Data/rikData/coviData/'
	elif hostname == 'mjq':
		dataDir = '/home/Data/covid/'

	# dataDir = 'PATH_TO_CVSIM_DATA'

	# local directory built from cv.load_ecdp_data.py
	#'European Centre for Disease Prevention and Control Covid-19 Data Scraper'
	# pars['load_path'] = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'

	ECDPDir = dataDir + 'data/epi_data/corona-data/'

	DBDir = dataDir + 'db/'
	if not os.path.exists(DBDir):
		print( 'creating DBDir',DBDir)
		os.mkdir(DBDir)
		
	PlotDir = dataDir + 'plots/'
	if not os.path.exists(PlotDir):
		print( 'creating PlotDir',PlotDir)
		os.mkdir(PlotDir)
	
	# Run options
	do_plot = 1
	do_save = 0
	verbose = 1
	interv  = 0
	
	Calibrate = True
	SchoolClose = True
	# 201001: 'data' untested because only 4 countries have testing data
	UseTestRate = 'constant' #  'search' or 'constant' or 'data' 
	UseLogParam = False
	ParamLogTransformed = ['pop_infected','beta']
	
	cmaesSampler = CmaEsSampler()
	OptunaSampler = cmaesSampler # None
	
	PlotInitial = False
	PlotPeople = False
	PlotFitMeasures = False

	CVNameMap = {'New_Zealand':    'New Zealand',
				'North_Macedonia': 'The former Yugoslav Republic of Macedonia',
				'Taiwan':          'Taiwan Province of China'}
	
	educLevelSize = {'h':4, 'w':20, 'c':20, 'sk': 20, 'sp': 20, 'ss': 40 , 'su': 80}
	# start_day = '2020-01-21'
	end_day =   '2020-07-31'
	
	# all data columns
	# ['Unnamed: 0', 'key', 'population', 'aggregate', 'cum_diagnoses', 'cum_deaths', 'cum_recovered', 'cum_active', 'cum_tests', 'cum_hospitalized', 'hospitalized_current', 'cum_discharged', 'icu', 'icu_current', 'date', 'day', 'diagnoses', 'deaths', 'tests', 'hospitalized', 'discharged', 'recovered', 'active']

	datacols = ['date','population', 'cum_diagnoses', 'cum_deaths', 'cum_recovered']

	parsCommon = {# 'start_day': start_day,
					'end_day':  end_day,
					'pop_size':  1e5, 
		    		'rand_seed': 1, 
		    		# NB: creating multi-education contact levels vs single school one
		   		 	'pop_type': 'educlevel', 
				   	'asymp_factor': 2,
					'contacts': educLevelSize,
					'rescale': True,
					'verbose': 0., # 0.1
					'datacols': datacols,
					
					# NB: need to pass these global variables into create_sim()
					'useLogParam': UseLogParam,
					'useTestRate': UseTestRate,

		}
	
	to_plot = ['cum_diagnoses', 'cum_deaths']
	if UseTestRate=='search':
		 to_plot.append('cum_tests')
	
	# totPop data redundant and perhaps inconsistent
	# but needed to initialize sim: pars['tot_pop'] = tot_pop ?
	# 2do: replace!
# 	
# 	totPopFile = dataDir + 'data/pop19Total.csv'
# 	totPop = loadTotalPop(totPopFile)
	
	interveneFile = dataDir + 'intervene/educ_intervene-uniq.csv'
	allEducIntrvn = loadEducIntervn(interveneFile)
	
	educEndDateFile = dataDir + 'intervene/educ_end-intervene.csv'
	educEndDates = loadEndEducDates(educEndDateFile)	
		
	# 200916: Expt: use GOOD test rates
	testRateFile = dataDir + 'data/testRate/' + 'goodTestRates.csv'
	allTestRates = loadFinalTestRate(testRateFile)

	UNAgeDistFile = dataDir + 'data/ageDist/UN-WPP-fullAge2.csv'
	# NB: python dictionaries returned
	allAgeStratDict,allDecStratDict = loadAgeDistrib(UNAgeDistFile)
	UNAgeData = educAgeToNPArr(allAgeStratDict)

	tstCountry = ['New_Zealand', 'Malaysia', 'North_Macedonia', 'Taiwan', 'Senegal','Singapore']
	
	for country in sorted(EducCountryNames):

		if country not in allEducIntrvn:
			print('* Missing interventions?!',country)
			import pdb; pdb.set_trace()
			continue

		# separate database for cvsim trials data
		mydbfile = f'{DBDir}{country}-cvsim.db'
		if os.path.exists(mydbfile):
			print('DB %s exists; DELETING!' % mydbfile)
			os.remove(mydbfile)
		currDB = sqlite.connect(mydbfile)		
		
		dbfile = f'{DBDir}{country}.db'
		if os.path.exists(dbfile):
			# print('DB %s exists; skipping' % dbfile)
			# continue
			print('DB %s exists; DELETING!' % dbfile)
			os.remove(dbfile)

		pars = parsCommon.copy()
		
		storage   = f'sqlite:///{dbfile}'
		datafile = ECDPDir + country + '.csv'
											
		cntyInfo = getCountryInfo(datafile,minInfect=50)
		
		pars['start_day'] = cntyInfo['start_date']
		pars['tot_pop'] = cntyInfo['tot_pop']
					
		pars['location'] = country
		pars['country'] = country
			
		pars['datafile'] = datafile
		pars['storage'] = storage
		pop_scale = int(pars['tot_pop']/pars['pop_size'])
		pars['pop_scale'] = pop_scale
		pars['age_dist'] = UNAgeData[country]
		pars['interventions'] = []
			
		if UseTestRate == 'constant':	
			fixTestRate = TypicalTestRate
		elif UseTestRate == 'data':
			fixTestRate = allTestRates[country]			
		else:
			print(f'{country}: SEARCHING for UseTestRate')
			fixTestRate = None
		
		if 	fixTestRate != None:
			testIntrvn = cvintrv.test_prob(symp_prob=fixTestRate, asymp_prob=fixTestRate)
			testIntrvn.do_plot = False
			pars['interventions'].append(testIntrvn)
		
		if SchoolClose:

			educIntrvn = allEducIntrvn[country]
			
			if country in educEndDates:
				endDateSpec =  educEndDates[country]
			else:
				endDateSpec = None
			
			beta_changes = bldEducLevelBeta(educIntrvn,endDateSpec)
			
			for intrvn in beta_changes:
				intrvn.do_plot = True
	
			pars['interventions'] += beta_changes		

		CurrCtnyParams = pars	

		# initial run
		print(f'Running initial for {country}...')
		# pars, pkeys = get_bounds(CurrCtnyParams['pop_size']) # Get parameter guesses
		pars, pkeys = get_bounds() # Get parameter guesses
		sim = create_sim(pars.best,CurrCtnyParams)
		sim.run()
		
		if PlotInitial:
			sim.plot(to_plot=to_plot,do_save=True,fig_path=PlotDir + country + '-initial.png')
			pl.gcf().axes[0].set_title('Initial parameter values')
			objective(pars.best)
			pl.pause(1.0) # Ensure it has time to render
		
		if PlotPeople:
			peopleFig = sim.people.plot()
			peoplePlotFile = PlotDir + country + '-people.png'
			peopleFig.savefig(peoplePlotFile)
		
		if Calibrate:
			# Calibrate
			print(f'Starting calibration for {country}...')
			T = sc.tic()
			
			print(f'** Sampler={OptunaSampler}')
			pars_calib, study = calibrate()
			# pars_calib = calibrate2(currDB)
	
			sc.toc(T)
		
			# Plot result
			print('Plotting result...')
# 			pars2calibList = [pars_calib['pop_infected'],pars_calib['beta']]
# 			if UseTestRate=='search':
# 				pars2calibList.append(pars_calib['test_rate'])
			sim = create_sim(pars_calib,CurrCtnyParams)
				
			sim.run()
			sim.plot(to_plot=to_plot,do_save=True,fig_path=PlotDir + country +'-fit.png')
			pl.gcf().axes[0].set_title('Calibrated parameter values')
			
			
