''' cvsimTst:  testing of covasim
Created on Aug 11, 2020

@version: 0.2.1
Sept 30 2020

@author: rbelew@ucsd.edu
'''

from collections import defaultdict
import csv
from datetime import date
import os
import socket
import sys

import numpy as np
import pylab as pl

from sklearn import metrics as skmetric

import covasim as cv
import covasim.interventions as cvintrv
import covasim.misc as cvm

# from covasim.data import country_age_data as cad

import sciris as sc
import optuna as op
from optuna.samplers import CmaEsSampler

CurrCtnyParams = {}
n_trials  = 2
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

def create_sim(x):

	beta = x[0]
	pop_infected = x[1]
	if UseTestRate=='search':
		test_rate = x[2]

	# cliffckerr, https://github.com/InstituteforDiseaseModeling/covasim/issues/269
	# The data file and location are properties of the simulation, not model
	# parameters, so they should only be supplied as arguments to Sim() and
	# should not be present in the parameters object. Storage only pertains
	# to Optuna, and should also not be part of the pars object.

	pars = sc.objdict(
		pop_size	 = CurrCtnyParams['pop_size'],
		pop_scale	= CurrCtnyParams['pop_scale'],
		pop_type	 = CurrCtnyParams['pop_type'],
		start_day	= CurrCtnyParams['start_day'],
		end_day	  = CurrCtnyParams['end_day'],
		asymp_factor = CurrCtnyParams['asymp_factor'],
		contacts	 = CurrCtnyParams['contacts'],
		rescale	  = CurrCtnyParams['rescale'],
		verbose	  = CurrCtnyParams['verbose'],
		interventions = CurrCtnyParams['interventions'],
		# 

		beta		 = beta,
		pop_infected = pop_infected,
	)

	# Create the baseline simulation
	sim = cv.Sim(pars=pars,datafile=CurrCtnyParams['datafile'],location=CurrCtnyParams['location'], \
				age_dist = CurrCtnyParams['age_dist'])
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
	if UseTestRate=='search':
		testIntrvn = cvintrv.test_prob(symp_prob=test_rate, asymp_prob=test_rate)
		testIntrvn.do_plot = False
		
		pars['interventions'].append(testIntrvn)
		
		sim.update_pars(interventions=pars['interventions'])

	return sim

def get_bounds(popsize):
	''' Set parameter starting points and bounds '''
	
	# popsize = CurrCtnyParams['pop_size']
	piLB = int(popsize * .001)
	piUB = int(popsize * .1)
	piBest = int( (piLB + piUB)/2 )
	pdict = sc.objdict(
		beta		 = dict(best=0.00522, lb=0.003, ub=0.008),
		pop_infected = dict(best=piBest,  lb=piLB,   ub=piUB),
	)
	if UseTestRate=='search':
		trBest = TypicalTestRate
		trLB = trBest * 0.1
		trUB = trBest * 10
		pdict['test_rate'] = dict(best=trBest,  lb=trLB,   ub=trUB)

	# Convert from dicts to arrays
	pars = sc.objdict()
	for key in ['best', 'lb', 'ub']:
		pars[key] = np.array([v[key] for v in pdict.values()])

	return pars, pdict.keys()

def objective(x):
	''' Define the objective function we are trying to minimize '''

	# Create and run the sim
	sim = create_sim(x)
	sim.run()
	fit = sim.compute_fit()
	
	return fit.mismatch

def op_objective(trial):

	pars, pkeys = get_bounds() # Get parameter guesses
	x = np.zeros(len(pkeys))
	for k,key in enumerate(pkeys):
		x[k] = trial.suggest_uniform(key, pars.lb[k], pars.ub[k])

	return objective(x)

def worker():
# def worker(storage,study_name,pop_size):
	storage = CurrCtnyParams['storage']
	name = '200930_' + CurrCtnyParams['location']
	study = op.load_study(storage=storage, study_name=name)
	return study.optimize(op_objective, n_trials=n_trials)


def run_workers():
	return sc.parallelize(worker, n_workers,ncpus=4)


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

def calibrate(sampler=None):
	''' Perform the calibration wrt/ GLOBAL CurrCtnyParams
	'''
	
	storage = CurrCtnyParams['storage']
	name = CurrCtnyParams['location']

	make_study()
	run_workers()
	study = op.load_study(storage=storage, study_name=name)
	output = study.best_params
	return output, study


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

def findStartDate(datafile,minInfect=50):
	dataTbl = cvm.load_data(datafile)
	cummDiagVec = dataTbl['cum_diagnoses']
	for date in cummDiagVec.keys():
		if cummDiagVec[date] > minInfect:
			return date
	return None
	
if __name__ == '__main__':
	
	global PlotFitMeasures
	global PlotDir
	global UseTestRate
	global UseUNAgeData
	global UNAgeData
	
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
	
	PlotInitial = False
	PlotPeople = False
	Calibrate = True
	SchoolClose = True
	UseTestRate = 'constant' # 'data' or 'search' or 'constant'
	PlotFitMeasures = False

# 	n_trials  = 2
# 	n_workers = 4

# 	OptunaSampler = None

	cmaesSampler = CmaEsSampler()
	OptunaSampler = cmaesSampler
	print('** Using cmaesSampler')

	
	CVNameMap = {'New_Zealand':    'New Zealand',
				'North_Macedonia': 'The former Yugoslav Republic of Macedonia',
				'Taiwan':          'Taiwan Province of China'}
	
	educLevelSize = {'h':4, 'w':20, 'c':20, 'sk': 20, 'sp': 20, 'ss': 40 , 'su': 80}
	# start_day = '2020-01-21'
	end_day =   '2020-07-31'

	parsCommon = {# 'start_day': start_day,
					'end_day':  end_day,
					'pop_size':  1e5, 
		    		'rand_seed': 1, 
		    		# NB: creating multi-education contact levels vs single school one
		   		 	'pop_type': 'educlevel', 
				   	'asymp_factor': 2,
					'contacts': educLevelSize,
					'rescale': True,
					'verbose': 0. # 0.1,

		}
	
	to_plot = ['cum_diagnoses', 'cum_deaths', 'cum_tests']
	
	# totPop data redundant and perhaps inconsistent
	# but needed to initialize sim: pars['tot_pop'] = tot_pop ?
	# 2do: replace!
	
	totPopFile = dataDir + 'data/pop19Total.csv'
	totPop = loadTotalPop(totPopFile)
	
	interveneFile = dataDir + 'intervene/educ_intervene-uniq.csv'
	educIntrvn = loadEducIntervn(interveneFile)
	
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

		if country not in totPop:
			print('* Missing pop total?!',country)
			continue

		if country not in educIntrvn:
			print('* Missing interventions?!',country)
			continue
		
		dbfile = f'{DBDir}{country}.db'
		if os.path.exists(dbfile):
			# print('DB %s exists; skipping' % dbfile)
			# continue
			print('DB %s exists; DELETING!' % dbfile)
			os.remove(dbfile)

		storage   = f'sqlite:///{dbfile}'

		datafile = ECDPDir + country + '.csv'
											
		pars = parsCommon.copy()
		
		# ... different start dates ... we find typically 50-500 gives good enough stability
		pars['start_day'] = findStartDate(datafile,50)
		
		tot_pop = totPop[country]
		
		# guarantee country is known as part of  covasim.data.country_age_data.get()	
		# Use location as CV-internal name
		# 	  country as externally consistent
		
# 		CVCountryNames = cad.get().keys()
# 		if country not in CVCountryNames:
# 			print('* Mapping country "%s" -> CV "%s"' % (country,CVNameMap[country]))
# 			pars['location'] = CVNameMap[country]
# 		else:
# 			pars['location'] = country
			
		pars['location'] = country
		pars['country'] = country
			
		pars['datafile'] = datafile
		pars['storage'] = storage
		pars['tot_pop'] = tot_pop
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

			educIntrvn = educIntrvn[country]
			
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
		pars, pkeys = get_bounds(CurrCtnyParams['pop_size']) # Get parameter guesses
		sim = create_sim(pars.best)
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
			
# 			pars_calib, study = calibrate(sampler=cmaesSampler)
			pars_calib, study = calibrate()
	
			sc.toc(T)
		
			# Plot result
			print('Plotting result...')
			if UseTestRate=='search':
				sim = create_sim([pars_calib['beta'], pars_calib['pop_infected'], pars_calib['test_rate']])
			else:
				sim = create_sim([pars_calib['beta'], pars_calib['pop_infected']])
				
			sim.run()
			sim.plot(to_plot=to_plot,do_save=True,fig_path=PlotDir + country +'-fit.png')
			pl.gcf().axes[0].set_title('Calibrated parameter values')
			
			
