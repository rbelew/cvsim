from datetime import date, datetime,timedelta
import json

import numpy as np
import pandas as pd
import sciris as sc

import covasim as cv

def makeDF(waveData):
	''' convert waveData to dataframe corresponding to Covasim requirements
	''' 
	
	ndays = waveData['ndays']
	pop_size = waveData['pop19']
	startDate = datetime.strptime(waveData['startDate'],CVDateFormat)

	dateVec = pd.date_range(start=startDate,periods=ndays)
	dateStrVec = [str(dv) for dv in dateVec]

	dataDict = {'date': dateVec} 
	dataDict['new_diagnoses'] = waveData['cases']
	dataDict['new_deaths'] = waveData['deaths']
	
	df = pd.DataFrame(dataDict)

	return df

class ABCObjective(object):
	def __init__(self,iso3,modeIdx,runLbl,waveData):
				
		self.iso3 = iso3
		self.modeIdx = modeIdx
		self.runLbl = runLbl

		self.waveData = waveData 			
		self.startDate = self.waveData['startDate']
		self.endDate = self.waveData['endDate']
		self.ndays = self.waveData['ndays']
		self.pop_size = self.waveData['pop19']
		self.df = makeDF(waveData)

class Store_seird(cv.Analyzer):
	# https://github.com/InstituteforDiseaseModeling/covasim/blob/master/docs/tutorials/tut_analyzers.ipynb

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.t = []
		self.S = []
		self.E = []
		self.I = []
		self.R = []
		self.D = []
		return

	def apply(self, sim):
		# NB: simulation results are expressed in people 
		# This analyzer counts number of agents!		
		# https://github.com/amath-idm/covasim/issues/1355
		agents = sim.people # Shorthand
		self.t.append(sim.t)
		self.S.append(agents.susceptible.sum())
		self.E.append(agents.exposed.sum() - agents.infectious.sum())
		self.I.append(agents.flows['new_diagnoses'])
		self.R.append(agents.flows['new_recoveries'])
		self.D.append(agents.flows['new_deaths'])
		return

def testingDemo(waveData):

	# allDataKeys = ['pop19', 'startDate', 'endDate', 'ndays', 'deaths','cases','recover', 'initCases', 'initDeaths','initRecover','cummData', 'smoothed']
	for k in ['pop19', 'startDate', 'endDate', 'ndays','initCases']:
		print(k,waveData[k])

	abcObj = ABCObjective('ARG',2,'testingDemo',waveData)
	abcObj.popScale = float(abcObj.pop_size) / CovasimNAgents
	initInfect = abcObj.waveData['initCases'] / abcObj.popScale

	runPars = {'beta': 0.02,
			   'mu': 0.11,
			   'ntst': 20000,
			   'sympOR': 250}

	simPars = sc.objdict(
		pop_size	= CovasimNAgents,
		pop_scale	= abcObj.popScale,
		rescale	 = True,
		# NB: initial infected need to be scaled to simulation size
		pop_infected = initInfect,
		start_day	= abcObj.startDate,
		# NB: need to decrement ndays for covasim?
		n_days		= abcObj.ndays - 1,
		
		# 210927: need to use hybrid
		#	Warning; not loading household size for "Ethiopia" since no "h" key; keys are "a". Try "hybrid" population type?
		pop_type = 'hybrid',
		verbose = False,

		beta           = runPars['beta'],
		rel_death_prob = runPars['mu']
	)

	allIntervnt = []
	
	tp = cv.test_prob(symp_prob=1.0, asymp_prob=1.0, start_day=abcObj.startDate)

	tn_fixed = cv.test_num(daily_tests=runPars['ntst'], symp_test=runPars['sympOR'], start_day=abcObj.startDate)

	allIntervnt.append(tn_fixed)

	sim = cv.Sim(simPars,datafile = abcObj.df, analyzers=Store_seird(label='seird'),interventions=allIntervnt)
	
	print(f'testingDemo: FIXED: pop_size={abcObj.pop_size:e} simScale={sim["pop_scale"]} initInfect={initInfect} ')
	
	sim.run()

	cvaFit = sim.compute_fit()
	cvaMismatch = cvaFit.mismatch
		
	seirdLists = sim.get_analyzer('seird')
			
	# NB: simulation results are expressed in people 
	# The analyzer counts number of agents!		
	# https://github.com/amath-idm/covasim/issues/1355
	# convert analyzer's results to simple dict of np.arrays, factor in scaling
	slDict = seirdLists.__dict__
	seirVec = {}
	for k in ['S','E','I','R','D']:
		a = np.array(slDict[k])
		scaled = a * abcObj.popScale
		seirVec[k] = scaled


	cvaPlotVar = ['new_tests',
			 	'n_infectious',
			 	'new_infections',
			 	'new_diagnoses',
			 	'new_recoveries',
			  	'n_removed',
			  	'new_deaths',
			  	'r_eff']

	simPlotFile = DataDir + 'simPlot.png'
	sim.plot(cvaPlotVar,do_show=False,do_save=True,fig_path=simPlotFile)

CVDateFormat = '%Y-%m-%d'
CovasimNAgents = 1e5
DataDir =  '/YOUR_PATH_HERE/wave-ARG_2-data/'

# def main():
waveDataFile = DataDir + 'waveSmooth-ARG_2.json'
waveData = json.load(open(waveDataFile))

testingDemo(waveData)
	
