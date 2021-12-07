''' jhu4covasim: import facility of JHU data

	This utility downloads current data from the COVID-19 Data Repository 
	maintained by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University
	https://github.com/CSSEGISandData/COVID-19 and makes it available for analysis
	by tools like Covasim.
	
	Data is downloaded to a directory specified by `JHUDataDir` together
	with a note about the date of download.  Only global (vs. USA-specific) data is captured.
	CUMMULATIVE data regarding confirmed/diagnosed cases, deaths, and recovered patients
	is captured.  
	
	`addJHUCounts()` derives per-day NEW counts from the cummulative counts provided by JHU
	
	Several other utilities massage this data towards use by Covasim.
	- `getTestData()` extracts a subset of data within specified begin and end dates,
		optionally providing a smoothed version
	- `makeDF()` transforms this data into the sort of pandas dataframe used by Covasim	
	
	Caveats:
	- country_converter used to normalize country names and provide ISO3 which used as keys
	- Several nominal countries (Diamond Princess, MS Zaandam, Summer Olympics 2020) aren't
	- Only (non-ImperiaList, see below) "province" country data is collected
	- JHU quit publishing recovered data as of Aug 4 2021; cf https://github.com/CSSEGISandData/COVID-19/issues/4465

	Created on Dec 6, 2021

@author: rbelew@ucsd.edu
'''

import csv
from collections import defaultdict
from datetime import date, datetime,timedelta
import json
import os
import requests

import country_converter as coco
import numpy as np
import pandas as pd


def gkern1d(l=5, sig=1.):
	"""
	https://stackoverflow.com/a/43346070/1079688
	creates gaussian kernel with side length l and a sigma of sig
	"""
	ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
	gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
	return gauss / np.sum(gauss)

GKern14 = gkern1d(l=14, sig=5.0)			

# to allow pickling/JSON serialization
def dd2dict(dd):
	# NB: dd must use only primitive types
	return json.loads(json.dumps(dd))


ImperiaList = ['Denmark','France','Netherlands','New Zealand','United Kingdom']

def loadJHUCountry(jhuDir,verbose=False):
	''' return mergeDict: iso3 -> date -> {metaData, data: dtype -> cumm count}
	
		convert country names via country_converter
		use ISO3 as key
		NB: skip "provinces" of ImperiaList countries
		collect raw CUMMULATIVE statistics; daily new COUNTS added in addJHUCounts() 
		
		JHU uses the "Province/State" column both for states WITHIN a country
		and for ~colonies/protectorates of other "ImperiaList" nations!?
		The former are subsumed/shared by countries' stats and summed into that country's numbers
		The latter are at great geographic distances from the "mother" country and need to be considered distinct. Skipped for now
	'''
	
	stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int))) # cnty -> date -> dtype -> n
	skipped = defaultdict(list)
	prevISO3 = ''
	for dtype in JHU_dataTypes:
		inf = JHUDataDir + dtype + '.csv'
		reader = csv.DictReader(open(inf))
		for i,entry in enumerate(reader):
			# Province/State,Country/Region,Lat,Long,01/22/20,01/23/20,01/24/20,...

			cnty = entry['Country/Region']
			
			if entry['Province/State'].strip() != '' and cnty in ImperiaList:
				skipInfo = f'{entry["Province/State"]} / {cnty}'
				skipped[dtype].append(skipInfo)
				continue			
						
			# standardize JHU country name
			shortName = coco.convert([cnty],to="name_short")
			iso3 = coco.convert([cnty],to="iso3")
			if shortName != cnty:
				officialName = coco.convert([cnty],to="name_official")
				# Only report update once
				if dtype=='confirmed':
					print(f'loadJHUCountry: ISO3={iso3} Name change from "{cnty}" to "{shortName}" official="{officialName}"')

				cnty = shortName

			for k in entry.keys():
				# 01/22/20, mm/dd/yy
				if k.count('/') != 2:
					continue
				
				dbits = k.split('/')
				try:
					sdate = date(int('20'+dbits[2]), int(dbits[0]), int(dbits[1]))
					# 211023: make dates strings for json
					sdateStr = sdate.strftime(RikDateFormat)
				except Exception as e:
					print('loadJHUCountry: bad date?!', dtype, cnty, i, k)
					sdateStr = None
				
				cumm = int(entry[k])
				# NB: maintain SPARSE stats only
				if cumm > 0:
					# NB: for non-ImperiaList lines, ADD sub-region to country total					
					stats[cnty][sdateStr][dtype] += cumm

		print(f'loadJHUCountry: ALL {dtype} done')
		
	print(f'# loadJHUCountry: Skipped provinces')
	for dtype in JHU_dataTypes:
		print(f'* {dtype}')
		for skipInfo in skipped[dtype]:
			print(f'\t{skipInfo}')
				

	## Add meta data
	mergeData = {}
	allCnty = sorted(list(stats.keys()))
	for cnty in allCnty:
		jhuData = stats[cnty]
		dateSet = set(jhuData.keys())
		startDateStr = min(dateSet)
		endDateStr = max(dateSet)
		startDate = datetime.strptime(startDateStr,RikDateFormat)
		endDate = datetime.strptime(endDateStr,RikDateFormat)
		duration = (endDate-startDate).days + 1
		iso3 = coco.convert([cnty],to="iso3")
		
		mrgInfo = {'startDate': startDateStr,
			'endDate': endDateStr,
			'duration': duration,
			'ndays': len(dateSet),
			'iso3': iso3,
			'country': cnty}
		
		allDates = sorted(list(dateSet))
		prevDate = None
		dataTbl = {}
		for sdate in allDates:
			jhuDay = jhuData[sdate]
			if 'confirmed' in jhuDay:
				confirmed = jhuDay['confirmed']
			else:
				confirmed = dataTbl[prevDate]['confirmed'] if prevDate != None else 0
				
			if 'deaths' in jhuDay:
				deaths = jhuDay['deaths']
			else:
				deaths = dataTbl[prevDate]['deaths'] if prevDate != None else 0

			if 'recovered' in jhuDay:
				recovered = jhuDay['recovered']
			else:
				recovered = dataTbl[prevDate]['recovered'] if prevDate != None else 0
				
			dataTbl[sdate] = {'confirmed': confirmed, 'deaths': deaths, 'recovered': recovered}
			prevDate = sdate
			
		mrgInfo['data'] = dataTbl
		mergeData[iso3] = mrgInfo
		
	# NB: to allow pickling
	mergeDict = dd2dict(mergeData)
	print('loadJHUCountry: NCountry=%d' % (len(mergeDict)))
	return mergeDict

def addJHUCounts(jhuMrg):
	'''Add per-day COUNTS from CUMMULATIVE data
	'''

	allISO3 = sorted(list(jhuMrg.keys()))
	update = {}
	for iso3 in allISO3:
		cinfo = jhuMrg[iso3]
		cummData = cinfo['data']
		counts = {}
		negCount = defaultdict(int)
		allDates = sorted(list(cummData.keys()))
		for di,sdate in enumerate(allDates):
			if di==0:
				counts[sdate] = cummData[sdate].copy()
				continue
			prevDate = allDates[di-1]
			
			counts[sdate] = {}		
			for dtype in JHU_dataTypes:
				newCnt = cummData[sdate][dtype] - cummData[prevDate][dtype]
				# NB: negative counts disallowed
				if newCnt < 0:
					negCount[dtype] += 1
					newCnt = 0.
				counts[sdate][dtype] = newCnt
		cinfo['counts'] = counts
		update[iso3] = cinfo
		for dtype in JHU_dataTypes:
			if negCount[dtype] > 0:
				print(f'addJHUCounts: {iso3} has {negCount[dtype]} negative change in {dtype} cumm stats?')
	return update

def getTestData(iso3,isodata,modeIdx=0,partInfo=None,smooth=True):
	'''return pop19, ndays, deathVec, casesVec, recoverVec	
	if SMOOTH, apply to all vecs 
	v2: modeDates = partInfo dict
	returns empty dict {} on error
	'''
	
	pop19 = isodata['pop19']
	
	allDates = sorted(list(isodata['data'].keys()))

	if partInfo=={}:
		# Used to flag incomplete partitions
		return {}
	elif partInfo == None:
		startDate = allDates[0]
		endDate = allDates[-1]
	else:
		startDate = partInfo['sdate']
		maxDate = partInfo['maxdate']
		endDate = partInfo['edate']
		subDates = []
		for d in allDates:
			if d < startDate:
				continue
			if d > endDate:
				break
			subDates.append(d)
		allDates = subDates
	
	casesList = []
	deathList = []
	recoverList = []
	for id,d in enumerate(allDates):
		stats = isodata['counts'][d]

		casesList.append(float(stats['confirmed']))
		deathList.append(float(stats['deaths']))
		recoverList.append(float(stats['recovered']))		
						
	assert len(casesList)==len(deathList), 'mismatched infect/deaths?!'
			
	ndays = len(casesList)
	
	casesVec = np.ravel(casesList)	
	deathVec = np.ravel(deathList)
	recoverVec = np.ravel(recoverList)
	
	if smooth:
		smoothCase = np.convolve(casesVec, GKern14, mode='same')
		casesVec = smoothCase
		smoothDeath = np.convolve(deathVec, GKern14, mode='same')
		deathVec = smoothDeath
		smoothRecover = np.convolve(recoverVec, GKern14, mode='same')
		recoverVec = smoothRecover

	if casesVec[0] == 0:
		print(f'getTestData: {iso3}_{modeIdx} initCases==0!? startDate={startDate}')
		return {}
	
	print(f'getTestData: {iso3}_{modeIdx} startDate={startDate} endDate={endDate} ndays={ndays}')
	initCumm = isodata['data'][allDates[0]]

	# NB: anything in this history must be consumable by SQL;
	
	return {'pop19': pop19, 
			'startDate': startDate,
			'endDate': endDate,
			'maxDate': maxDate,
			'ndays': ndays, 
			'deaths': np.array(deathVec), 
			'cases': np.array(casesVec),
			'recover': np.array(recoverVec),
			'initCases':  initCumm['confirmed'],
			'initDeaths': initCumm['deaths'],
			'initRecover':  initCumm['recovered'],
			 }

def makeDF(testData):
	''' convert testData to dataframe corresponding to Covasim requirements
	cf. https://docs.idmod.org/projects/covasim/en/latest/tutorials/tut_people.html#Data-input-and-formats
	''' 
	
# 	egfile = '/System/Volumes/Data/rikData/doc/covasimV3/tutorials/example_data.csv'
# 	egdf = pd.read_csv(egfile)

	ndays = testData['ndays']
	pop_size = testData['pop19']
	startDate = datetime.strptime(testData['startDate'],RikDateFormat)

	dateVec = pd.date_range(start=startDate,periods=ndays)
	dateStrVec = [str(dv) for dv in dateVec]
	
	dataDict = {'date': dateVec, # str(dateStrVec),
				'new_diagnoses': testData['cases'],
				'new_deaths':	 testData['deaths']}
	
	df = pd.DataFrame(dataDict)

	return df


JHU_dataTypes = ['confirmed', 'deaths', 'recovered']

if __name__ == '__main__':
	RikDateFormat = '%Y-%m-%d'
	JHUDataDir = '/System/Volumes/Data/rikData/coviData/data/JHU2/'
	JHU_git_root = 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/'
	JHU_fileTemplate = 'time_series_covid19_%s_global.csv'
	
	if not os.path.exists(JHUDataDir):
		print(f'creating new directory {JHUDataDir}')
		os.mkdir(JHUDataDir)
		
# 	for dtype in JHU_dataTypes:
# 		fullURL = JHU_git_root + JHU_fileTemplate % dtype
# 		response = requests.get(fullURL)
# 		content = response.content.decode('iso-8859-1')
# 		outs = open(JHUDataDir+f'{dtype}.csv','w')
# 		outs.write(content)
# 		outs.close()
# 		print(f'{dtype} retrieved')
	
	outs = open(JHUDataDir+'README.txt','w')
	now = datetime.now()
	msg = f'Johns Hopkins CSSE COVID-19 Dataset harvested from {JHU_git_root} on {str(now)}.\n'
	outs.write(msg)
	outs.close()

# 	jhuDict = loadJHUCountry(JHUDataDir,verbose=True)
	
# 	jhuJSONTMPFile = JHUDataDir + 'jhuDataTMP.json'
# 	jhuDict = json.load(open(jhuJSONTMPFile))
	
	# jhuDict2 = addJHUCounts(jhuDict)
	
	jhuJSONFile = JHUDataDir + 'jhuData.json'
# 	json.dump(jhuDict2,open(jhuJSONFile,'w'))
	jhuDict2 = json.load(open(jhuJSONFile))

	## Example extracting a subset of days from Argentina's data
	ARG2_full = jhuDict2['ARG']
	ARG2_full['pop19'] = 44780675 # from ECDC
	
	modeIdx = 2
	ARG2_dates = {'sdate': '2021-03-02', 'maxdate': '2021-06-23', 'edate': '2021-10-19', 'deathMax': '2021-06-20', 'recoverMax': '2021-07-06'}
	
	ARG2_sub = getTestData('ARG',ARG2_full,modeIdx,ARG2_dates)
	
	# Create pandas dataframe, suitable for use by Covasim
	# cf. https://docs.idmod.org/projects/covasim/en/latest/tutorials/tut_people.html#Data-input-and-formats
	# but build the pandas dataframe directly
	covasimDF = makeDF(ARG2_sub)
	print(covasimDF)
	
	
