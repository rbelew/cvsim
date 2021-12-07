''' jhu4covasim: prototype import facility of JHU GLOBAL cases, deaths and recovered data
	for inclusion in Covasim?

	- country_converter used to normalize country names and provide ISO3 used as keys
	- Several nominal countries (Diamond Princess, MS Zaandam, Summer Olympics 2020) aren't
	- Only (non-Imperialist, see below) "province" country data is collected
	- JHU quit publishing recovered data as of Aug 4 2021; cf https://github.com/CSSEGISandData/COVID-19/issues/4465

	Created on Dec 6, 2021

@author: rik
'''

import csv
from collections import defaultdict
from datetime import date, datetime,timedelta
import json
import os
import requests

import country_converter as coco

# to allow pickling/JSON serialization
def dd2dict(dd):
	# NB: dd must use only primitive types
	return json.loads(json.dumps(dd))


# JHU uses the "Province/State" column both for states WITHIN a country
# and for ~colonies/protectorates of other "ImperiaList" nations!?
# The former are subsumed/shared by countries' stats and summed into that country's numbers
# The latter are at great geographic distances from the "mother" country and need to be considered distinct. Skipped for now

ImperiaList = ['Denmark','France','Netherlands','New Zealand','United Kingdom']

def loadJHUCountry(jhuDir,verbose=False):
	''' return mergeDict: iso3 -> date -> {metaData, data: dtype -> cumm count}
	
		convert country names via country_converter
		use ISO3 as key
		NB: skip "provinces" of ImperiaList countries
		collect raw CUMMULATIVE statistics; daily new COUNTS added in addJHUCounts() 
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
			
			# NB: negative counts disallowed			
# 			counts[sdate] = {dtype: max(0., (cummData[sdate][dtype] - cummData[prevDate][dtype])) for dtype in JHU_dataTypes}
			for dtype in JHU_dataTypes:
				newCase = cummData[sdate][dtype] - cummData[prevDate][dtype]
				if newCase < 0:
					negCount[dtype] += 1
					newCase = 0.
				
# 				
# 				'cases': max(0., (cummData[sdate]['cases'] - cummData[prevDate]['cases'])), 
# 							'deaths': max(0., (cummData[sdate]['deaths'] - cummData[prevDate]['deaths'])), 
# 							'recover': max(0., (cummData[sdate]['recover'] - cummData[prevDate]['recover']))}
		cinfo['counts'] = counts
		update[iso3] = cinfo
		for dtype in JHU_dataTypes:
			if negCount[dtype] > 0:
				print(f'addJHUCounts: {iso3} has {negCount[dtype]} negative change in {dtype} cumm stats?')
	return update


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

	jhuDict = loadJHUCountry(JHUDataDir,verbose=True)
	
# 	jhuJSONTMPFile = JHUDataDir + 'jhuDataTMP.json'
# 	jhuDict = json.load(open(jhuJSONTMPFile))
	
	jhuDict2 = addJHUCounts(jhuDict)
	
	jhuJSONFile = JHUDataDir + 'jhuData.json'
	json.dump(jhuDict2,open(jhuJSONFile,'w'))
