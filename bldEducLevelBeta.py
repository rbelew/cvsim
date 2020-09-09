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
