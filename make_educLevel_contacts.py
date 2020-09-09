def make_educLevel_contacts(pop_size, ages, contacts):
    '''
    Create "educLevel" contacts -- similar to 'hybrid' with
    microstructured contacts for households and random contacts for workplaces, community 
    
    but SPLIT school into kindergarten,primary,secondary,university age groups
    ASSUME: increasingly large groups in secondary, university levels
    
    supports interventions specific to these eductional levels
    '''

    layer_keys = ['h', 'w', 'c', 'sk', 'sp', 'ss', 'su']
    # ASSUME: increasingly large groups in secondary, university levels
    educLevelSize = {'h':4, 'w':20, 'c':20, 'sk': 20, 'sp': 20, 'ss': 40 , 'su': 80}
    
    contacts = educLevelSize
    
    educLevel_ages = {'sk': [4,5],
                      'sp': [6,10],
                      'ss': [11,18],
                      'su': [19,24]}
    
    work_ages   = [25, 65]

    # Create the empty contacts list -- a list of {'h':[], 's':[], 'w':[]}
    contacts_list = [{key:[] for key in layer_keys} for i in range(pop_size)]

    # Start with the household contacts for each person
    h_contacts, _, clusters = make_microstructured_contacts(pop_size, {'h':contacts['h']})

    # Make community contacts
    c_contacts, _ = make_random_contacts(pop_size, {'c':contacts['c']})

    # Get the indices of people in each age bin
    ages = np.array(ages)
    
    w_inds = sc.findinds((ages >= work_ages[0])   * (ages < work_ages[1]))

    elev_inds = {}
    for elev in educLevel_ages.keys():
        elev_inds[elev] = sc.findinds((ages >= educLevel_ages[elev][0]) * (ages < educLevel_ages[elev][1]))
        

    # Create the school and work contacts for each person
    w_contacts, _ = make_random_contacts(len(w_inds), {'w':contacts['w']})

    elev_contacts = {}
    for elev in educLevel_ages.keys():
        elev_contacts[elev], _ = make_random_contacts(len(elev_inds[elev]), {elev:contacts[elev]})

    # Construct the actual lists of contacts
    
    # Copy over household contacts -- present for everyone
    for i in range(pop_size):   contacts_list[i]['h']   =        h_contacts[i]['h']  

    # Copy over work contacts
    for i,ind in enumerate(w_inds):
        foo =  w_contacts[i]['w']
        contacts_list[ind]['w'] = w_inds[ foo ]
        
    for i in range(pop_size):   contacts_list[i]['c']   =        c_contacts[i]['c']  # Copy over community contacts -- present for everyone

     # Copy over school contacts
    for elev in educLevel_ages.keys():
        for i,ind in enumerate(elev_inds[elev]): 
            foo = elev_contacts[elev][i][elev]
            contacts_list[ind][elev] = elev_inds[elev][ foo ]

    return contacts_list, layer_keys, clusters

