def init():
    global th_ZoneSelection
    th_ZoneSelection = 12000

    global nr_zones_to_connect_tt
    nr_zones_to_connect_tt = 2

    global nr_stations_to_connect_euclidean
    nr_stations_to_connect_euclidean = 3

    global  min_nr_of_passenger
    min_nr_of_passenger = 2

    print('\n treshhold Zone Selection [m] : ', th_ZoneSelection)
    print(' number of min passengers  : ', min_nr_of_passenger)
    print(' nr stations connected euclidean : ', nr_stations_to_connect_euclidean)
    print(' nr stations connected TT  : ', nr_zones_to_connect_tt)

def paths():
    # paths
    global path_nodesSBB
    path_nodesSBB = 'input/Node_SBB_ZoneNR.csv'

    global p_allZonesOD_debug
    p_allZonesOD_debug = 'input/zones_debug.csv'

    global p_centroidZones
    p_centroidZones = 'input/centroids_zones.csv'

    global p_centroidZonesSel
    p_centroidZonesSel = 'input/zonesSelected_' + str(th_ZoneSelection) + '.csv'

    global p_allZonesOD
    p_allZonesOD = 'input/01_DWV_OEV.txt'

    global p_allZonesTT
    p_allZonesTT = 'input/2010_TravelTime_allZones.txt'

    global p_OD_selectedZones
    p_OD_selectedZones = 'input/Demand_DWV_' + str(min_nr_of_passenger) + 'pass_' + str(th_ZoneSelection) + '_zTT_' \
                         + str(nr_zones_to_connect_tt) + '_sEu_' + str(nr_stations_to_connect_euclidean) + '.csv'

    global p_OD_selectedZonesTT
    p_OD_selectedZonesTT = 'input/zonesSelected_tt_' + str(th_ZoneSelection) + '.csv'

    global p_OD_selectedZonesTT_internal
    p_OD_selectedZonesTT_internal = 'input/zonesSelected_tt_' + str(th_ZoneSelection) + '_internal.csv'

