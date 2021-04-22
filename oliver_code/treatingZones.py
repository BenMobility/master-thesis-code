import utils
import dataReader
import numpy as np

import numpy.lib.recfunctions as rfn
# import plots


def treating_reading_demand_zones_tt(min_nr_of_passenger, nodesSBB, nr_stations_to_connect_euclidean, nr_zones_to_connect_tt,
                p_OD_selectedZones, p_OD_reducedTT, p_allZonesOD, p_allZonesTT,
                p_centroidZones, p_centroidZonesSel, select_zones, th_ZoneSelection):
    if select_zones:  # Create from original file
        zoneCentroid = dataReader.dataReaderZones_original(p_centroidZones)
        # select a subset of zones
        zonesSelected = findClosestStation(zoneCentroid, nodesSBB, th_ZoneSelection,
                                                         saveAsZonesSelected=True)
        # select demand of zones
        zonesSelDemand = selectDemandofZones(p_allZonesOD, zonesSelected, min_nr_of_passenger,
                                                           th_ZoneSelection, nr_zones_to_connect_tt,
                                                           nr_stations_to_connect_euclidean, saveoutput=True)

        # select travel times
        zonesSelTT = selectTTofZones(p_allZonesTT, zonesSelected, th_ZoneSelection, nr_stations_to_connect_euclidean,
                                     nr_zones_to_connect_tt, nodesSBB, saveoutput=True)


    else:  # Load existing subset of zones and demand
        zonesSelected = dataReader.dataReaderZones_load(p_centroidZonesSel)
        # names: zone, xcoord, ycoord

        zonesSelDemand = dataReader.dataReaderOD_load(p_OD_selectedZones)
        # names: fromZone, toZone, passenger
        # with open(path_allZonesTT, "r") as f:
        #   lines = f.readlines()
        # with open(path_allZonesTT, "w") as f:
        #   for line in lines:
        #       res = [int(i) for i in line.strip("\n").split() if i.isdigit()]
        #       if res[0] in zonesSelected.zone and res[1] in zonesSelected.zone:
        #           f.write(line)

        # zonesSelTT = selectTTofZones(path_allZonesTT, zonesSelected, th_zone_selection, nr_stations_to_connect_euclidean,
        #                             nb_zones_to_connect, nodesSBB, saveoutput=True)

        zonesSelTT = dataReader.dataReaderTT_load(p_OD_reducedTT)

        zonesSelTT[zonesSelTT.fromZone == 1]
        # zonesSelTT.dtype.names = 'fromZone', 'toZone', 'tt'
        # zonesSelTT_internal = dataReader.dataReaderOD_load(p_OD_selectedZonesTT_internal)
        # zonesSelTT_internal.dtype.names = 'fromZone', 'toZone', 'tt'

    return zonesSelected, zonesSelDemand, zonesSelTT


def findClosestStation(zones, stations, threshold, saveAsZonesSelected = False):

    '''
    :param zones: home zones with names ; zone, x,y
    :param stations: sbb stations with names : zone, SbbID, ViriatoID, NodeName, xcoord, ycoord, Code
    :param threshold: distance in meters to consider the zone or not
    :return: all zones within threshold distance of a station
    '''
    print('\n start finding all stations within threshold of :', threshold, ' [m]')
    checkdebug = False
    # calculate the distance from all zones to every staion
    xyzones = [zones.xcoord, zones.ycoord]
    xystations = [stations.xcoord, stations.ycoord]

    distance = utils.distance_euclidien(xyzones, xystations)

    minDistance = [np.amin(distance, axis = 1), np.argmin(distance, axis = 1)]
    minDistance = np.asarray(minDistance)

    zonesSelected = zones
    # loop trough all zones

    de = 0
    nr = 0
    deleteIndeces = np.zeros(1)
    selectIndeces = np.zeros(1)
    selectDistance = np.zeros(1)

    for i in range(0, minDistance.shape[1]):
        if minDistance[0, i] <= threshold:
            # select this zone
            nr = nr + 1
            if nr == 1:
                selectIndeces[0] = i
                selectDistance[0] = minDistance[0, i]
            else:
                selectIndeces = np.append(selectIndeces, i)
                selectIndeces = selectIndeces.astype(int)
                selectDistance = np.append(selectDistance, minDistance[0, i])

        else:
            # delete this zone
            de = de + 1
            if de == 1:
                deleteIndeces[0]=i
            else:
                deleteIndeces = np.append(deleteIndeces, i)

            deleteIndeces = deleteIndeces.astype(int)

    zonesDeleted = zonesSelected[deleteIndeces]
    zonesSelected = zonesSelected[selectIndeces]

    a = np.reshape(zonesSelected.zone, (zonesSelected.shape[0], 1))
    b = np.reshape(zonesSelected.xcoord, (zonesSelected.shape[0], 1))
    c = np.reshape(zonesSelected.ycoord, (zonesSelected.shape[0], 1))

    zonesSel_4output = np.concatenate((a, b, c, np.reshape(selectDistance, (selectDistance.shape[0], 1))), axis=1)

    zonesSel_4output = np.core.records.fromarrays(zonesSel_4output.transpose(),
                                                  names='zone, xcoord, ycoord, distance',
                                                  formats='i8, f8, f8, f8')
    # print(zonesSel_4output.distance)
    if checkdebug:
        rz = np.random.randint(2000)
        rs = np.random.randint(300)
        xystations = np.asarray(xystations)
        xyzones = np.asarray(xyzones)
        xyzones = xyzones[:, rz]
        xystations= xystations[:, rs]

        xystations = np.reshape(xystations, (1, 2))
        xyzones = np.reshape(xyzones, (1, 2))

        print(xyzones)
        print(xystations)

        distanceDC = utils.distance_euclidien(xyzones, xystations)
        print('Distance to debug :   From Station Index  ', rz, '  To Zone Index', rs, '\n'
              , distanceDC, ' m')

    if saveAsZonesSelected:
        np.savetxt('input/zonesSelected_'+ str(threshold) +'.csv', zonesSel_4output, delimiter=',', header="zone,x,y, distanceToStation", fmt="%s,%f,%f,%f")
        np.savetxt('input/zonesDELETED_'+ str(threshold) +'.csv', zonesDeleted, delimiter=',', header="zone,x,y", fmt="%s,%f,%f")
    print('number of zones selected = ', nr, '/', zones.shape[0]+1,  '\n')

    return zonesSel_4output


def selectDemandofZones(path, selectedZones, minimal_passengers, threshold,
                        nr_zones_to_connect_tt, nr_stations_to_connect_euclidean, saveoutput=False):
    '''
    :param path: to the OD File with all zones
    :param selectedZones: zones to be considered, output of findClosestStation method
    :param saveoutput: if true, file saved as 'od_selectedZones_DWV.csv'
    :param threshold: threshold for zone selection
    :param minimal_passengers: to consider demand
    :param nr_zones_to_connect_tt: number of connected zones with stations to home zones
    :param nr_stations_to_connect_euclidean: nr of stations connected by euclidean distance
    :return: OD of a subset of all zones

    '''
    read_whole_distance_file = False
    if read_whole_distance_file:
        od_distance = dataReader.dataReaderOD_original('input/2010_Distanz_Systemfahrplan_B.txt')
        od_distance.dtype.names = 'fromZone', 'toZone', 'distance'
        # trips in area of interest
        inzones = np.logical_and(np.isin(od_distance.fromZone, selectedZones.zone), np.isin(od_distance.toZone, selectedZones.zone))
        od_distance = od_distance[inzones]
        filename = 'input/zonesSelected_distance_reduced' + str(threshold) + '.csv'
        np.savetxt(filename, od_distance, delimiter=',', header="fromZone, toZone, min", fmt="%i,%i,%f")
    else:
        od_distance = dataReader.dataReaderTT_load('input/zonesSelected_distance_reduced' + str(threshold) + '.csv')

    od = dataReader.dataReaderOD_original(path)
    print('\n Demand of Zone Selection starts')
    print('  Total number of passenger trips is : ', od.shape[0])
    # plots.BoxplotPass(od.passenger)

    print('  Median amount of passenger per trip', np.median(od.passenger))
    # names='fromZone, toZone, passenger'
    total_passengers = np.sum(od.passenger)
    print(' Total number of passengers is : ', total_passengers)

    inzones = np.logical_and(np.isin(od.fromZone, selectedZones.zone), np.isin(od.toZone, selectedZones.zone))
    od_sel = od[inzones]

    tripsinArea = od_sel.shape[0]
    print('  trips within area of interest : ', tripsinArea)
    print('  Median pas/trip are of interest: ', np.median(od_sel.passenger))
    passenger_iAoI = np.sum(od_sel.passenger)
    print('  Number of passengers in AoI is : ', passenger_iAoI)
    size = od_sel.shape[0]

    # remove trips with less than th passengers
    # minmPass = od_sel.passenger >= minimal_passengers
    # od_sel = od_sel[minmPass]
    # print('  trips with more than ', minimal_passengers, ' passengers : ', od_sel.shape[0], ' / ', tripsinArea)
    # print('  Median pas/trip more then min Pass ', np.median(od_sel.passenger))
    # print('  Number of passengers after removing is : ', np.sum(od_sel.passenger), ' of ', passenger_iAoI)

    # remove internal trips
    internal = np.logical_and(od_sel.fromZone == od_sel.toZone, od_sel.toZone == od_sel.fromZone)
    od_internal = od_sel[internal]
    od_sel = od_sel[internal != True]

    print('  internal trips : ', size - od_sel.shape[0])  # zero is equal to false
    print('  trips without internals : ', od_sel.shape[0], ' / ', tripsinArea)
    print('  Median pas/trip wo. internals: ', np.median(od_sel.passenger), '\n')
    print('  Number of passengers without internal is : ', np.sum(od_sel.passenger), ' of ', passenger_iAoI)

    distance = np.zeros(od_sel.shape[0], dtype=[('distance30km', 'f8')])
    od_fromZone = np.array(od_sel.fromZone, dtype=[('fromZone', 'i8')])
    od_toZone = np.array(od_sel.toZone, dtype=[('toZone', 'i8')])
    od_pass = np.array(od_sel.passenger, dtype=[('passenger', 'f8')])
    od_sel = rfn.merge_arrays((od_fromZone, od_toZone, od_pass, distance), asrecarray=True)

    index_of_od_in_od_distance = 0
    for od in od_sel:
        # index = np.where(od_distance.fromZone == od.fromZone)
        # index2 = np.where(od_distance.toZone == od.toZone)
        # find the distance of current od
        # index = np.where((od_distance.fromZone == od.fromZone) & (od_distance.toZone == od.toZone))
        notFound = True
        while notFound:
            od_dist = od_distance[index_of_od_in_od_distance]
            if (od_distance[index_of_od_in_od_distance].fromZone == od.fromZone) &\
                    (od_distance[index_of_od_in_od_distance].toZone == od.toZone):
                if od_distance[index_of_od_in_od_distance].distance > 30000:
                    od.distance = 1
                    index_of_od_in_od_distance = index_of_od_in_od_distance + 1
                    notFound = False
                    continue
                else:
                    notFound = False
                    index_of_od_in_od_distance = index_of_od_in_od_distance + 1
                    continue
            else:
                index_of_od_in_od_distance = index_of_od_in_od_distance + 1
        # subset = od_distance[np.in1d(od_distance.fromZone, od.fromZone)]
        # subset = subset[np.in1d(subset.toZone, od.toZone)]
        # subset = od_distance[index]

        # if subset.distance > 30000:
        #    od.distance = 1

    filename = 'input/Demand_DWV_' + str(minimal_passengers) + 'pass_' + str(threshold) +\
               '_zTT_' + str(nr_zones_to_connect_tt) + '_sEu_'+str(nr_stations_to_connect_euclidean)+'.csv'
    np.savetxt(filename, od_sel, delimiter=',', header="fromZone, toZone, passenger, distance > 30 km", fmt="%i,%i,%f, %i")

    filename_int = 'input/Demand_DWV_' + str(minimal_passengers) + 'pass_' + str(threshold) + \
                   '_zTT_' + str(nr_zones_to_connect_tt) + '_sEu_' + str(nr_stations_to_connect_euclidean) + '_internal.csv'
    np.savetxt(filename_int, od_internal, delimiter=',',
               header="zone, fromZone, toZone", fmt="%i,%i,%f")

    return od_sel


def get_all_k_closest_stations_to_zone(nodesSBB, zonesSelected, thNrofZones):
    k = thNrofZones
    distance = distanceToStations(zonesSelected, nodesSBB)
    list_of_stations_euclidean = []
    for i in range(0, zonesSelected.shape[0]):  # loop trough all home zone
        # identify 3 closest stations euclidean distance
        idx = np.argpartition(distance[i, :], k)
        origin_zone = zonesSelected.zone[i]
        stations_zone_euclidean = nodesSBB.zone[idx[0:k]]
        stationID = nodesSBB.ViriatoID[idx[0:k]]
        station_code = nodesSBB.Code[idx[0:k]]
        for j in range(0, stationID.shape[0]):
            list_of_stations_euclidean.append([origin_zone, stations_zone_euclidean[j], stationID[j], station_code[j]])

    return list_of_stations_euclidean


def selectTTofZones(path, selectedZones, threshold, threshhold_euclidean, threshold_tt, nodesSBB, saveoutput=False):
    '''
    :param path: to the TT File with all zones
    :param selectedZones: zones to be considered, output of findClosestStation method
    :param threshold: distance to consider a zone or not for zone selection
    :param saveoutput: if true, file saved as 'od_selectedZones_DWV.csv'
    :return: OD traveltime of a subset of all zones
    '''
    k2 = threshold_tt + 2  # get the traveltime to the k + 2 closest stations
    stationID = []
    tt_array_reduced = []
    od = dataReader.dataReaderOD_TT_original(path)
    od.dtype.names = 'fromZone', 'toZone', 'tt'
    print('\n TT of Zone Selection starts \n')
    # names='fromZone, toZone, tt'

    # trips in area of interest
    inzones = np.logical_and(np.isin(od.fromZone, selectedZones.zone), np.isin(od.toZone, selectedZones.zone))
    od_sel = od[inzones]
    tripsinArea = od_sel.shape[0]
    # print(' Median pas/trip are of interest: ', np.median(od_sel.passenger))

    # remove internal trips
    # internal = np.logical_and(od_sel.fromZone == od_sel.toZone, od_sel.toZone == od_sel.fromZone)
    # od_internal = od_sel[internal]
    # od_sel = od_sel[internal != True]
    # remove all stations without commercial stop
    nodesSBB = nodesSBB[nodesSBB.commercial_stop == 1]
    # filter the travel time matrix and select and save only the needed tt's
    initializer = True
    for zone in selectedZones.zone:
        # add closest according to travelTime
        # select all TT fromZone to allZones
        subArrayTravelTime = od_sel[od_sel.fromZone == zone]
        # select all TT fromZone to stationZone
        subArrayTravelTime = subArrayTravelTime[np.isin(subArrayTravelTime.toZone, nodesSBB.zone)]
        # select closest stationZones from fromZone
        index_closestTravelTime = np.argpartition(subArrayTravelTime.tt, k2)[0:k2]
        subArrayTravelTime = subArrayTravelTime[index_closestTravelTime]
        # select all stations within closest zones
        stations2 = nodesSBB.ViriatoID[
            np.isin(nodesSBB.zone, subArrayTravelTime.toZone)]  # subArrayTravelTime.toZone == nodesSBB.zone]
        # append ID
        stationID = np.append(stationID, stations2)
        # tt_array_reduced.append(subArrayTravelTime)

        # bring it together in a nice form to save afterwards as textfile
        a = np.reshape(subArrayTravelTime.fromZone, (subArrayTravelTime.shape[0], 1))
        b = np.reshape(subArrayTravelTime.toZone, (subArrayTravelTime.shape[0], 1))
        c = np.reshape(subArrayTravelTime.tt, (subArrayTravelTime.shape[0], 1))

        if initializer:
            tt_array_reduced = np.concatenate((a, b, c), axis=1)
            initializer = False
        else:
            abc = np.concatenate((a, b, c), axis=1)
            tt_array_reduced = np.append(tt_array_reduced, abc, axis=0)

        # origin_destination_to_station_nodes_edges_creator(nodesSBB, tt_array_reduced, weights, zones_for_nodes)

    # get also the tt of the closest zones by euclidean distance and add the tt to the reduced file
    closest_stations_to_zone = get_all_k_closest_stations_to_zone(nodesSBB, selectedZones, threshhold_euclidean)
    for close_stations in closest_stations_to_zone:
        origin_zone = close_stations[0]
        station_zone = close_stations[1]
        stationID = close_stations[2]
        # np.where(origin_zone)
        subArrayTravelTime = od_sel[od_sel.fromZone == origin_zone]
        subArrayTravelTime = subArrayTravelTime[subArrayTravelTime.toZone == station_zone]
        tt_origin_station_zone = subArrayTravelTime.tt
        # bring it together in a nice form to concatenate
        a = np.reshape(origin_zone, (subArrayTravelTime.shape[0], 1))
        b = np.reshape(station_zone, (subArrayTravelTime.shape[0], 1))
        c = np.reshape(tt_origin_station_zone, (subArrayTravelTime.shape[0], 1))
        abc = np.concatenate((a, b, c), axis=1)
        tt_array_reduced = np.append(tt_array_reduced, abc, axis=0)

    tt_array_reduced = np.core.records.fromarrays(tt_array_reduced.transpose(),
                                    names='fromZone, toZone, tt',
                                    formats='i8, i8, f8')

    if saveoutput:
        # filename = 'input/zonesSelected_tt_' + str(threshold) + '.csv'
        # np.savetxt(filename, od_sel, delimiter=',', header="fromZone, toZone, min", fmt="%i,%i,%f")

        # filename = 'input/zonesSelected_tt_' + str(threshold) + '_internal.csv'
        # np.savetxt(filename, od_internal, delimiter=',', header="fromZone, toZone, min", fmt="%i,%i,%f")

        filename = 'input/zonesSelected_tt_reduced' + str(threshold) + '.csv'
        np.savetxt(filename, tt_array_reduced, delimiter=',', header="fromZone, toZone, min", fmt="%i,%i,%f")

    return tt_array_reduced


def distanceToStations(zones, stations):
    '''
    :param zones: recarray of zones with coordinates
    :param stations: recarray of stations with coordinates
    :return: distance matrix
    '''
    # calculate the distance from all zones to every staion
    xyzones = [zones.xcoord, zones.ycoord]
    xystations = [stations.xcoord, stations.ycoord]

    distance = utils.distance_euclidien(xyzones, xystations)
    return distance
