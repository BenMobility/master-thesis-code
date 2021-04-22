import numpy as np
import numpy.lib.recfunctions as rfn
import algorithm_platform_methods as ap
import time


def ReadSBBNodes(path_nodesSBB):
    '''
    :param path_nodesSBB: path to the node with the sbb nodes
    :return: stations as structured array with names : 'zone, SbbID, ViriatoID, NodeName, xcoord, ycoord'
    '''
    # load SBB railway stations
    print('\n start reading SBB nodes from \n', path_nodesSBB )
    stations = np.genfromtxt(path_nodesSBB, delimiter=',',  dtype=str, skip_header=1)
    #     0             1       2           3               4            5         6
    # NodeID_SBB	NodeID	Zone_Nr     NodeName	X_Coordinate	Y_Coordinate  SID
    #               7                       8                      9                10      11
    # passagierfrequenz_bezugsjahr	passagierfrequenz_dtv	passagierfrequenz_dmw   PR      Intercity

    # station_zoneNR
    # 0                          1       2                 3            4       5
    # zone nr of station       SBB ID   SBB ID Viriato     Node Name    x       y
    station_zoneNR = stations[:, 2]
    station_zoneNR = station_zoneNR.astype(int).reshape(station_zoneNR.shape[0], 1)
    # add SBB ID and SBB ID Viriato and Name
    station_zoneNR = np.concatenate((station_zoneNR, np.asarray(stations[:, 0:3], dtype=str).reshape(station_zoneNR.shape[0],3)), axis = 1)
    # add x and y coordinate
    station_zoneNR = np.concatenate((station_zoneNR, np.asarray(stations[:, 4:6], dtype=float).reshape(station_zoneNR.shape[0],2)), axis = 1)
    # pass frequency
    # station_zoneNR = np.concatenate((station_zoneNR, np.asarray(stations[:, 8], dtype = float).reshape(station_zoneNR.shape[0],1)), axis = 1)
    # station_zoneNR = station_zoneNR.tolist()
    # dtype = [('zone', 'int'), ('SbbID', '<U32'), ('ViriatoID', '<U32'), ('NodeName', '<U32'), ('xcoord', 'float'), ('ycoord', 'float')]
    # station_zoneNR = np.array(station_zoneNR, dtype=dtype)

    #convert nd array into structured array
    station_zoneNR = np.core.records.fromarrays(station_zoneNR.transpose(),
                                             names='zone, SbbID, ViriatoID, NodeName, xcoord, ycoord',
                                             formats='i8, U13, U13, U13, f8, f8')

    print('Successfully read stations:  with names: ', station_zoneNR.dtype.names)


    return station_zoneNR


def nodesSBB_add_code_of_viriato(code_id_dictionary, nodesSBB):

    print(nodesSBB.shape[0], ' total number of nodes in database')
    # names : 'zone, SbbID, ViriatoID, NodeName, xcoord, ycoord, Code'
    nodesSBB = rfn.append_fields(nodesSBB, 'Code', np.ones(nodesSBB.shape[0]) * -1, dtypes='i8', usemask=False,
                                 asrecarray=True)
    nodesSBB = rfn.append_fields(nodesSBB, 'Visited', np.zeros(nodesSBB.shape[0]), dtypes=bool, usemask=False,
                                 asrecarray=True)
    # add the code from viriato to the list of nodes
    for key, value in code_id_dictionary.items():
        idx = np.where(nodesSBB.ViriatoID == value)
        nodesSBB.Code[idx] = key
        nodesSBB.Visited[idx] = True

    # find the code for the unvisited nodes, rather difficult
    # stations_inArea = nodesSBB.copy()
    # nodesSBB = nodesSBB[nodesSBB.Code != -1]
    # for unvisited_node in nodesSBB[nodesSBB.Visited != False]:
    #    node = ap.nodes(unvisited_node.ViriatoID)

    print(nodesSBB.shape[0], ' number of nodes visited in database')

    return nodesSBB


def process_stations(path_nodesSBB):
    '''
    :param path_nodesSBB: path to the nodes SBB file
    :return: all stations visited by trains in timewindow (nodesSBB) as recarray with names:
            'zone, SbbID, ViriatoID, NodeName, xcoord, ycoord', 'code'
            saved in input as Stations_used.csv
    '''
    node_ids = ap.get_all_visited_nodes_in_time_window()
    code_id_dictionary, id_code_dictionary = ap.get_all_viriato_node_ids_to_code_dictionary(node_ids)
    # save_dictionary(code_id_dictionary, 'code_id_dictionary_Scen2005_10052003')
    # code_id_dictionary = pickle.load(open("code_id_dictionary.pickle", "rb"))
    nodesSBB = ReadSBBNodes(path_nodesSBB)
    nodesSBB = nodesSBB_add_code_of_viriato(code_id_dictionary, nodesSBB)
    # remove Horgen & Rüti kilometer from AoI
    nodesSBB = nodesSBB[nodesSBB.ViriatoID != '85HG']
    nodesSBB = nodesSBB[nodesSBB.ViriatoID != '85RUEKM']
    # names : 'zone, SbbID, ViriatoID, NodeName, xcoord, ycoord', 'Code'
    np.savetxt('input/Stations_used.csv', nodesSBB, delimiter=',', header="zone, SbbID, ViriatoID, NodeName, xcoord, ycoord, Code",
               fmt="%i,%s,%s,%s,%f,%f,%i,%s")

    return nodesSBB, code_id_dictionary, id_code_dictionary


def dataReaderZones_original(path_Zones):
    '''
    :param path_Zones: path to the file with centroids of zones
    :return: structured array with zone id and x y coords
    '''
    zoneCentroids = np.genfromtxt(path_Zones, delimiter=',', dtype=str, skip_header=1)

    zoneCentroids = np.core.records.fromarrays(zoneCentroids.transpose(),
                                             names='zone, xcoord, ycoord',
                                             formats='i8, f8, f8')

    # print(' \n Successfully read centroids from : \n', path_Zones, '\n', ' with names: ', zoneCentroids.dtype.names)
    # print(zoneCentroids[0].zone)
    return zoneCentroids


def dataReaderZones_load(path_Zones):
    '''
    :param path_Zones: path to  subset already exitsting
    :return: structured array with zone id and x y coords
    '''
    zoneCentroids = np.genfromtxt(path_Zones, delimiter=',', dtype=str, skip_header=1)

    zoneCentroids = np.core.records.fromarrays(zoneCentroids.transpose(),
                                             names='zone, xcoord, ycoord, distance',
                                             formats='i8, f8, f8, f8')

    # print(' Successfully read centroids from : \n', path_Zones, '\n' , ' with names: ', zoneCentroids.dtype.names)
    # print(zoneCentroids[0].zone)
    return zoneCentroids


def dataReaderOD_original(path_OD):
    '''
    :param path_OD: path to the file with OD demand of zones
    :return: structured array with, from zone,to zone, pass
    '''
    # print('Start reading OD demand file')
    od = np.genfromtxt(path_OD)
    od = np.core.records.fromarrays(od.transpose(),
                                             names='fromZone, toZone, passenger',
                                             formats='i8, i8, f8')

    # print(' Successfully read OD from : \n', path_OD, '\n', 'with names: ', od.dtype.names)
    return od


def dataReaderOD_TT_original(path_OD):
    '''
    :param path_OD: path to the file with OD demand of zones
    :return: structured array with, from zone,to zone, pass
    '''

    # print('Start reading OD Travel time file')
    tic = time.time()
    od = np.genfromtxt(path_OD)
    toc = time.time() - tic
    # print('time used to read: ', toc)

    od = np.core.records.fromarrays(od.transpose(),
                                             names='fromZone, toZone, traveltime',
                                             formats='i8, i8, f8')

    # print('Successfully read OD from : \n', path_OD, '\n', ' with names: ', od.dtype.names)
    # print(od.fromZone)
    return od


def dataReaderOD_load(path_OD):
    '''
    :param path_OD: path to the file with OD demand of zones
    :return: structured array with, from zone,to zone, pass
    '''
    # print('Start reading OD file')
    od = np.genfromtxt(path_OD, delimiter=',', skip_header=1)

    od = np.core.records.fromarrays(od.transpose(),
                                             names='fromZone, toZone, passenger, distance30km',
                                             formats='i8, i8, f8, i8')

    # print('Successfully read OD from : \n', path_OD, '\n' , ' with names: ', od.dtype.names)
    # print(od.fromZone)
    return od


def  dataReaderTT_load(path_TT):
    '''
    :param path_OD: path to the file with OD demand of zones
    :return: structured array with, from zone,to zone, pass
    '''
    # print('Start reading OD file')
    od = np.genfromtxt(path_TT, delimiter=',', skip_header=1)
    # print(od[np.where(od[:, 0] == 1)])
    # Remove dupliactes
    od = np.unique(od, axis=0)

    od = np.core.records.fromarrays(od.transpose(),
                                             names='fromZone, toZone, tt',
                                             formats='i8, i8, f8')

   #  print('Successfully read TT from : \n', path_TT, '\n' , ' with names: ', od.dtype.names)
    # print(od.fromZone)
    return od


def dataReader_load_od_departure_time(path_od_depTime):
    '''
    :param path_Zones: path to  subset already exitsting
    :return: structured array with zone id and x y coords
    '''
    od_departure_time = np.genfromtxt(path_od_depTime, delimiter=',', dtype=str, skip_header=1)


    od_departure_time = np.core.records.fromarrays(od_departure_time.transpose(),
                                             names='fromZone, toZone, priority, desired_dep_time',
                                             formats='i8, i8, f8,  U25')  # datetime64[m]
    # arrival_time_this_node = datetime.datetime.strptime(train_path_nodes['ArrivalTime'], "%Y-%m-%dT%H:%M:%S")

    # print(' Successfully read OD with desired departure time from : \n', path_od_depTime, '\n' , ' with names: ',
    #      od_departure_time.dtype.names)
    # print(zoneCentroids[0].zone)
    return od_departure_time

def dataReader_load_od_departure_time_grouped(path_od_depTime):
    '''
    :param path_Zones: path to  subset already exitsting
    :return: structured array with zone id and x y coords
    '''
    od_departure_time = np.genfromtxt(path_od_depTime, delimiter=',', dtype=str, skip_header=1)


    od_departure_time = np.core.records.fromarrays(od_departure_time.transpose(),
                                             names='fromZone, toZone, priority, desired_dep_time, group_size',
                                             formats='i8, i8, f8,  U25, i8')  # datetime64[m]
    # arrival_time_this_node = datetime.datetime.strptime(train_path_nodes['ArrivalTime'], "%Y-%m-%dT%H:%M:%S")

    # print(' Successfully read OD with desired departure time from : \n', path_od_depTime, '\n' , ' with names: ',
    #      od_departure_time.dtype.names)
    #  print(zoneCentroids[0].zone)
    return od_departure_time

def dataReader_load_od_departure_time_backup(path_od_depTime):
    '''
    :param path_Zones: path to  subset already exitsting
    :return: structured array with zone id and x y coords
    '''
    od_departure_time = np.genfromtxt(path_od_depTime, delimiter=',', dtype=str, skip_header=1)


    od_departure_time = np.core.records.fromarrays(od_departure_time.transpose(),
                                             names='fromZone, toZone, passenger, desired_dep_time',
                                             formats='i8, i8, i8, U25')  # datetime64[m]

    # print(' Successfully read OD with desired departure time from : \n', path_od_depTime, '\n' , ' with names: ',
    #      od_departure_time.dtype.names)
    # print(zoneCentroids[0].zone)
    return od_departure_time