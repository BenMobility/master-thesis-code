import numpy as np
import datetime
import networkx as nx

def AddSBB_NodeID(path_dataset):
    '''
    input: path to dataset of nodes, exported from viriato database (BSP.mdb)
    return: same file but with additional SBB Node ID, which will be saved as Node_SBB.txt
    '''
    data = np.genfromtxt(path_dataset, delimiter=',', dtype=str)  # dtype = 'float'
    # print(data.dtype)

    NodeIDs = data[1:, 0]

    # c = np.array()
    c = ["NodeID_SBB"]

    # remove the two numbers in front of id name of Viriato
    for a in range(NodeIDs.shape[0]):
        #    print(NodeIDs.size)
        b = NodeIDs[a]
        # print(b,'b')
        b1 = b[0]  # "
        # print(b1,'b1')
        bbb = b[-1:]  # "
        # print(bbb,'bbb')
        bb = b[3:-1]
        # print(bb, 'bb')
        b = b1 + bb + bbb
        # print(b,'final b')
        # print(b)
        # print(b)
        c.append(b)
    # print(c[0:2])
    c = np.array(c, dtype=str)
    # add the sbb ideas into first column of the data file
    data2 = np.column_stack([c, data])  # add the
    # data2 = np.empty([data.shape[0], data.shape[1]+1], dtype=str)
    print(data2[0:2, :])
    # data2[:,0] = c[:]
    # print(data2)
    # data = [c, data]
    # print(c)
    np.savetxt('../Node_SBB.txt', data2, delimiter=',', fmt="%s")
    return data2


def Identify_SSZ(data):
    '''
        input:  dataset of nodes from viriato database with the added zone Nr and Pass frequ, obtained
                with joined tables from Qgis
        return: SSZ, MSZ
    '''
    # data = np.genfromtxt(path, skip_header=1, delimiter=',', dtype=str)  # dtype = 'float'
    #     0             1       2           3               4               5       6
    # NodeID_SBB	NodeID	Zone_Nr     X_Coordinate	Y_Coordinate	    SID	    NodeName
    #               7                       8                      9
    #  passagierfrequenz_bezugsjahr	passagierfrequenz_dtv	passagierfrequenz_dmw

    # print(data[0:2, :])
    print(data.shape, 'Data shape')

    # uniqueZone,  <-- unique zones, indeces, number of occurence
    #print(data[:, 6], 'all zones')
    uniqueZone, indices, count = np.unique(data[:, 2], return_index=True, return_counts=True)
    #print(' unique zones : ')
    #print(uniqueZone)
    #print(count, 'count')  # number of stations in zone

    # get all the zones with only one station
    singleStationZone = uniqueZone[count == 1]
    # print('all single station zones: ', len(singleStationZone))
    # print(singleStationZone)
    indOfSingleStationZone = indices[count == 1]
    data_SingleStationZone = data[indOfSingleStationZone, :]
    indOfMultipleStationZone= indices[count > 1]
    data_MultipleStationZone = data[indOfMultipleStationZone, :]

    # print(data_SingleStationZone, len(data_SingleStationZone), 'length')

    # print(zoneWithSingleStation, 'zone with single station')
    # print(indOfZoneWithSingleStation, 'index of zone with single station')

    # print(data[indices[:]], ' sbb node id')
    # uniqueZone = [uniqueZone, data[indices, 0]]
    return data_SingleStationZone[:,2], data_MultipleStationZone[:,2]


def Check_symmetric(a, tol):
    res = np.allclose(a, a.T, atol=tol)
    if res == True:
        print('Matrix is symmetric with tolerance : ', tol)
    else:
        print('Warning: Matrix not symmetric with tolerance : ', tol)
    return res


def distance_euclidien(p, q):
    """
    :param p: input array [x y] coordinates
    :param q: input array [x y] coordinates
    :return: distance array from p to q [p , q]
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    if p.shape[0] == 2 and p.shape[1] != 2:
        p = np.transpose(p)
    if q.shape[0] == 2 and q.shape[1] != 2:
        q = np.transpose(q)

    rows, cols = np.indices((p.shape[0], q.shape[0]))
    distances = []
    distances = ((p[rows.ravel()]-q[cols.ravel()])**2)
    distances = np.round(np.sqrt(distances.sum(axis=1)), 5)
    if p.shape[0] == 1:
        pass
    else:
        distances = np.reshape(distances, (p.shape[0], q.shape[0]))
    print(distances.shape, 'shape of distance matrix')
    return distances


def build_dict(seq, key):
    '''
    :param seq: list of dictionaries
    :param key: key for the dictionaries
    :return: converted dictionary keyed by key
    '''
    return dict((d[key], dict(d, index=index)) for (index, d) in enumerate(seq))


def transform_edges_into_dict_key_target(list_edges):
    edges_by_target = dict()
    for edge in list_edges:
        if edge[1] in edges_by_target:
            edges_by_target[edge[1]].append(edge)
        else:
            edges_by_target[edge[1]] = [edge]
    return edges_by_target


def transform_edges_into_dict_key_source(list_edges):
    edges_by_target = dict()
    for edge in list_edges:
        if edge[0] in edges_by_target:
            edges_by_target[edge[0]].append(edge)
        else:
            edges_by_target[edge[0]] = [edge]
    return edges_by_target


def transform_odt_into_dict_key_source(odt_for_sp):
    '''
    :param odt_for_sp: odt matrix with [0] origin, [1] destination, [2] priority, [3] group siye
    :return: source-target dictionary with key origin, and value [(destination, group size)] for all destinations from specific origin
    '''
    source_target_dict = dict()
    for od in odt_for_sp:
        if od[0] in source_target_dict:
            source_target_dict[od[0]].append((od[1], od[3]))
        else:
            source_target_dict[od[0]] = [(od[1], od[3])]
    return source_target_dict


def reshapeODMatrix(saveoutput, saveoutputname=None, data=None , path=None, symmetriecheck=False):
    '''
            input:  path to OD dataset from NPVM
            return: OD data in Matrix form, Origins in rows, Destinations in Column. First Value in each col, row
            corresponds to the zone ID
    '''

    print('start reading file')

    if data is None and path is None:
        raise Exception("Data and path is None, please specify one !")


    if data is None and path is not None:
        data = np.genfromtxt(path, skip_header=0)
    elif data is not None and path is None:
        data = data


    # print(data,'data')

    uniqueOrig, indOrig, countOrig = np.unique(data[:, 0], return_index=True, return_counts=True)
    uniqueDest, indDest, countDest = np.unique(data[:, 1], return_index=True, return_counts=True)
    # print(uniqueDest==uniqueOrig)
    # print(len(uniqueDest))
    # print(len(uniqueOrig))
    # print(uniqueOrig == uniqueDest)
    # print(uniqueDest[countDest<2949])
    # print(data)
    od = data[:, 2]
    # od [origin zone, destination zone]
    od = od.reshape(len(uniqueOrig), len(uniqueDest))

    for i in range(0, len(uniqueOrig)):  # check if the od's are still the same as in original file
        if any(od[i, :] == data[i * len(uniqueDest):len(uniqueDest) * i + len(uniqueDest), 2]) == True:
            pass  # print('True')
        else:
            raise Exception('Error: some OD of input do not match new shaped OD')
    print('check if the ODs are still the same as in original file successfully passed')
    origins = data[indOrig, 0]  # to get the origins in the same order as OD
    destinations = data[indDest, 1]  # to get the destination in the same order as OD
    destinations = np.append(0, destinations)

    #add origin at first place in each row
    od = np.insert(od, [0], origins.reshape(len(uniqueOrig), 1), axis=1)
    #add destination at top of each colon
    od = np.insert(od, 0, destinations, axis=0)

    if symmetriecheck == True:
        Check_symmetric(od, 0.1)
    else:
        pass

    if saveoutput == True and saveoutputname is not None:
        print('saved file as', saveoutputname)
        np.savetxt(saveoutputname, od, fmt='%.6e', delimiter=', ')
    else:
        print('od not saved !')

    return od


def selectODofSubsetZones(path_zones, path_data, saveoutput, saveoutputname, treshhold=None):
    '''
            input:  path to file with selected Zones and OD dataset from NPVM, saveoutput true if you want to save it
                    saveoutputname e.g. 'myfile.txt', treshhold* (opt) to delete values smaller than th (tt in minutes)
            return: OD from and to selected zones, either origin or destination is within selected subset,
                    the ods with less than 1 passenger travelling is removed !
    '''
    print('start of zone subset selection ')
    data = np.genfromtxt(path_data, skip_header=0)
    zones_input = np.genfromtxt(path_zones, skip_header=1)

    # data = [from, to, passenger]
    # # select the data with all o/d of all zones selected
    zones_select = np.isin(data, zones_input)
    data_zonesSel = np.append(data[zones_select[:, 0], :], data[zones_select[:, 1], :], axis=0)

    if treshhold is None:
        pass
    else:
        data_zonesSel = np.delete(data_zonesSel, np.where(data_zonesSel[:, 2] < treshhold), axis=0)

    if saveoutput == True:
        print('saved file as ', saveoutputname)
        np.savetxt(saveoutputname, data_zonesSel, fmt='%i %i %1.4f', delimiter=', ')
    else:
        print('od not saved !')

    return data_zonesSel


def utility_stations(stat_CompSet):
    # 0                          1                  2       3   4   5   6   7      8            9
    # zone nr of station    train frq dmw      SBB ID       x   y   PR  IC  tt  zone of ZSZ     dist ,

    d = np.asarray(stat_CompSet[:, 9], dtype=float)
    f = np.asarray(stat_CompSet[:, 1], dtype=float)/30
    f[f <= 10] = 20
    pr = stat_CompSet[:,5]
    pr[pr == ''] = 0
    pr = np.asarray(stat_CompSet[:,5], dtype=int)
    pr[pr > 1] = 1
    pr[pr == 0] = 0
    ic = np.asarray(stat_CompSet[:,6], dtype=int)


    # find the correct index for the distance beta
    # dist = np.arange(1000, 10500, 500)
    # dist = np.append([250, 500], dist)

    d_cat = np.asarray(d[:]/500, dtype=int)
    d_cat[d_cat > 20] = 20
    b_dist = [7.379, 6.852, 6.328, 5.734, 5.147, 4.797, 4.150, 3.723, 3.411, 2.940, 2.765, 2.471, 2.282, 1.953, 1.902,
              1.748, 1.626, 1.487, 1.143, 1.237, 0.955]

    b_freq = 0.004
    b_pr = 0.419
    # if intercity station --> same as adding 300 trains per day
    u_ic = b_freq * 300



    u_j = np.empty([stat_CompSet.shape[0],1])
    #u_j[:, 0] = stat_CompSet[:, 2]
    for i in range(0,stat_CompSet.shape[0]):
        u_j[i] = b_dist[d_cat[i]] + b_freq * f[i] + b_pr*pr[i] + u_ic*ic[i]
        #           *np.log(d[i])+ b_distsqu * np.square(np.log(d[i])) + b_frq*np.log(f[i]) + b_frqsqu*np.square(np.log(f[i]))+ b_distfrq*(np.log(d[i])*np.log(f[i]))

    u_j = np.round(u_j,5)
    stat_CompSet = np.append(stat_CompSet,u_j, axis = 1)
    u_j = np.round(np.exp(u_j),5)
    U = np.round(np.sum(u_j, axis = 0),5)

    p_j = np.round(np.divide(u_j, U), 5)
    stat_CompSet = np.append(stat_CompSet,p_j, axis = 1)

    return stat_CompSet


def round_time_to_minute(time_to_round):
    '''
    :param time_to_round: input as a datetime object
    :return: rounded to hourly value, neglecting date
    '''
    if time_to_round.second > 29 and time_to_round.minute < 59:
        minute_rounded = time_to_round.minute + 1
        hour_rounded = time_to_round.hour
    elif time_to_round.second > 29 and time_to_round.minute == 59:
        minute_rounded = 0
        hour_rounded = time_to_round.hour + 1
    else:
        minute_rounded = time_to_round.minute
        hour_rounded = time_to_round.hour
    return hour_rounded, minute_rounded


def round_seconds_up_to_minutes(time_to_round):
    '''
    :param time_to_round: input are the datetime timedelta in seconds
    :return: round up to minutes, so each started second is uprounded to a minute
    '''
    seconds = time_to_round % 60
    minutes = int(time_to_round / 60)

    if seconds > 1:
        minutes += 1

    return datetime.timedelta(minutes=minutes)


def check_acyclicity_of_Graph(G):
    '''
    :param G: Graph to test the acyclicity
    :return: True or False
    '''
    is_graph_acyclic = nx.is_directed_acyclic_graph(G)
    if not is_graph_acyclic:
        cycles = nx.find_cycle(G)
        print('Graph is not acyclic, check trains with double visited nodes', cycles)
        G.nodes[cycles[0][1]]
        for edge in cycles:
            edge = G.edges.data([edge[0], edge[1]])
            # print('edge of cycle', edge)
            G[cycles[0][1]][cycles[0][1]]
    return is_graph_acyclic


def getNumberOfPassenger(zoneID, data):
    """
    :param zoneID: Input zone ID, OD matrix
    :return:
    """
    z = zoneID
    toZone = np.where(data[:, 1] == z)
    sumToGuarda = np.sum(data[toZone,2])
    fromGuarda = np.where(data[:, 0] == z)
    sumfromGuarda = np.sum(data[fromGuarda,2])
    # fromtoGuarda = data[data[:,0] == z,:]
    print(sumToGuarda, ' to zone',  z, '\n \n ', sumfromGuarda, ' from zone ')


def get_list_of_all_minutes_track_is_used(dep_time_this_node, travel_time_to_next_node):
    dep_time_rounded = round_time_to_minute(dep_time_this_node)
    dep_time_this_node = datetime.datetime(dep_time_this_node.year, dep_time_this_node.month, dep_time_this_node.day,
                                           hour=dep_time_rounded[0], minute=dep_time_rounded[1], second=0)

    run_time_seconds = round_seconds_up_to_minutes(travel_time_to_next_node.seconds)
    run_time_minutes = int(run_time_seconds.seconds / 60)
    minutes_track_is_used = [dep_time_this_node + datetime.timedelta(minutes=x) for x in range(1, run_time_minutes + 1)]
    return minutes_track_is_used


def duration_transform_to_timedelta(duration):
    if duration == 'P0D':
        return datetime.timedelta(seconds=0)
    not_used = ['P', 'Y', 'D']
    hour, minute, second = 0, 0, 0
    time_start = False
    skip_char = False

    idx = -1
    for char in duration:
        idx += 1
        if char in not_used:
            continue
        elif skip_char:
            skip_char = False
            continue
        elif char == 'M' and not time_start:
            continue
        elif char == 'T':
            time_start = True
        elif char == 'H':
            hour = value
        elif char == 'M' and time_start:
            minute = value
        elif char == 'S':
            second = value
        elif char.isdigit():
            if duration[idx+1].isdigit():
                value = int(char)*10 + int(duration[idx+1])
                skip_char = True
            else:
                value = int(char)

    return datetime.timedelta(hours=hour, minutes=minute, seconds=second)


def transform_timedelta_to_ISO8601(timedelta):
    if timedelta.seconds == 0:
        return 'P0D'
    else:
        return 'PT' + str(int(timedelta.seconds)) + 'S'

