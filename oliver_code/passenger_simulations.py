import datetime
import random
import numpy as np
import numpy.lib.recfunctions as rfn
import networkx as nx
import shortestpath as sp
import time


def od_with_departure_time(time_window, zonesSelDemand, time_discretization, threshold):
    '''
    :param time_window: fromTime toTime
    :param zonesSelDemand: the selected od trips
    :param time_discretization: e.g. 5 min
    :param max_group_size: set
    :param threshold: zone selection threshold
    :param saveoutput: if true, file is saved as textfile
    :return: od with "fromZone, toZone, priority = 1, desired dep time"
    '''
    print('\n start to simulate the departure times of each passenger')
    start_time = datetime.datetime.strptime(time_window['FromTime'], "%Y-%m-%dT%H:%M:%S")
    end_time = datetime.datetime.strptime(time_window['ToTime'], "%Y-%m-%dT%H:%M:%S")
    time_interval = end_time - start_time
    from_zone = np.empty(0)
    to_zone = np.empty(0)
    nr_passengers = np.empty(0)
    desired_dep_time = np.empty(0)

    # od_departure_time = np.empty((0, 4))
    debug = False
    debug_n = 0
    print('  debug mode is ', str(debug))
    for od in zonesSelDemand:
        debug_n = debug_n + 1
        if debug:
            if debug_n > 1000:
                break
        t = non_homo_poisson_simulator(od, start_time, time_interval)
        if len(t) == 0:  # take care of od's with no simulated passengers
            continue
        n = 0  # nr of passenger in group
        timer_to_group = start_time + time_discretization
        for time in t:
            # identify the time slot of the passenger
            while time >= timer_to_group:
                timer_to_group = timer_to_group + time_discretization
            # save the passenger
            from_zone = np.append(from_zone, od.fromZone)
            to_zone = np.append(to_zone, od.toZone)
            nr_passengers = np.append(nr_passengers, 1)
            desired_dep_time = np.append(desired_dep_time, timer_to_group)

    from_zone = np.array(from_zone, dtype=[('fromZone', 'i8')])
    to_zone = np.array(to_zone, dtype=[('toZone', 'i8')])
    nr_passengers = np.array(nr_passengers,  dtype=[('priority', 'f8')])
    desired_dep_time = np.array(desired_dep_time, dtype=[('desired_dep_time', 'datetime64[m]')])
    od_departure_time = rfn.merge_arrays((from_zone, to_zone, nr_passengers, desired_dep_time), asrecarray=True)

    np.savetxt('OD_desired_departure_time_8000_10min.csv', od_departure_time, delimiter=',',
               header="fromZone, toZone, priority, desired dep time,", fmt="%i,%i,%f,%s")

    return od_departure_time


def get_correct_tph(od):
    # distribution of trips per hour in percentage for agglo - agglo and seperated by distance
    # short < 30 km
    # long > 30 km

    if od.distance30km == 0:
        tph_shortDistance = {6: 0.078814, 7: 0.137074, 8: 0.061484, 9: 0.026012}
        trips_per_hour = tph_shortDistance
    else:
        tph_longDistance = {6: 0.182223, 7: 0.117423, 8: 0.051569, 9: 0.024712}
        trips_per_hour = tph_longDistance
    return trips_per_hour


def non_homo_poisson_simulator(od, start_time, time_interval):

    trips_per_hour = get_correct_tph(od)
    lambda_t = {}  # passengers per hour
    for key, value in trips_per_hour.items():
        lambda_t[key] = value * od.passenger
    # non homogeneous poisson process
    t = []
    t_return = []
    u_list = []
    w_list = []
    s = datetime.timedelta(hours=0)
    s_list = [s]  # list for debug reasons
    d_list = []
    n = 0
    m = 0
    lambda_max = max(lambda_t.values())
    while s < time_interval:
        u = random.uniform(0, 1)
        u_list.append(u)
        w = - np.log(u) / lambda_max
        w_list.append(w)
        s = s + datetime.timedelta(hours=w)
        s_list.append(s)
        d = random.uniform(0, 1)
        d_list.append(d)
        # get the arrival rate of the simulated point
        lambda_s = start_time + s
        try:
            lambda_s = lambda_t[lambda_s.hour]
        except KeyError:
            # print('KeyError, s = ', s)
            # print(' with w = ', w)
            lambda_s = 0
        # accept or decline
        if d <= lambda_s / lambda_max:
            t.append(start_time + s)
            n = n + 1
        m = m + 1
    if n == 0:
        if od.passenger > 10:
            # print(' no passengers simulated for od, ', od)
            # print('u list ', u_list)
            # print('w list ', w_list)
            # print('s list  ', s_list)
            # print('d list ', d_list)
            # print('lambda max ', lambda_max)
            # print(' lambda t', lambda_t)
            t_return = t
        return t_return
    elif t[n-1] <= start_time + time_interval: # check if the last entry is within the the time interval
        t_return = t
    else:  # if the last value is outside time interval, return only n-1 values
        t_return = t[0:n - 1]
    # if len(t_return) == 0 and od.passenger > 10:
    #    print('why does it not stop ?')
    return t_return


def create_priority_list(od_departure_time):
    '''
    :param od_departure_time: record with all od, number of p and desired dep time
    :return: od_departure_time adds the random prioritity of each p
    '''
    total_number_passenger = od_departure_time.shape[0]
    # as comparison see: https://web.archive.org/web/20160221075946/http://www.zvv.ch/zvv/de/ueber-uns/zuercher-verkehrsverbund/kennzahlen-und-statistiken/fahrgastzahlen.html
    np.random.seed(42)
    random_priority_values = list(np.random.uniform(0, 1, total_number_passenger))

    od_departure_time.priority = random_priority_values
    # sort array by priorities
    od_departure_time.sort(order='priority')
    # reverse the order
    od_departure_time = od_departure_time[::-1]
    return od_departure_time


def group_passengers(od_departure_time, max_group_size, threshold):
    print('start of grouping passengers with max group size : %d ' %max_group_size)
    od_last_iteration = od_departure_time[0]
    group_size = 1
    od_list = list()

    for od in od_departure_time[1:]:
        if od == od_last_iteration:
            group_size = group_size + 1
        else:
            od_for_list = list(od_last_iteration)
            if group_size < max_group_size:
                od_for_list.append(group_size)
                od_list.append(od_for_list)
                group_size = 1
            else:
                while group_size >= max_group_size:
                    od_for_list.append(max_group_size)
                    od_list.append(od_for_list)
                    od_for_list = list(od_last_iteration)
                    group_size = group_size - max_group_size
                    if group_size == 0:
                        group_size = group_size + 1
                    elif group_size <= max_group_size:
                        od_for_list.append(group_size)
                        od_list.append(od_for_list)
                        group_size = 1
        od_last_iteration = od

    # take care of the last od !
    od_for_list = list(od_last_iteration)
    if group_size < max_group_size:
        od_for_list.append(group_size)
        od_list.append(od_for_list)
    else:
        while group_size >= max_group_size:
            od_for_list.append(max_group_size)
            od_list.append(od_for_list)
            od_for_list = list(od_last_iteration)
            group_size = group_size - max_group_size
            if group_size <= max_group_size and group_size != 0:
                od_for_list.append(group_size)
                od_list.append(od_for_list)

    passengers_before = od_departure_time.shape[0]
    passengers_after = np.asarray(od_list)
    passengers_after = passengers_after[:, 4].astype(np.int)
    passengers_after = np.sum(passengers_after)
    print('passengers before grouping : %d' % passengers_before)
    print('passengers after grouping : %d' % passengers_after)
    print('size of passenger od matrix after grouping : ', len(od_list))

    od_departure_time = np.array([tuple(x) for x in od_list], dtype=[('fromZone', 'i'), ('toZone', 'i'),('priority', 'f'),
                                                                     ('desired_dep_time', 'U18'), ('group_size', 'i')])
    filename = 'input/OD_desired_departure_time_' + str(threshold) + '_grouped.csv'
    np.savetxt(filename, od_departure_time, delimiter=',', header="fromZone, toZone, priority, desired dep time, group_size",
                   fmt="%i,%i,%f,%s,%i")

    return od_departure_time


def remove_arcs_not_in_time_range_of_od(G, desired_dep_time, max_waiting_time, min_waiting_time, origin):
    edges_to_remove = list()
    edges_to_add_again = list()
    # show neighbours of the origin
    neighbors_origin = [n for n in G.neighbors(origin)]
    # identify those with departure time close to desired dep time
    for station in neighbors_origin:
        dep_time_station = G.nodes[station]['departureTime']
        tt_to_station = G[origin][station]['weight']
        wait_at_station = dep_time_station - desired_dep_time - tt_to_station
        # condition to remove arc, waiting time is between min and max waiting time
        if wait_at_station < min_waiting_time or max_waiting_time < wait_at_station:  # train leaves before p is at the station --> remove arc
            edges_to_remove.append([origin, station])
            edges_to_add_again.append([origin, station, G[origin][station]['weight']])
    return edges_to_add_again, edges_to_remove


def passenger_assignment_algorithm(G, od_departure_time):
    # todo, not used anymore
    max_waiting_time = datetime.timedelta(minutes=60)  # identify trains to be used by a p depending on desired dep time
    min_waiting_time = datetime.timedelta(minutes=-15)
    capacity_train = 100000
    arcs_exceeding_capacity = list()
    tic_whole = time.time()
    for od in od_departure_time:
        tic_iteration_od = time.time()
        origin = od.fromZone
        destination = od.toZone
        desired_dep_time = datetime.datetime.strptime(od.desired_dep_time, "%Y-%m-%dT%H:%M")
        tic_edges_remove = time.time()
        edges_to_add_again, edges_to_remove = remove_arcs_not_in_time_range_of_od(G, desired_dep_time, max_waiting_time,
                                                                                  min_waiting_time, origin)
        G.remove_edges_from(edges_to_remove)  # remove edges, where p has to wait to long or train departs too early
        print('\n time to remove edges : ' + str(time.time()-tic_edges_remove))
        tic_path_search = time.time()
        try:
            path_od = sp.single_source_dijkstra(G, origin, destination)  # get the path
        except nx.exception.NetworkXNoPath:
            print('no path for od', od)
            G.add_weighted_edges_from(edges_to_add_again)  # add the removed edges again
            # print('time to find path : ' + str(time.time() - tic_path_search))
            continue
        # print('time to find path : ' + str(time.time()-tic_path_search))

        edges_used = np.asarray([path_od[1][1:-2], path_od[1][2:-1]]).transpose()  # get the edges used in the path

        tic_flow_assign = time.time()
        for n in range(0, edges_used.shape[0]):
            if G[edges_used[n, 0]][edges_used[n, 1]]['type'] == 'driving' or \
                    G[edges_used[n, 0]][edges_used[n, 1]]['type'] == 'waiting':
                # assign passengers on arcs
                G[edges_used[n, 0]][edges_used[n, 1]]['flow'] = G[edges_used[n, 0]][edges_used[n, 1]]['flow'] + 1
                if G[edges_used[n, 0]][edges_used[n, 1]]['flow'] >= capacity_train:
                    # capacity of arc exceeded --> remove arc
                    arcs_exceeding_capacity.append([edges_used[n, 0], edges_used[n, 1], G[edges_used[n, 0]][edges_used[n, 1]]['weight']])
        if len(arcs_exceeding_capacity) > 0:  # remove the arcs which exceeds the capacity
            G.remove_edges_from(arcs_exceeding_capacity)
        print('time assign flow : ' + str(time.time() - tic_flow_assign))

        tic_add_edges = time.time()
        G.add_weighted_edges_from(edges_to_add_again)  # add the removed edges again
        print('time add edges again : ' + str(time.time() - tic_add_edges))
        print('time for one od iteration : ' + str(time.time() - tic_iteration_od))

    toc_whole = time.time() - tic_whole
    print('\n passenger assignment finished ! it took  ', str(toc_whole), ' seconds to assign them on arcs ')



# BACKUPS
'''
def od_groups_with_departure_time_backup(time_window, zonesSelDemand, time_discretization, max_group_size,
                                         threshold, saveoutput=False):
    start_time = datetime.datetime.strptime(time_window['FromTime'], "%Y-%m-%dT%H:%M:%S")
    end_time = datetime.datetime.strptime(time_window['ToTime'], "%Y-%m-%dT%H:%M:%S")
    time_interval = end_time - start_time
    from_zone = np.empty(0)
    to_zone = np.empty(0)
    nr_passengers = np.empty(0)
    desired_dep_time = np.empty(0)

    # od_departure_time = np.empty((0, 4))
    debug = False
    debug_n = 0

    for od in zonesSelDemand:
        debug_n = debug_n + 1
        if debug:
            if debug_n > 1000:
                break
        t = non_homo_poisson_simulator(od, start_time, time_interval)
        # Todo debug, sometimes quite big ODs with more than 20 passengers per day have no passengers travelling
        if len(t) == 0:  # take care of od's with no simulated passengers
            continue
        n = 0  # nr of passenger in group
        timer_to_group = start_time + time_discretization
        for time in t:
            # identify the time slot of the group
            while time >= timer_to_group:
                if n != 0:
                    if n <= max_group_size:
                        from_zone = np.append(from_zone, od.fromZone)
                        to_zone = np.append(to_zone, od.toZone)
                        nr_passengers = np.append(nr_passengers, n)
                        n = 0
                        desired_dep_time = np.append(desired_dep_time, timer_to_group)
                    else:
                        # split the group
                        while n > 0:
                            if n - max_group_size > 0:
                                from_zone = np.append(from_zone, od.fromZone)
                                to_zone = np.append(to_zone, od.toZone)
                                nr_passengers = np.append(nr_passengers, max_group_size)
                                desired_dep_time = np.append(desired_dep_time, timer_to_group)
                                n = n - max_group_size
                            elif n - max_group_size < 0:
                                from_zone = np.append(from_zone, od.fromZone)
                                to_zone = np.append(to_zone, od.toZone)
                                nr_passengers = np.append(nr_passengers, n)
                                desired_dep_time = np.append(desired_dep_time, timer_to_group)
                                n = 0
                timer_to_group = timer_to_group + time_discretization
            # save the previous group
            n = n + 1
        # take care of the passengers of the last iteration
        if n != 0:
            if n <= max_group_size:
                from_zone = np.append(from_zone, od.fromZone)
                to_zone = np.append(to_zone, od.toZone)
                nr_passengers = np.append(nr_passengers, n)
                n = 0
                desired_dep_time = np.append(desired_dep_time, timer_to_group)
            else:
                # split the group
                while n > 0:

                    if n - max_group_size > 0:
                        from_zone = np.append(from_zone, od.fromZone)
                        to_zone = np.append(to_zone, od.toZone)
                        nr_passengers = np.append(nr_passengers, max_group_size)
                        desired_dep_time = np.append(desired_dep_time, timer_to_group)
                        n = n - max_group_size

                    elif n - max_group_size < 0:
                        from_zone = np.append(from_zone, od.fromZone)
                        to_zone = np.append(to_zone, od.toZone)
                        nr_passengers = np.append(nr_passengers, n)
                        desired_dep_time = np.append(desired_dep_time, timer_to_group)
                        n = 0

    from_zone = np.array(from_zone, dtype=[('fromZone', 'i8')])
    to_zone = np.array(to_zone, dtype=[('toZone', 'i8')])
    nr_passengers = np.array(nr_passengers,  dtype=[('passenger', 'i8')])
    desired_dep_time = np.array(desired_dep_time, dtype=[('desired_dep_time', 'datetime64[m]')])
    od_departure_time = rfn.merge_arrays((from_zone, to_zone, nr_passengers, desired_dep_time), asrecarray=True)

    if saveoutput:
        filename = 'input/OD_desired_departure_time_' + str(threshold) + '.csv'
        np.savetxt(filename, od_departure_time, delimiter=',', header="fromZone, toZone, passenger, desired dep time",
                   fmt="%i,%i,%i,%s")

    return od_departure_time


def create_priority_list_backup(od_departure_time, create_list = False):

    :param od_departure_time: record with all od, number of p and desired dep time
    :param create_list: if true, iteration through all ods and assigning a random variable to each p (takes long !!!),
                        otherwise it returns just a list of size(total passengers) with random values
    :return: list of uniform random values of size = total nr of passengers or detailled list with a priority for each p

    total_number_passenger = od_departure_time.shape[0]
    # as comparison see: https://web.archive.org/web/20160221075946/http://www.zvv.ch/zvv/de/ueber-uns/zuercher-verkehrsverbund/kennzahlen-und-statistiken/fahrgastzahlen.html
    random_priority_values = list(np.random.uniform(0, 1, total_number_passenger))

    if create_list:
        # create passenger priority lis
        # test = od_departure_time[0].desired_dep_time - datetime.timedelta(minutes=5)
        # priority_of_passenger2 = {}
        priority_passenger = []
        for od in od_departure_time:
            if od.passenger == 1:
                random_index = np.random.randint(0, len(random_priority_values))
                # priority_of_passenger[(od.fromZone, od.toZone, od.passenger, od.desired_dep_time)] = 1 #random_priority_values.pop(random_index)
                priority_passenger.append([od.fromZone, od.toZone, od.passenger, od.desired_dep_time,
                                           random_priority_values.pop(random_index)])
            else:
                for i in range(0, od.passenger):
                    random_index = np.random.randint(0, len(random_priority_values))
                    priority_passenger.append([od.fromZone, od.toZone, od.passenger, od.desired_dep_time,
                                               random_priority_values.pop(random_index)])
        return priority_passenger

    else:
        od_departure_time.priority = random_priority_values
        return od_departure_time


'''
