import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def main():
    # threeD_plot()
    scenario = 'Run_20_11'
    path = 'z_pickles\\' + scenario + '\\'
    input_file = 'z_pickle.pkl'
    input_file_archive = 'solution_archive.pkl'

    solution_archive = load_file_from_pickle(path+input_file_archive)

    '''
     z_for_pickle = {'z_op_cur': z_op_current, 'z_de_cur': z_de_current, 'z_tt_cur': z_tt_current,
                    'z_op_acc': z_op_accepted, 'z_de_acc': z_de_accepted, 'z_tt_acc': z_tt_accepted,
                    'z_cur_acceppted': z_cur_accepted, 'z_cur_archived': z_cur_archived}
    '''

    solution = load_file_from_pickle(path+input_file)
    plt.rcParams['font.size'] = 10.0

    fig = all_four_plots_op_as_scale_all_archives(path, solution)
    fig.tight_layout()  #
    filename = 'all_four_plots_all_archive_op_as_scale'
    plt.savefig(path + filename, papertype='a5', orientation='landscape')
    plt.show()

    fig = all_four_plots_op_as_scale_non_dominated_archives(path, solution, solution_archive)

    fig.tight_layout()  #
    filename = 'all_four_plots_non_dom_archive_op_as_scale'
    plt.savefig(path + filename, papertype='a5', orientation='landscape')
    plt.show()


    fig = all_four_plots_dev_as_scale_non_dominated_archives(path, solution, solution_archive)

    fig.tight_layout()  #
    filename = 'all_four_plots_non_dom_archive_dev_as_scale'
    plt.savefig(path + filename, papertype='a5', orientation='landscape')
    plt.show()


    # <plot_temperatures(path, solution)


def all_four_plots_op_as_scale_all_archives(path, solution):
    fig, axs = plt.subplots(2, 2, figsize=((16 / 1.5), (9 / 1.5)))
    intensity = 0.7
    scatter_size = 15
    greyish_blue = '#5e819d'
    royal_blue = '#0504aa'
    dark_magenta = '#960056'
    scatter_color = greyish_blue
    # plot of tt versus iterations
    solution_y_axis = solution['z_tt_acc']
    # filename = 'results_tt_iterations.png'
    ylabel = 'z_p [min]'
    ax = axs[0, 0]
    plot_z_versus_iterations_as_scatter(ax, intensity, scatter_color, scatter_size, solution_y_axis, ylabel)
    # plot of op versus iterations
    solution_y_axis = solution['z_op_acc']
    # filename = 'results_op_iterations.png'
    ylabel = 'z_o [km]'
    ax = axs[0, 1]
    plot_z_versus_iterations_as_scatter(ax, intensity, scatter_color, scatter_size, solution_y_axis, ylabel)
    # plot dev versus iterations
    solution_y_axis = solution['z_de_acc']
    # filename = 'results_de_iterations.png'
    ylabel = 'z_d [min]'
    ax = axs[1, 0]
    plot_z_versus_iterations_as_scatter(ax, intensity, scatter_color, scatter_size, solution_y_axis, ylabel)
    solution_archived = prepare_solution_archive_for_plot(solution)
    ax = axs[1, 1]
    plot_2d_scatter_deviation_tt_op_as_scale(ax, path, solution_archived)
    return fig


def all_four_plots_op_as_scale_non_dominated_archives(path, solution, solution_archive):
    fig, axs = plt.subplots(2, 2, figsize=((16 / 1.5), (9 / 1.5)))
    intensity = 0.7
    scatter_size = 15
    greyish_blue = '#5e819d'
    royal_blue = '#0504aa'
    dark_magenta = '#960056'
    scatter_color = greyish_blue
    # plot of tt versus iterations
    solution_y_axis = solution['z_tt_acc']
    # filename = 'results_tt_iterations.png'
    ylabel = 'z_p [min]'
    ax = axs[0, 0]
    plot_z_versus_iterations_as_scatter(ax, intensity, scatter_color, scatter_size, solution_y_axis, ylabel)
    # plot of op versus iterations
    solution_y_axis = solution['z_op_acc']
    # filename = 'results_op_iterations.png'
    ylabel = 'z_o [km]'
    ax = axs[0, 1]
    plot_z_versus_iterations_as_scatter(ax, intensity, scatter_color, scatter_size, solution_y_axis, ylabel)
    # plot dev versus iterations
    solution_y_axis = solution['z_de_acc']
    # filename = 'results_de_iterations.png'
    ylabel = 'z_d [min]'
    ax = axs[1, 0]
    plot_z_versus_iterations_as_scatter(ax, intensity, scatter_color, scatter_size, solution_y_axis, ylabel)
    solution_non_dom = prepare_solution_archive_non_dominated_for_plot(solution_archive)
    ax = axs[1, 1]
    plot_2d_scatter_deviation_tt_op_as_scale(ax, path, solution_non_dom)
    return fig


def all_four_plots_dev_as_scale_non_dominated_archives(path, solution, solution_archive):
    fig, axs = plt.subplots(2, 2, figsize=((16 / 1.5), (9 / 1.5)))
    intensity = 0.7
    scatter_size = 15
    greyish_blue = '#5e819d'
    royal_blue = '#0504aa'
    dark_magenta = '#960056'
    scatter_color = greyish_blue
    # plot of tt versus iterations
    solution_y_axis = solution['z_tt_acc']
    # filename = 'results_tt_iterations.png'
    ylabel = 'z_p [min]'
    ax = axs[0, 0]
    plot_z_versus_iterations_as_scatter(ax, intensity, scatter_color, scatter_size, solution_y_axis, ylabel)
    # plot of op versus iterations
    solution_y_axis = solution['z_op_acc']
    # filename = 'results_op_iterations.png'
    ylabel = 'z_o [km]'
    ax = axs[0, 1]
    plot_z_versus_iterations_as_scatter(ax, intensity, scatter_color, scatter_size, solution_y_axis, ylabel)
    # plot dev versus iterations
    solution_y_axis = solution['z_de_acc']
    # filename = 'results_de_iterations.png'
    ylabel = 'z_d [min]'
    ax = axs[1, 0]
    plot_z_versus_iterations_as_scatter(ax, intensity, scatter_color, scatter_size, solution_y_axis, ylabel)
    solution_non_dom = prepare_solution_archive_non_dominated_for_plot(solution_archive)
    ax = axs[1, 1]
    plot_2d_scatter_op_tt_dev_as_scale(ax, path, solution_non_dom)
    return fig


def plot_temperatures(path, solution):


    temperature = solution['t_it']
    iterations = np.arange(0, len(temperature))
    fig, ax = plt.subplots()
    names = []

    ax.set(xlabel='Iteration', ylabel='Temperature',
           title='Title', ylim=(0, max(temperature) * 1.05))
    ax.grid()
    ax.plot(iterations, temperature)
    fig.savefig(path + 'temperature')
    plt.show()


def prepare_solution_archive_for_plot( solution):
    idx = -1
    z_op_arch, z_de_arch, z_tt_arch = [], [], []
    for archived in solution['z_cur_archived']:
        idx += 1
        if archived:
            z_op_arch.append(solution['z_op_cur'][idx])
            z_de_arch.append(solution['z_de_cur'][idx])
            z_tt_arch.append(solution['z_tt_cur'][idx])

    solution_archived = {'z_op_arch': z_op_arch, 'z_de_arch': z_de_arch, 'z_tt_arch': z_tt_arch}
    return solution_archived


def prepare_solution_archive_non_dominated_for_plot(solution_archive):
    z_op_arch, z_de_arch, z_tt_arch = [], [], []
    for solution in solution_archive:
        z_op_arch.append(solution.total_dist_train)
        z_de_arch.append(solution.deviation_timetable)
        z_tt_arch.append(solution.total_traveltime)

    solution_archived = {'z_op_arch': z_op_arch, 'z_de_arch': z_de_arch, 'z_tt_arch': z_tt_arch}
    return solution_archived




def plot_z_versus_iterations_as_scatter(ax, intensity, scatter_color, scatter_size, solution_y_axis,
                                        label_y_axis, filename=None, path=None):

    iterations = np.arange(0, len(solution_y_axis))
    # fig = plt.figure()
    # intensity = np.arange(0.3, 1.001, (1.001 - 0.5) / (len(z_op_acc) + 1))
    # ax = fig.gca()
    ax.set_ylim(min(solution_y_axis) - max(solution_y_axis) / 20, max(solution_y_axis) + max(solution_y_axis) / 20)
    ax.set_ylabel(label_y_axis)
    ax.set_xlabel('iterations')
    ax.scatter(iterations, solution_y_axis, s=scatter_size, edgecolors=scatter_color, facecolors='none',
               alpha=intensity, marker='o' )
    # c=scatter_color
    # plt.savefig(path + "figures\\" + filename)
    # plt.show()


def plot_2d_scatter_deviation_tt_op_as_scale(ax, path, solution):
    # figure 2D with axes x total tt, y z_d, z operational cost size of bulles
    z_op_acc, z_de_acc, z_tt_acc = solution['z_op_arch'], solution['z_de_arch'], solution['z_tt_arch']
    # fig = plt.figure()
    greyish_blue = '#5e819d'
    royal_blue = '#0504aa'
    dark_magenta = '#960056'
    # intensity = np.arange(0.3, 1.001, (1.001 - 0.5) / (len(z_op_acc) + 1))
    intensity = 0.7
    min_size = 10
    max_size = 60
    size_op = []
    for i in range(0, len(z_op_acc)):
        size = min_size + (max_size - min_size) * (z_op_acc[i] - min(z_op_acc)) / (max(z_op_acc) - min(z_op_acc))
        size_op.append(size)
    # ax = fig.gca()
    ax.set_xlim(min(z_tt_acc) - 1000, max(z_tt_acc) + 1000)
    ax.set_xlabel('z_p [min]')
    ax.set_ylim(min(z_de_acc) - 1000, max(z_de_acc) + 1000)
    ax.set_ylabel('z_d [min]')
    scatter = ax.scatter(z_tt_acc, z_de_acc, s=size_op, alpha=intensity, c=dark_magenta, marker='o')

    # produce a legend with a cross section of sizes from the scatter
    handles, labels = scatter.legend_elements(prop="sizes",c=dark_magenta, alpha=0.6, fmt='o')
    handles = [handles[0], handles[-1]]
    labels = [min(z_op_acc), max(z_op_acc)]
    legend2 = ax.legend(handles, labels, loc="lower right", title="z_o [km]")

    # filename = 'results_tt_dev_op_sized.png'
    # plt.savefig(path + "figures\\" + filename)
    # plt.show()



def plot_2d_scatter_op_tt_dev_as_scale(ax, path, solution):
    # figure 2D with axes x total tt, y z_d, z operational cost size of bulles
    z_op_acc, z_de_acc, z_tt_acc = solution['z_op_arch'], solution['z_de_arch'], solution['z_tt_arch']
    # fig = plt.figure()
    greyish_blue = '#5e819d'
    royal_blue = '#0504aa'
    dark_magenta = '#960056'
    # intensity = np.arange(0.3, 1.001, (1.001 - 0.5) / (len(z_op_acc) + 1))
    intensity = 0.7
    min_size = 10
    max_size = 60
    size_dev = []
    for i in range(0, len(z_de_acc)):
        size = min_size + (max_size - min_size) * (z_de_acc[i] - min(z_de_acc)) / (max(z_de_acc) - min(z_de_acc))
        size_dev.append(size)
    # ax = fig.gca()
    ax.set_xlim(min(z_tt_acc) - 1000, max(z_tt_acc) + 1000)
    ax.set_xlabel('z_p [min]')
    ax.set_ylim(min(z_op_acc) - 100, max(z_op_acc) + 100)
    ax.set_ylabel('z_o [km]')
    scatter = ax.scatter(z_tt_acc, z_op_acc, s=size_dev, alpha=intensity, c=dark_magenta, marker='o')

    # produce a legend with a cross section of sizes from the scatter
    handles, labels = scatter.legend_elements(prop="sizes", c=dark_magenta, alpha=0.6, fmt='o')
    handles = [handles[0], handles[-1]]
    labels = [min(z_de_acc), max(z_de_acc)]
    legend2 = ax.legend(handles, labels, loc="upper right", title="z_d [min]")

    # filename = 'results_tt_dev_op_sized.png'
    # plt.savefig(path + "figures\\" + filename)
    # plt.show()





def threeD_plot():
    '''
     z_for_pickle = {'z_op_cur': z_op_current, 'z_de_cur': z_de_current, 'z_tt_cur': z_tt_current,
                    'z_op_acc': z_op_accepted, 'z_de_acc': z_de_accepted, 'z_tt_acc': z_tt_accepted,
                    'z_cur_acceppted': z_cur_accepted, 'z_cur_archived': z_cur_archived}
    '''

    mpl.rcParams['legend.fontsize'] = 10

    # path = 'z_pickles\\Run_15_11_reactionFactor_09_400iterations_finalTemp550516\\'
    path = 'z_pickles\\Run_15_11_reactionFactor_098\\'
    # file = 'z_pickle2.pkl'
    file_accepted = 'z_pickle_accepted.pkl'
    file_current = 'z_pickle_current.pkl'

    solution_accepted = load_file_from_pickle(path+file_accepted)
    solution_current = load_file_from_pickle(path+file_current)
    z_op_acc, z_de_acc, z_tt_acc = solution_accepted['z_op'], solution_accepted['z_de'], solution_accepted['z_tt']
    z_op_cur, z_de_cur, z_tt_cur = solution_current['z_op'], solution_current['z_de'], solution_current['z_tt']
    # z_tt_cur = z_tt_cur[1:]
    # z_de_cur = z_de_cur[1:]
    # z_op_cur = z_op_cur[1:]
    nr_iterations = len(z_op_acc)
    points_per_frame = 20

    rest = nr_iterations % points_per_frame
    shape = (int(nr_iterations / points_per_frame), points_per_frame)

    if rest == 0:
        z_op_shaped_acc, z_de_shaped_acc, z_tt_shaped_acc = reshape_input_data_shaped(z_op_acc, z_de_acc, z_tt_acc, shape)
        z_op_shaped_cur, z_de_shaped_cur, z_tt_shaped_cur = reshape_input_data_shaped(z_op_cur, z_de_cur, z_tt_cur, shape)
    else:
        z_op_rest_acc, z_de_rest_acc, z_tt_rest_acc = z_op_acc[-rest:], z_de_acc[-rest:], z_tt_acc[-rest:]
        z_op_rest_cur, z_de_rest_cur, z_tt_rest_cur = z_op_cur[-rest:], z_de_cur[-rest:], z_tt_cur[-rest:]

        z_op_shaped_acc, z_de_shaped_acc, z_tt_shaped_acc = z_op_acc[:-rest], z_de_acc[:-rest], z_tt_acc[:-rest]
        z_op_shaped_cur, z_de_shaped_cur, z_tt_shaped_cur = z_op_cur[:-rest], z_de_cur[:-rest], z_tt_cur[:-rest]

        shape_rest = (rest,)
        z_op_shaped_acc, z_de_shaped_acc, z_tt_shaped_acc = reshape_input_data_shaped(z_op_shaped_acc, z_de_shaped_acc, z_tt_shaped_acc, shape)
        z_op_rest_acc, z_de_rest_acc, z_tt_rest_acc = reshape_input_data_rest(z_op_rest_acc, z_de_rest_acc, z_tt_rest_acc, shape_rest)

        z_op_shaped_cur, z_de_shaped_cur, z_tt_shaped_cur = reshape_input_data_shaped(z_op_shaped_cur, z_de_shaped_cur, z_tt_shaped_cur, shape)
        z_op_rest_cur, z_de_rest_cur, z_tt_rest_cur = reshape_input_data_rest(z_op_rest_cur, z_de_rest_cur, z_tt_rest_cur, shape_rest)

    fig = plt.figure()
    three_dimensional = True
    if three_dimensional:
        ax = fig.gca(projection='3d')
        ax.set_xlabel('operational cost')
        ax.set_xlim(min(z_op_acc)-300, max(z_op_acc)+300)
        ax.set_ylabel('deviation timetable')
        ax.set_ylim(min(z_de_acc)-300, max(z_de_acc)+300)
        ax.set_zlabel('total travel time')
        ax.set_zlim(min(z_tt_acc)-10000, max(z_tt_acc)+10000)


    greyish_blue = '#5e819d'
    royal_blue = '#0504aa'
    dark_magenta = '#960056'

    intensity = np.arange(0.5, 1.001, (1.001-0.5)/(shape[0]+1))
    for i in range(0, shape[0]+1):
        if i <= shape[0]-1:
            ax.scatter(z_op_shaped_acc[i, :], z_de_shaped_acc[i, :], z_tt_shaped_acc[i, :], c=dark_magenta, alpha=intensity[i])
            ax.scatter(z_op_shaped_cur[i, :], z_de_shaped_cur[i, :], z_tt_shaped_cur[i, :], c=greyish_blue, alpha=intensity[i])

        else:
            if rest == 0:
                continue
            else:
                ax.scatter(z_op_rest_acc, z_de_rest_acc, z_tt_rest_acc, c=dark_magenta, alpha=intensity[i])
                ax.scatter(z_op_rest_cur, z_de_rest_cur, z_tt_rest_cur, c=greyish_blue, alpha=intensity[i])


    ax.legend()
    filename = 'results1.png'
    plt.savefig(path + filename)
    plt.show()


def load_file_from_pickle(path):
    import pickle

    f = open(path, 'rb')
    solution = pickle.load(f)

    return solution


def reshape_input_data_shaped(z_op_shaped, z_de_shaped, z_tt_shaped, shape):

    z_op_shaped = np.reshape(np.array(z_op_shaped), shape)
    z_de_shaped = np.reshape(np.array(z_de_shaped), shape)
    z_tt_shaped = np.reshape(np.array(z_tt_shaped), shape)

    return  z_op_shaped, z_de_shaped, z_tt_shaped


def reshape_input_data_rest(z_op_rest, z_de_rest, z_tt_rest, shape_rest):

    z_op_rest = np.reshape(np.array(z_op_rest), shape_rest)
    z_de_rest = np.reshape(np.array(z_de_rest), shape_rest)
    z_tt_rest = np.reshape(np.array(z_tt_rest), shape_rest)

    return z_op_rest, z_de_rest, z_tt_rest


def generate_the_input(nr_of_solutions):

    z_op = random_np_array_of_shape(nr_of_solutions, solution_range=100)
    z_de = random_np_array_of_shape(nr_of_solutions, solution_range=30000*2)
    z_tt = random_np_array_of_shape(nr_of_solutions, solution_range=30000)

    return z_op, z_de, z_tt


def random_np_array_of_shape(nr_of_solutions, solution_range):
    z = np.random.random((nr_of_solutions,)) * solution_range
    for rnd in z:
        rnd = np.round(rnd * solution_range, 1)

    return z


if __name__ == '__main__':
    main()
