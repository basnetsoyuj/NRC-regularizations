from quick_plot_helper import *
from log_alias import *
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

# Global variables:
default_performance_smooth = 5
# font_size = {'title': 18,
#              'legend': 12,
#              'xlabel': 15,
#              'ylabel': 15,
#              'tick_label': 12}
font_size = {'legend': 6}

data_folder = '../results'
save_folder = './figures/'

y_use_log = None

# dataset_group = [['reacher', 'swimmer', 'hopper'],
#                  ['carla2d', 'carla1d', 'age']]
# variant_group = [[reacher_final, swimmer_final, hopper_final],
#                  [carla2d_final, carla1d_final, age_final]]
# # use_ratio = [((3/8, 1/40, 5/8), (0.25, 1/40, 0.25)), None]
# # dataset_group = [['reacher', 'swimmer', 'hopper', 'carla2d', 'carla1d', 'age']]
# # variant_group = [[reacher_final, swimmer_final, hopper_final, carla2d_final, carla1d_final, age_final]]
# # # use_ratio = [((3/8, 1/40, 5/8, 1, 1, 1), (0.25, 1/40, 0.25, 1, 1, 1))]
# measure_group = [['NRC1', 'NRC2', 'NRC3_epoch'], ['trainError', 'testError']]
# legend_group = [['NRC1', 'NRC2', 'NRC3'], ['Train MSE', 'Test MSE']]
# xlabel = 'Epoch'
# ylabel = ['NRC1-3', 'Accuracy']
# # use_ratio = [None, None]
# plot_full_datasets(dataset_group, variant_group, measure_group, legend_group,
#                        figsize=None,
#                        linewidth=1,
#                        marker=None,
#                        ms=6,
#                        base_data_folder=data_folder,
#                        save_name_prefix=f'figure2',
#                        save_name_suffix=None,
#                        save_folder_path=save_folder,
#                        ymin=None,
#                        ymax=None,
#                        y_use_log=y_use_log,
#                        y_log_scale=False,
#                        xlabel=xlabel,
#                        ylabel=ylabel,
#                        font_size=font_size,
#                        x_always_scientific=False,
#                        use_ratio=[None, None]
#                        )
#
dataset_group = [['reacher', 'swimmer', 'hopper', 'carla2d']]
variant_group = [[reacher_final, swimmer_final, hopper_final, carla2d_final]]
# use_ratio = [((3/8, 1/40, 5/8, 1, 1, 1), (0.25, 1/40, 0.25, 1, 1, 1))]
measure_group = [['NRC3_c']]
legend_group = [['NRC3']]
xlabel = r'$\gamma / \lambda_{min}$'
ylabel = ['Last Epoch NRC3']
# use_ratio = [None, None]
plot_full_datasets(dataset_group, variant_group, measure_group, legend_group,
                       figsize=None,
                       linewidth=None,
                       marker=None,
                       ms=6,
                       base_data_folder=data_folder,
                       save_name_prefix=f'figure2',
                       save_name_suffix=None,
                       save_folder_path=save_folder,
                       ymin=None,
                       ymax=None,
                       y_use_log=y_use_log,
                       y_log_scale=False,
                       xlabel=xlabel,
                       ylabel=ylabel,
                       font_size=None,
                       x_always_scientific=False,
                       use_ratio=[None, None]
                       )


# datasets = ['swimmer', 'carla2d']
# variant_group = [[swimmer_wd0, swimmer_wd5e5, swimmer_wd5e4, swimmer_wd5e3, swimmer_final], [carla2d_wd]]
# measures = ['NRC1', 'NRC2', 'NRC3_epoch']
# legend_group = [[r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-5$', r'$\lambda_{WD}=5e-4$', r'$\lambda_{WD}=5e-3$', r'$\lambda_{WD}=1e-2$'],
#                 [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-4$', r'$\lambda_{WD}=5e-3$', r'$\lambda_{WD}=5e-2$', r'$\lambda_{WD}=1e-1$']]
# xlabel = 'Epoch'
# ylabel = ['NRC1', 'NRC2', 'NRC3']
# plot_multi_variants(datasets, variant_group, measures, legend_group,
#                     figsize=None,
#                     linewidth=None,
#                     marker=None,
#                     ms=6,
#                     base_data_folder=data_folder,
#                     save_name_prefix=f'figure4',
#                     save_name_suffix=None,
#                     save_folder_path=save_folder,
#                     ymin=None,
#                     ymax=None,
#                     y_use_log=None,
#                     y_log_scale=False,
#                     xlabel=xlabel,
#                     ylabel=ylabel,
#                     font_size=None,
#                     x_always_scientific=False)


