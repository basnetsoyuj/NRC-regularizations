import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import OrderedDict
from pathlib import Path

# the path leading to where the experiment file are located
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "monospace"

DEFAULT_BASE_PATH = '../results'
DEFAULT_SAVE_PATH = './figures/'
DEFAULT_ENVS = ('hopper',)
DEFAULT_LINESTYLES = tuple(['solid' for _ in range(8)])
DEFAULT_COLORS = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:purple')
DEFAULT_Y_VALUE = 'AverageTestEpRet'
DEFAULT_SMOOTH = 1
y_to_y_label = {
    'NRC1': 'NRC1',
    'NRC2': 'NRC2',
    'NRC3_c': 'NRC3_last_epoch',
    'NRC3_epoch': 'NRC3',
    'trainError': 'Training MSE Loss',
    'testError': 'Testing MSE Error'
}


# TODO we provide a number of things to a function to do plotting
#  labels, the data folder for each label, the colors, the dashes
#  and then, the y label to use
#  and also what name to save and where to save (we put default values as macros in a certain file? )

def do_smooth(x, smooth):
    y = np.ones(smooth)
    z = np.ones(len(x))
    smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
    return smoothed_x


def combine_data_in_seeds(seeds, column_name, skip=0, smooth=1):
    vals_list = []
    for d in seeds:
        if isinstance(d, pd.DataFrame):
            vals_to_use = d[column_name].to_numpy().reshape(-1)
        else:
            vals_to_use = d[column_name].reshape(-1)
        # yhat = savgol_filter(vals_to_use, 21, 3)  # window size 51, polynomial order 3
        # TODO might want to use sth else...
        if smooth > 1 and smooth <= len(vals_to_use):
            yhat = do_smooth(vals_to_use, smooth)
        else:
            yhat = vals_to_use

        if skip > 1:
            yhat = yhat[::skip]
        vals_list.append(yhat)
    return np.concatenate(vals_list)


def quick_plot_with_wandb(dataset, variant, legends, colors=DEFAULT_COLORS, linestyles=DEFAULT_LINESTYLES,
                          base_data_folder=DEFAULT_BASE_PATH,
                          save_name_prefix='test_save_figure', save_name_suffix=None,
                          save_folder_path=DEFAULT_SAVE_PATH,
                          y_value=DEFAULT_Y_VALUE, verbose=True, ymin=None, ymax=None,
                          y_use_log=None, xlabel='Number of Updates', ylabel='NRC1-3', axis_font_size=10,
                          y_log_scale=False, x_always_scientific=True, smooth=1):
    full_path = os.path.join(base_data_folder, dataset, variant)
    print("Loading data from:", full_path)

    for i, y_to_plot in enumerate(y_value):
        csv_path = os.path.join(full_path, y_to_plot + '.csv')
        with open(csv_path, 'r') as f:
            df = pd.read_csv(f, delimiter=",", header=0)
            if y_to_plot in ['NRC1', 'NRC2', 'trainError', 'testError']:
                x_to_plot = df.iloc[:, 0].to_numpy().reshape(-1)
                y_to_plot = df.iloc[:, 1].to_numpy().reshape(-1)
            elif y_to_plot in ['NRC3_c']:
                x_to_plot = df["c"].to_numpy().reshape(-1)
                y_to_plot = df["NRC3_old"].to_numpy().reshape(-1)
            elif y_to_plot in ['NRC3_epoch']:
                x_to_plot = df["epoch"].to_numpy().reshape(-1)
                y_to_plot = df["NRC3_old"].to_numpy().reshape(-1)

        ax = sns.lineplot(x=x_to_plot, y=y_to_plot, n_boot=20, label=legends[i], color=colors[i],
                          linestyle=linestyles[i],
                          linewidth=2, marker='o', ms=4)

    plt.xlabel(xlabel, fontsize=axis_font_size)
    plt.ylabel(ylabel, fontsize=axis_font_size)
    if ymin is not None:
        ax.set_ylim(ymin=ymin)
    if ymax is not None:
        ax.set_ylim(ymax=ymax)
    if y_log_scale or (isinstance(y_use_log, list) and y_to_plot in y_use_log):
        plt.yscale('log')
    if x_always_scientific:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.yticks(fontsize=axis_font_size)
    plt.xticks(fontsize=axis_font_size)
    plt.legend(fontsize=axis_font_size)
    plt.tight_layout()
    plt.tight_layout()
    plt.grid(True)
    save_folder_path_with_y = os.path.join(save_folder_path, ylabel)
    if save_folder_path is not None:
        if not os.path.isdir(save_folder_path_with_y):
            path = Path(save_folder_path_with_y)
            path.mkdir(parents=True)
        if save_name_suffix:
            suffix = '_' + save_name_suffix + '.png'
        else:
            suffix = '.png'
        save_path_full = os.path.join(save_folder_path_with_y, save_name_prefix + '_' + ylabel + suffix)

        plt.savefig(save_path_full)
        if verbose:
            print(save_path_full)
        plt.close()
    else:
        plt.show()


def plot_full_datasets(dataset_group, variants, measure_group, legend_group,
                       figsize=(20, 5),
                       colors=DEFAULT_COLORS,
                       linestyles=DEFAULT_LINESTYLES,
                       linewidth=2,
                       marker=None,
                       ms=2,
                       base_data_folder=DEFAULT_BASE_PATH,
                       save_name_prefix=None,
                       save_name_suffix=None,
                       save_folder_path=DEFAULT_SAVE_PATH,
                       verbose=True,
                       ymin=None,
                       ymax=None,
                       y_use_log=None,
                       y_log_scale=False,
                       xlabel='Number of Updates',
                       ylabel=('NRC1-3',),
                       font_size=None,
                       x_always_scientific=True,
                       use_ratio=None,
                       smooth=1):
    font_size = font_size or {}
    num_columns = len(dataset_group[0])
    num_rows = len(measure_group) * len(dataset_group)
    fig, axes = plt.subplots(num_rows, num_columns, figsize=figsize)
    for l, datasets in enumerate(dataset_group):
        for i, y_values in enumerate(measure_group):
            for j, dataset in enumerate(datasets):
                full_path = os.path.join(base_data_folder, dataset, variants[l][j])
                print("Loading data from:", full_path)

                ax = axes[2 * l + i][j] if num_rows > 1 else axes[j]  # When num_rows = 1, ax is 1D array.
                ax2 = None
                if 'NRC3_epoch' in y_values and dataset not in ['carla1d', 'age']:
                    ax2 = ax.twinx()

                for k, y_metric in enumerate(y_values):
                    if dataset in ['carla1d', 'age'] and y_metric == 'NRC3_epoch':
                        continue
                    if dataset in ['reacher', 'swimmer', 'hopper']:
                        csv_path = os.path.join(full_path, y_metric + '.csv')
                        with open(csv_path, 'r') as f:
                            df = pd.read_csv(f, delimiter=",", header=0)
                            if y_metric in ['NRC1', 'NRC2', 'trainError', 'testError']:
                                x_to_plot = df.iloc[:, 0].to_numpy().reshape(-1)
                                y_to_plot = df.iloc[:, 1].to_numpy().reshape(-1)
                            elif y_metric in ['NRC3_c']:
                                x_to_plot = df["c"].to_numpy().reshape(-1)
                                x_to_plot = x_to_plot / x_to_plot.max()
                                y_to_plot = df["NRC3"].to_numpy().reshape(-1)
                            elif y_metric in ['NRC3_epoch']:
                                x_to_plot = df["epoch"].to_numpy().reshape(-1)
                                y_to_plot = df["NRC3"].to_numpy().reshape(-1)
                    elif dataset in ['carla2d']:
                        if y_metric in ['NRC1', 'NRC2', 'NRC3_epoch', 'trainError', 'testError']:
                            csv_path = os.path.join(full_path, 'Carla2D_100ep.csv')
                            with open(csv_path, 'r') as f:
                                df = pd.read_csv(f, delimiter=",", header=0)
                                x_to_plot = df['Epoch']
                                if y_metric in ['NRC1', 'NRC2']:
                                    y_to_plot = df[y_metric]
                                elif y_metric == 'NRC3_epoch':
                                    y_to_plot = df['NRC3']
                                elif y_metric == 'trainError':
                                    y_to_plot = df['Train Loss']
                                elif y_metric == 'testError':
                                    y_to_plot = df['Test Loss']
                        elif y_metric in ['NRC3_c']:
                            csv_path = os.path.join(full_path, 'Carla2D_gamma.csv')
                            with open(csv_path, 'r') as f:
                                df = pd.read_csv(f, delimiter=",", header=0)
                                x_to_plot = df['gamma']
                                x_to_plot = x_to_plot / x_to_plot.max()
                                y_to_plot = df["NRC3"]

                    elif dataset in ['carla1d']:
                        if y_metric in ['NRC1', 'NRC2', 'NRC3_epoch', 'trainError', 'testError']:
                            csv_path = os.path.join(full_path, 'Carla1D_100ep.csv')
                            with open(csv_path, 'r') as f:
                                df = pd.read_csv(f, delimiter=",", header=0)
                                x_to_plot = df['Epoch']
                                if y_metric in ['NRC1', 'NRC2']:
                                    y_to_plot = df[y_metric]
                                elif y_metric == 'NRC3_epoch':
                                    y_to_plot = df['NRC3']
                                elif y_metric == 'trainError':
                                    y_to_plot = df['Train Loss']
                                elif y_metric == 'testError':
                                    y_to_plot = df['Test Loss']

                    elif dataset in ['age']:
                        if y_metric in ['NRC1', 'NRC2', 'NRC3_epoch', 'trainError', 'testError']:
                            csv_path = os.path.join(full_path, 'Age1D.csv')
                            with open(csv_path, 'r') as f:
                                df = pd.read_csv(f, delimiter=",", header=0)
                                x_to_plot = df['Epoch']
                                if y_metric in ['NRC1', 'NRC2']:
                                    y_to_plot = df[y_metric]
                                elif y_metric == 'NRC3_epoch':
                                    y_to_plot = df['NRC3']
                                elif y_metric == 'trainError':
                                    y_to_plot = df['Train Loss']
                                elif y_metric == 'testError':
                                    y_to_plot = df['Test Loss']

                    total_size = x_to_plot.shape[0]
                    use_size = total_size
                    gap = 1
                    if use_ratio[l]:
                        use_size = int(total_size * use_ratio[l][i][j])

                    # Want to make x-axis length of mujoco datasets uniform with 1000
                    num_points = 200
                    if dataset in ['reacher', 'hopper', 'swimmer']:
                        gap = use_size // num_points
                        x_to_plot = x_to_plot

                    # Mujoco exps forget to square the nrc3
                    if dataset in ['reacher', 'hopper', 'swimmer'] and y_metric == 'NRC3_epoch':
                        y_to_plot = y_to_plot ** 2

                    x_to_plot = x_to_plot[:use_size:gap]
                    y_to_plot = y_to_plot[:use_size:gap]

                    if ax2 is not None:
                        if y_metric in ['NRC1', 'NRC2']:
                            ax_linestyle = 'solid' if y_metric == 'NRC1' else 'dashed'
                            sns.lineplot(ax=ax, x=x_to_plot, y=y_to_plot,
                                         n_boot=20, color=colors[0], label=legend_group[i][k],
                                         linestyle=ax_linestyle, linewidth=linewidth,
                                         marker=marker, ms=ms)
                            ax.tick_params(axis='y', labelcolor=colors[0])
                        elif y_metric in ['NRC3_epoch']:
                            sns.lineplot(ax=ax2, x=x_to_plot, y=y_to_plot,
                                         n_boot=20, color=colors[1], label=legend_group[i][k],
                                         linestyle='solid', linewidth=linewidth,
                                         marker=marker, ms=ms)
                            ax2.tick_params(axis='y', labelcolor=colors[1])
                    else:
                        sns.lineplot(ax=ax, x=x_to_plot, y=y_to_plot,
                                     n_boot=20, label=legend_group[i][k], color=colors[k],
                                     linestyle=linestyles[k], linewidth=linewidth,
                                     marker=marker, ms=ms)

                # if dataset in ['reacher', 'swimmer', 'hopper']:
                #     ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
                #     ax.xaxis.get_offset_text().set_position((1.0, 0))
                #     ax.xaxis.get_offset_text().set_size(8)
                if 'trainError' in y_values and dataset in ['reacher', 'carla1d']:
                    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
                    ax.yaxis.get_offset_text().set_size(8)


                if ax2 is not None:
                    lines_1, labels_1 = ax.get_legend_handles_labels()
                    lines_2, labels_2 = ax2.get_legend_handles_labels()
                    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best', fontsize=font_size.get('legend'))
                    ax2.get_legend().remove()
                    # TODO: Without the following line, why is there always a right y label on 3x1 subplot?????????
                    ax2.set_ylabel('')
                else:
                    ax.legend(fontsize=font_size.get('legend'))

                # ax.set_xlim(left=0)
                if ymin is not None and i == 0:
                    ax.set_ylim(ymin=ymin)
                if ymax is not None and i == 0:
                    ax.set_ylim(ymax=ymax)

                if 'NRC1' in y_values:
                    ax.set_yscale('log')

                if x_always_scientific:
                    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

                ax.tick_params(axis='both', which='major', direction='out',
                               labelsize=font_size.get('tick_label'))

                # if y_log_scale or (isinstance(y_use_log, list) and any([y in y_use_log for y in y_values])):
                #     ax.set_yscale('log')

                # xtl = ax.get_xticklabels()
                # ytl = ax.get_yticklabels()
                # if ytl:
                #     ytl[0].set_verticalalignment('bottom')
                #     ytl[-1].set_verticalalignment('top')
                # # if xtl:
                # #     xtl[0].set_horizontalalignment('left')

                ax.grid(True, linestyle='dashed')

                # Global format
                if i == 0:
                    if dataset in ['carla1d', 'carla2d']:
                        title = dataset.upper()
                        title = title[:5] + " " + title[5:]
                    elif dataset == 'age':
                        title = 'UTKface'
                    else:
                        title = dataset.upper()
                    ax.set_title(title, fontsize=font_size.get('title'), fontweight='bold')

                if (i + 1) * (l + 1) == num_rows:  # Only label x-axis on the bottom row
                    ax.set_xlabel(xlabel, fontsize=font_size.get('xlabel'), fontweight=None)
                else:  # TODO: Without the following line, why is there always a x label on the 3rd row?????????
                    ax.set_xlabel('')

                if j == 0:  # Only label y-axis on the first column
                    ax.set_ylabel(ylabel[i], fontsize=font_size.get('ylabel'), fontweight='bold')
                else:
                    ax.set_ylabel('')

    plt.tight_layout()
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=0)
    save_folder_path_with_y = os.path.join(save_folder_path)
    if save_folder_path is not None:
        if not os.path.isdir(save_folder_path_with_y):
            path = Path(save_folder_path_with_y)
            path.mkdir(parents=True)
        if save_name_suffix:
            suffix = '_' + save_name_suffix + '.png'
        else:
            suffix = '.png'
        save_path_full = os.path.join(save_folder_path_with_y, save_name_prefix + suffix)

        # plt.savefig(save_path_full)
        if verbose:
            print(save_path_full)
        plt.show()
        plt.close()
    else:
        plt.show()


def plot_multi_variants(datasets, variant_group, measures, legend_group,
                        figsize=(20, 5),
                        colors=DEFAULT_COLORS,
                        linestyles=DEFAULT_LINESTYLES,
                        linewidth=2,
                        marker=None,
                        ms=2,
                        base_data_folder=DEFAULT_BASE_PATH,
                        save_name_prefix=None,
                        save_name_suffix=None,
                        save_folder_path=DEFAULT_SAVE_PATH,
                        verbose=True,
                        ymin=None,
                        ymax=None,
                        y_use_log=None,
                        y_log_scale=False,
                        xlabel='Number of Updates',
                        ylabel=('NRC1-3',),
                        font_size=None,
                        x_always_scientific=True,
                        smooth=1):
    font_size = font_size or {}
    num_rows = len(datasets)
    num_columns = len(measures)
    fig, axes = plt.subplots(num_rows, num_columns, figsize=figsize)
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, num=5))[::-1]
    for i, dataset in enumerate(datasets):
        for j, y_metric in enumerate(measures):
            ax = axes[i][j] if num_rows > 1 else axes[j]  # When num_rows = 1, ax is 1D array.

            if dataset in ['carla2d']:
                full_path = os.path.join(base_data_folder, dataset, variant_group[i][0])
                if y_metric in ['NRC1', 'NRC2', 'NRC3_epoch', 'trainError', 'testError']:
                    csv_path = os.path.join(full_path, '5WD_Carla2D_150ep.csv')
                    for k in range(5):
                        with open(csv_path, 'r') as f:
                            df = pd.read_csv(f, delimiter=",", header=0)
                            x_to_plot = df['Epoch']
                            if y_metric == 'NRC1':
                                y_to_plot = df.iloc[:, 4 * k + 1].to_numpy().reshape(-1)
                            elif y_metric == 'NRC2':
                                y_to_plot = df.iloc[:, 4 * k + 2].to_numpy().reshape(-1)
                            elif y_metric == 'NRC3_epoch':
                                y_to_plot = df.iloc[:, 4 * k + 3].to_numpy().reshape(-1)
                            elif y_metric == 'trainError':
                                y_to_plot = df['Train Loss']
                            elif y_metric == 'testError':
                                y_to_plot = df['Test Loss']

                        size = x_to_plot.shape[0]
                        num_points = size
                        gap = size // num_points
                        x_to_plot = x_to_plot[::gap]
                        y_to_plot = y_to_plot[::gap]
                        legend = None
                        if j+1 == len(measures):
                            legend = legend_group[i][k]
                        sns.lineplot(ax=ax, x=x_to_plot, y=y_to_plot,
                                     n_boot=20, label=legend, color=colors[k],
                                     linestyle=linestyles[k], linewidth=linewidth,
                                     marker=marker, ms=ms)
            else:
                for k, variant in enumerate(variant_group[i]):
                    full_path = os.path.join(base_data_folder, dataset, variant)
                    print("Loading data from:", full_path)
                    if dataset in ['reacher', 'swimmer', 'hopper']:
                        csv_path = os.path.join(full_path, y_metric + '.csv')
                        with open(csv_path, 'r') as f:
                            df = pd.read_csv(f, delimiter=",", header=0)
                            if y_metric in ['NRC1', 'NRC2', 'trainError', 'testError']:
                                x_to_plot = df.iloc[:, 0].to_numpy().reshape(-1)
                                y_to_plot = df.iloc[:, 1].to_numpy().reshape(-1)
                            elif y_metric in ['NRC3_c']:
                                x_to_plot = df["c"].to_numpy().reshape(-1)
                                y_to_plot = df["NRC3"].to_numpy().reshape(-1)
                            elif y_metric in ['NRC3_epoch']:
                                x_to_plot = df["epoch"].to_numpy().reshape(-1)
                                y_to_plot = df["NRC3"].to_numpy().reshape(-1)
                                y_to_plot = y_to_plot**2

                    size = x_to_plot.shape[0]
                    num_points = size
                    gap = size // num_points
                    x_to_plot = x_to_plot[::gap] * 100
                    y_to_plot = y_to_plot[::gap]
                    legend = None
                    if j + 1 == len(measures):
                        legend = legend_group[i][k]
                    sns.lineplot(ax=ax, x=x_to_plot, y=y_to_plot,
                                 n_boot=20, label=legend, color=colors[k],
                                 linestyle=linestyles[k], linewidth=linewidth,
                                 marker=marker, ms=ms)

                    # ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
                    # ax.xaxis.get_offset_text().set_position((1.0, 0))
                    # ax.xaxis.get_offset_text().set_size(8)
                    ax.xaxis.set_ticks([0, int(2e5), int(4e5), int(6e5),int(8e5)])
                    ax.set_xticklabels([0, '200K', '400K', '600K', '800K'])


            ax.grid(True, linestyle='dashed')
            if dataset == 'carla2d':
                ax.set_yscale('log')
            if j+1 == len(measures):
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            # ax.set_xlim(left=0)
            if ymin is not None:
                ax.set_ylim(ymin=ymin)
            if ymax is not None:
                ax.set_ylim(ymax=ymax)
            # if y_log_scale or (isinstance(y_use_log, list) and any([y in y_use_log for y in y_values])):
            #     ax.set_yscale('log')
            if x_always_scientific:
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            ax.tick_params(axis='both', which='major', direction='out',
                           labelsize=font_size.get('tick_label', 10))

            # xtl = ax.get_xticklabels()
            # ytl = ax.get_yticklabels()
            # if ytl:
            #     ytl[0].set_verticalalignment('bottom')
            #     ytl[-1].set_verticalalignment('top')
            # if xtl:
            #     xtl[0].set_horizontalalignment('left')

            if j == 0:  # Only label x-axis on the bottom row
                title = dataset.upper() if dataset == 'swimmer' else 'CARLA 2D'
                ax.set_ylabel(title, fontsize=font_size.get('ylabel'), fontweight='bold')
            if i + 1 == num_rows:
                ax.set_xlabel(xlabel, fontsize=font_size.get('xlabel'), fontweight=None)
            if i == 0:
                ax.set_title(ylabel[j], fontsize=font_size.get('title'), fontweight='bold')
            # if j == 0:  # Only label y-axis on the first column
            #     ax.set_ylabel(ylabel[i], fontsize=font_size.get('y_label), fontweight='bold')
            # else:
            #     ax.set_ylabel('')

    plt.tight_layout()
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=0)
    save_folder_path_with_y = os.path.join(save_folder_path)
    if save_folder_path is not None:
        if not os.path.isdir(save_folder_path_with_y):
            path = Path(save_folder_path_with_y)
            path.mkdir(parents=True)
        if save_name_suffix:
            suffix = '_' + save_name_suffix + '.png'
        else:
            suffix = '.png'
        save_path_full = os.path.join(save_folder_path_with_y, save_name_prefix + suffix)

        # plt.savefig(save_path_full)
        if verbose:
            print(save_path_full)
        plt.show()
        plt.close()
    else:
        plt.show()
