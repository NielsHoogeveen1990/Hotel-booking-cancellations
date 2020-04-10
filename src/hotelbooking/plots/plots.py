import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def create_distplots(df, cols=4):
    """
    This function creates distribution plots for all numerical features.
    :param df: dataframe
    :param cols: specified amount of columns in the subplots.
    :return: seaborn distplots
    """
    num_vars = df.select_dtypes('number').columns

    if (len(num_vars) % cols) != 0:
        rows = (len(num_vars) // cols) + 1
    else:
        rows = (len(num_vars) // cols)

    fig, ax = plt.subplots(rows, cols, figsize=(20, 10))
    for variable, subplot in zip(num_vars.tolist(), ax.flatten()):
        sns.distplot(df[variable], ax=subplot)

    plt.tight_layout()
    plt.show()


def create_countplots(df, cols=2):
    """
    This function creates count plots for all categorical features.
    :param df: dataframe
    :param cols: specified amount of columns in the subplots
    :return: seaborn countplots
    """
    cat_vars = df.select_dtypes('object').columns

    if (len(cat_vars) % cols) != 0:
        rows = (len(cat_vars) // cols) + 1
    else:
        rows = (len(cat_vars) // cols)

    fig, ax = plt.subplots(rows, cols, figsize=(20, 10))
    for variable, subplot in zip(cat_vars.tolist(), ax.flatten()):
        sns.countplot(df[variable], ax=subplot, palette='Set2')

    plt.tight_layout()
    plt.show()


def plot_kde_categorical_target(df, target, cols=4):
    """
    This function creates KDE plots for all numerical features.
    For every class (target), a KDE will be plotted in a subplot.
    :param df: dataframe
    :param target: target
    :param cols: specified amount of columns in the subplots
    :return: seaborn KDE plots
    """
    num_vars = df.select_dtypes('number').columns

    if (len(num_vars) % cols) != 0:
        rows = (len(num_vars) // cols) + 1
    else:
        rows = (len(num_vars) // cols)

    target_uniques = np.unique(df[target].values)
    palette = sns.husl_palette(len(target_uniques))
    palette_dict = dict(zip(target_uniques, palette))

    fig, ax = plt.subplots(rows, cols, figsize=(20, 10))
    for variable, subplot in zip(num_vars.tolist(), ax.flatten()):
        for target_unique in target_uniques:
            sns.kdeplot(df.loc[lambda d: d[target] == target_unique][variable],
                        ax=subplot,
                        shade=True,
                        color=palette_dict[target_unique])

    fig.legend(labels=target_uniques)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, width=12, height=10):
    """
    This function create a correlation heatmap for all numerical features to detect multi-collinearity.
    :param df: dataframe
    :param width: width of the heatmap
    :param height: height of the heatmap
    :return: seaborn heatmap
    """
    fig, ax = plt.subplots(figsize=(width, height))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)

    matrix = np.triu(df.corr())
    sns.heatmap(df.corr(), cmap=colormap, annot=True, mask=matrix)
    plt.show()