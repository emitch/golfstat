import os, sys, scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from parsedata import gather, index_stats_in_data
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.cross_validation import LeaveOneOut
from sklearn.preprocessing import Imputer

# stat names corresponding to rankings, should be exluded in stat gathering
rank_stats = ['All-Around Ranking', 'FedExCup Season Points', 'Money Leaders']

def beautify(fig):
    # Set background color to white
    fig.patch.set_facecolor((1,1,1))

    # Remove frame and unnecessary tick-marks
    ax = plt.subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tick_params(axis='y', which='both', left='off', right='off')
    plt.tick_params(axis='x', which='both', top='off', bottom='off')
    return ax

def rank_correlation(stats, rankings):
    """ calculate the correlation coefficient of each stat with each
    ranking and return them all """
    # get dimensions
    n_stats = stats.shape[1]
    n_rankings = rankings.shape[1]

    # get correlation of each stat/ranking combination
    corr = np.empty([n_stats, n_rankings])
    for i in range(n_stats):
        for j in range(n_rankings):
            corr[i,j], _ = scipy.stats.spearmanr(stats[:,i], rankings[:,j])

    return corr

def ordinal_loo(stats, rankings, model=LinearRegression()):
    """ Perform leave-one-out validation on an ordinal regression """
    n_obs = stats.shape[0]
    n_rankings = rankings.shape[1]  # should always be 3
    errs = np.empty([n_obs, n_rankings])

    # impute stats and rankings
    stats = Imputer().fit_transform(stats)
    rankings[np.isnan(rankings)] = 999

    loo = LeaveOneOut(n_obs)
    for train, test in loo:
        # fit given model
        reg = model
        reg.fit(stats[train, :], rankings[train, :])
        # test on held-out data point
        pred_rank = reg.predict(stats[test, :])
        errs[test,:] = pred_rank - rankings[test]
        # TODO: better quantification of error?

    # fit on entire dataset and return coefficients
    reg = model
    reg.fit(stats, rankings)

    return reg.coef_, errs

def coefs_over_time(start, end, stats_list=None, model=LinearRegression()):
    """" BROKEN FOR START < 1985.  NEED TO FIX IMPUTATION/NaN-HANDLING"""
    # index stats once
    if stats_list is None:
        # get all stats from big index thing
        stat_as_index, index_as_stat, _, _ = index_stats_in_data()
    else:
        # make a temporary index of the stats we want to see
        # TODO: detect which stats are interesting and use those
        stat_as_index = {}
        index_as_stat = [None] * len(stats_list)
        for i, stat in enumerate(stats_list):
            stat_as_index[stat] = i
            index_as_stat[i] = stat

    n_stats = len(stat_as_index)
    n_rankings = 3
    n_years = end-start+1

    # initialize 3D numpy array to hold correlations for each year
    coefs = np.empty([n_rankings, n_stats, n_years], dtype=float)
    errs = np.empty([n_rankings, n_years], dtype=float)
    for year in range(start, end+1):
        print('Fitting models for %d\r' % year, end="")
        # get the good good
        stats, ranks, _, _ = gather(
            [str(year)], stat_as_index, index_as_stat)

        coefs[:, :, year-start], err = ordinal_loo(stats, ranks)
        errs[:, year-start] = np.mean(err)

    # Each correlation over time is a slice along dimension 3
    for rank in range(n_rankings):
        fig = plt.figure(figsize=(12,8))
        for stat in range(n_stats):
            plt.plot(range(start, end+1), coefs[rank, stat, :], '-')
        plt.title('Regression dependence of ' + rank_stats[rank] + ' over time by stat')
        plt.xlabel('Year')
        plt.ylabel('Regression Coefficients')
        # shrink box and make legend,
        ax = beautify(fig)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        plt.legend(index_as_stat, frameon=False,
            bbox_to_anchor=(1.0, 0.75), loc=2, mode='expand')

    plt.show()
    return coefs, errs

def correlations_over_time(start, end, stats_list=None):
    # index stats once
    if stats_list is None:
        # get all stats from big index thing
        stat_as_index, index_as_stat, _, _ = index_stats_in_data()
    else:
        # make a temporary index of the stats we want to see
        # TODO: detect which stats are interesting and use those
        stat_as_index = {}
        index_as_stat = [None] * len(stats_list)
        for i, stat in enumerate(stats_list):
            stat_as_index[stat] = i
            index_as_stat[i] = stat

    n_stats = len(stat_as_index)
    n_rankings = 3
    n_years = end-start+1

    # initialize 3D numpy array to hold correlations for each year
    corr = np.empty([n_stats, n_rankings, n_years], dtype=float)
    for year in range(start, end+1):
        print('Gathering stats for %d\r' % year, end="")
        # get the good good
        stats, ranks, _, _ = gather(
            [str(year)], stat_as_index, index_as_stat)

        # calculate spearman correlations
        corr[:, :, year-start] = rank_correlation(stats, ranks)

    # Each correlation over time is a slice along dimension 3
    for rank in range(n_rankings):
        fig = plt.figure(figsize=(12,8))
        for stat in range(n_stats):
            plt.plot(range(start, end+1), corr[stat, rank, :], '-')
        plt.title('Correlation with ' + rank_stats[rank] + ' over time by stat')
        plt.xlabel('Year')
        plt.ylabel('Spearman correlation')
        # shrink box and make legend,
        ax = beautify(fig)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        plt.legend(index_as_stat, frameon=False,
            bbox_to_anchor=(1.0, 0.75), loc=2, mode='expand')

    plt.show()
    return corr

# main
if __name__ == "__main__":
    ###########################################
    # Sorted Correlations in recent years
    ###########################################

    # Load data
    years = ['2015','2014']
    stats, ranks, index_as_stat, stat_as_index = gather(years)

    # calculate correlations
    rank_corrs = rank_correlation(stats, ranks)

    # sort by correlation with all-around ranking
    sorted_indices = np.argsort(np.absolute(rank_corrs[:,1]))[::-1]
    rank_corrs = rank_corrs[sorted_indices,:]
    ordered_stats = index_as_stat[sorted_indices]

    print('%45s:\tSpearman Correlation per Ranking' % 'stat')
    for i, corr in enumerate(rank_corrs):
        print('%45s:\t% .3f\t% .3f\t% .3f' %
            (ordered_stats[i], corr[0], corr[1], corr[2]))

    ###########################################
    # Correlations of different stats over time
    ###########################################
    print('Tracking stats ')
    interesting_stats = ordered_stats[:9]
    #correlations_over_time(1980, 2015, interesting_stats)
    coefs_over_time(1985, 2013, model=LassoCV())
