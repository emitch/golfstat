import os, sys, scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from parsedata import gather, index_features_in_data
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.cross_validation import LeaveOneOut

# stat names corresponding to rankings, should be exluded in feature gathering
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

    loo = LeaveOneOut(n_obs)
    for train, test in loo:
        # fit given model
        mod = model
        mod.fit(stats[train, :], rankings[train, i])
        # test on held-out data point
        pred_rank = mod.predict(stats[test])
        errs[test,:] = pred_rank - rankings[test]
        # TODO: better quantification of error

    # TODO: analyze coefficients etc.

    return errs

def correlations_over_time(start, end, stats_list=None):
    # index features once
    if stats_list is None:
        # get all features from big index thing
        feature_as_index, index_as_feature, _ = index_features_in_data()
    else:
        # make a temporary index of the features we want to see
        # TODO: detect which stats are interesting and use those
        feature_as_index = {}
        index_as_feature = [None] * len(stats_list)
        for i, stat in enumerate(stats_list):
            feature_as_index[stat] = i
            index_as_feature[i] = stat

    n_features = len(feature_as_index)
    n_rankings = 3
    n_years = end-start+1

    # initialize 3D numpy array to hold correlations for each year
    corr = np.empty([n_features, n_rankings, n_years], dtype=float)
    for year in range(start, end+1):
        print('Gathering stats for %d\r' % year, end="")
        # get the good good
        stats, ranks, _, _ = gather(
            [str(year)], feature_as_index, index_as_feature)

        # calculate spearman correlations
        corr[:, :, year-start] = rank_correlation(stats, ranks)

    # Each correlation over time is a slice along dimension 3
    for rank in range(n_rankings):
        fig = plt.figure(figsize=(12,8))
        for stat in range(n_features):
            plt.plot(range(start, end+1), corr[stat, rank, :], '-')
        plt.title('Correlation with ' + rank_stats[rank] + ' over time by stat')
        plt.xlabel('Year')
        plt.ylabel('Spearman correlation')
        # shrink box and make legend,
        ax = beautify(fig)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        plt.legend(index_as_feature, frameon=False,
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
    stats, ranks, index_as_feature, feature_as_index = gather(years)

    # calculate correlations
    rank_corrs = rank_correlation(stats, ranks)

    # sort by correlation with all-around ranking
    sorted_indices = np.argsort(np.absolute(rank_corrs[:,1]))[::-1]
    rank_corrs = rank_corrs[sorted_indices,:]
    ordered_features = index_as_feature[sorted_indices]

    print('%45s:\tSpearman Correlation per Ranking' % 'stat')
    for i, corr in enumerate(rank_corrs):
        print('%45s:\t% .3f\t% .3f\t% .3f' %
            (ordered_features[i], corr[0], corr[1], corr[2]))

    ###########################################
    # Correlations of different stats over time
    ###########################################
    print('Tracking stats ')
    interesting_stats = ordered_features[:9]
    correlations_over_time(1980, 2015, interesting_stats)
