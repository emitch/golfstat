from bs4 import BeautifulSoup as Soup
from scipy.stats import spearmanr
import requests, json, os
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LassoCV

def scrape_scorecards():
    """ mine the pgatour data endpoint for individual tournament scorecards
    and copy the .json files and folder structure to the disk """

    # format for scorecard url
    source_url = 'http://www.pgatour.com/data/r/'

    # parse base page to get each tournament
    source_soup = Soup(requests.get(source_url).text, 'html.parser')

    # open each linked tournament and continue searching
    for tourn in source_soup.find_all('a'):
        # skip non-tournament links
        tourn_path = tourn.get('href')
        if not tourn_path.strip('/').isdigit(): continue
        # parse tournament page and follow all 'year' links
        tourn_url = source_url + tourn_path
        tourn_soup = Soup(requests.get(tourn_url).text, 'html.parser')

        # follow each link, hoping for a scorecards folder
        for year in tourn_soup.find_all('a'):
            # skip non year links
            year_path = year.get('href')
            if not year_path.strip('/').isdigit(): continue
            if len(year_path) != 5: continue
            # get page response
            score_url = tourn_url + year_path + 'scorecards/'
            response = requests.get(score_url)
            # skip nonexistant scorecards
            if response.status_code == 404: continue
            # parse the page
            score_soup = Soup(response.text, 'html.parser')
            # locate each json page
            for link in score_soup.find_all('a'):
                # skip .xml versions
                linked_file = link.get('href')
                if not linked_file.endswith('.json'): continue
                # open json page
                data = requests.get(score_url + linked_file).text

                # create folder structure
                tourn_folder = 'scorecards/' + tourn_path
                if not os.path.exists(tourn_folder):
                    os.mkdir(tourn_folder)
                destination = tourn_folder + year_path
                if not os.path.exists(destination):
                    os.mkdir(destination)

                # save
                with open(destination + linked_file, 'w') as file:
                    file.write(data)
                # output data
                print('Saved %s\r' % (destination + linked_file), end='')

def scrape_courses():
    """ get the par and distance information for each course used in the
    tournaments with scorecard data.  scrape scorecards must be run first
    as we use the existing local folder structure to build the urls """
    # pga tour data endpoint
    baseurl = 'http://www.pgatour.com/data/r/'
    # iterate through tournament folders
    for tourn in os.listdir('scorecards/'):
        # skip hidden folders
        if tourn.startswith('.'): continue
        for year in os.listdir('scorecards/{0}/'.format(tourn)):
            # skip hidden folders
            if year.startswith('.'): continue
            # current location in folder structure
            path = tourn + '/' + year + '/course.json'
            print('Scraping %s\r' % path, end='')
            # concatenate with baseurl to get endpoint
            data = requests.get(baseurl + path).text
            with open('scorecards/' + path, 'w') as file: file.write(data)

def players_in_tourn(tourn_id, year):
    """ return an iterable of player ids corresponding to tournament
    participation """
    # navigate to tournament folder
    path = 'scorecards/{0}/'.format(tourn_id)
    # initialize list of players
    players = []
    for file in os.listdir(path + year):
        # add each player id to the list
        player_id = file.strip('.json')
        if player_id.isdigit(): players.append(player_id)

    return players


def course_stat_bias(data, tourn_id, stat_as_index, years=None):
    """ Calculate the spearman correlation of each stat with tournament
    performance on a given course, indicating how important each stat
    is for success at a tournament """
    # get years for tournament
    if years is None:
        years = [x for x in os.listdir('scorecards/' + tourn_id) if x.isdigit()]
    # get scores for each player
    scores = []
    for year in years:
        players = players_in_tourn(tourn_id, year)
        for player in players:
            # skip missing players
            if player not in data: continue
            if year not in data[player]: continue
            if tourn_id not in data[player][year]: continue

            # compile performance and stuff
            shots = data[player][year][tourn_id]['summary']['total_shots']
            rnds = data[player][year][tourn_id]['summary']['num_rounds']
            scores.append(float(shots)/float(rnds))

    # initialize vector of spearman correlations
    corrs = np.empty(len(stat_as_index))
    for stat in stat_as_index:
        # vector of stat for each player
        stats = []
        for year in years:
            players = players_in_tourn(tourn_id, year)
            for player in players:
                # skip missing players
                if player not in data: continue
                if year not in data[player]: continue
                if tourn_id not in data[player][year]: continue
                # compile performance and stuff
                try: val = parse_stat(data[player][year]['stats'][stat]['value'])
                except KeyError: val = np.nan
                stats.append(val)

        # calculate correlation
        corrs[stat_as_index[stat]], _ = spearmanr(scores, stats)

    return corrs

def parse_stat(raw_val):
    # parse stat to float, taking account of wonky formatting
    if '%' in raw_val: val = float(raw_val[:-1])/100
    elif '$' in raw_val: val = float(raw_val[1:])
    elif "'" in raw_val:
        # convert distance to inches
        split = raw_val.split("' ")
        val = int(split[0]) * 12 + int(split[1][:-1])
    else:
        try: val = float(raw_val)
        except: val = np.nan
    return val

def compile_unbiased_stats_and_results(data, stat_as_index):
    """ train a model to predict performance on the level of tournaments
    using individual player stats and their importance for each course """
    n_stats = len(stat_as_index)
    # First gather all the course biases, final feature vectors are player stats
    # modulated by the degree to which a given stat predicts success on a course
    biases = {}
    for tourn_id in os.listdir('scorecards/'):
        if not tourn_id.isdigit(): continue
        biases[tourn_id] = course_stat_bias(data, tourn_id, stat_as_index)

    # now compile feature vectors, iterating through players and looking up
    # course biases for each tournament, also get tournament data
    feature_vector_list = []
    scores = []
    for player in data:
        for year in data[player]:
            for tourn in data[player][year]:
                if not tourn.isdigit(): continue
                # make feature vectors
                fv = np.empty(n_stats)
                for stat in stat_as_index:
                    try: val = parse_stat(data[player][year]['stats'][stat]['value'])
                    except KeyError: val = np.nan
                    fv[stat_as_index[stat]] = biases[tourn][stat_as_index[stat]] * val
                # add to running list
                feature_vector_list.append(fv)
                # get results
                shots = data[player][year][tourn]['summary']['total_shots']
                rnds = data[player][year][tourn]['summary']['num_rounds']
                scores.append(float(shots)/float(rnds))

    # concatenate list into single np matrix
    biased_stats = np.vstack(feature_vector_list)
    scores = np.array(scores)

    return biased_stats, scores

def hole_stat_bias(data, tourn_id, years, h):
    raise
     """ Calculate the spearman correlation of each stat with tournament
    performance on a given hole, indicating how important each stat
    is for success on a hole """
    # get years for tournament
    if years is None:
        years = [x for x in os.listdir('scorecards/' + tourn_id) if x.isdigit()]
    # get scores for each player
    scores = []
    for year in years:
        players = players_in_tourn(tourn_id, year)
        for player in players:
            # skip missing players
            if player not in data: continue
            if year not in data[player]: continue
            if tourn_id not in data[player][year]: continue

            # compile performance and stuff
            score = data[player][year][tourn_id]['scorecard']['h'][h-1]['sc']

            scores.append()

    # initialize vector of spearman correlations
    corrs = np.empty(len(stat_as_index))
    for stat in stat_as_index:
        # vector of stat for each player
        stats = []
        for year in years:
            players = players_in_tourn(tourn_id, year)
            for player in players:
                # skip missing players
                if player not in data: continue
                if year not in data[player]: continue
                if tourn_id not in data[player][year]: continue
                # compile performance and stuff
                try: val = parse_stat(data[player][year]['stats'][stat]['value'])
                except KeyError: val = np.nan
                stats.append(val)

        # calculate correlation
        corrs[stat_as_index[stat]], _ = spearmanr(scores, stats)

    return corrs, info

if __name__ == '__main__':
    # load data
    import parsedata
    data = parsedata.player_data_from_years(
        ['2014', '2015', '2016'], dict_by_id=True)
    with open('stat_as_index.json', 'r') as f:
        stat_as_index = json.load(f)
    # compile and shit
    print('Compiling stats...')
    bs, sc = compile_unbiased_stats_and_results(data, stat_as_index)
    # impute NaNs
    print('Imputing...')
    imp = Imputer()
    bs = imp.fit_transform(bs)

    # train model and evaluate
    print('Training model...')
    n_obs = len(sc)
    pred = np.empty(n_obs)
    for train, test in KFold(n_obs, 100):
        # fit
        las = LassoCV()
        las.fit(bs[train,:], sc[train])
        # predict
        pred[test] = las.predict(bs[test,:])

    # average errors
    print('Calculating errors...')
    err = pred - sc
    rmse = np.nanmean(err ** 2) ** .5
    print('Root-mean-square-error: %.4d' % rmse)

