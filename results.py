from bs4 import BeautifulSoup as Soup
import requests, json, os

def scrape_scorecards():
    # format for scorecard url
    source_url = 'http://www.pgatour.com/data/r/'
    start_year, end_year = 1980, 2016

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
                # output results
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


def compile_scorecards():
    # Do some sort of pre-processing to make accessing the scorecards easier
    return None

if __name__ == '__main__':
    scrape_courses()

