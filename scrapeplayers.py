from bs4 import BeautifulSoup as bs
import json

html_file = open('Players.htm', 'r')

soup = bs(html_file, 'html.parser')
players = []
for player in soup.find_all('span', {'class':'name'}):
    a_tag = player.find_all('a')[0]
    players.append({'name':a_tag.string, 'link':a_tag.get('href')})

out_file = open('player.json', 'w')
json.dump(players, out_file)

out_file.close()
html_file.close()
