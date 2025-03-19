import requests

def season_player_adv_stats(year:int) -> list:
    response = requests.get('https://barttorvik.com/getadvstats.php', params={
            'year': str(year),
            'csv': '1'
        })
    if response.status_code != 200:
        raise Exception(f'[{response.status_code}] {response.text}')
    return response.text


if __name__ == '__main__':
    import sys
    year = sys.argv[1]
    data = season_player_adv_stats(year)
    with open(f'/Users/andrewgrowney/Data/torvik/player_advstats/{year}.csv', 'w') as f:
        f.write(data)        
