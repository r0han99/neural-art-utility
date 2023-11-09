import urllib.request
import pandas as pd

def read_data():
    artist_links = pd.read_csv('../artist_image_links.csv')
    
    names = artist_links['name']
    artist_links = artist_links['image_links']

    return artist_links, names


def download_image(url,name):
    fullname = str(name)+".jpg"
    try:
        print(f"{name}", end=" ")
        urllib.request.urlretrieve(url,fullname)
    except:
        print(f"Failed, {link}")

    print("Found!")
    



if __name__ == "__main__":

    links, names = read_data()

    for link, name in zip(links, names):
        download_image(link, name)