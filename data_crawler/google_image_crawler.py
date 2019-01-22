from icrawler.builtin import GoogleImageCrawler
from data_fish import *
import os
down_path = '/hdd_2/fishApp/Data/'

for fName in fishName.keys():
    dirName = down_path + fName
    print(dirName)

    if os.path.isdir(dirName) == False:
        os.makedirs(dirName, exist_ok=True)

    google_crawler = GoogleImageCrawler(feeder_threads=1,\
                                         parser_threads=2,\
                                         downloader_threads=4,\
                                         storage={'root_dir' : dirName})
    for down_name in fishName[fName]:
        print(down_name)
        google_crawler.crawl(keyword=down_name, max_num=5000)