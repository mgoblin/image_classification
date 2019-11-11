# download.py

import os
import sys
import urllib3
from urllib.parse import urlparse
import pandas as pd
import itertools
import shutil

from urllib3.util import Retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

classes = ["cat", "fish"]
set_types = ["train", "test", "val"]


def download_image(url, klass, data_type):
    basename = os.path.basename(urlparse(url).path)
    filename = "{}/{}/{}".format(data_type, klass, basename)
    if not os.path.exists(filename):
        try:
            http = urllib3.PoolManager(
                retries=Retry(connect=2, read=2, redirect=5)
            )
            resp = http.request("GET", url, preload_content=False)

            if resp.status == 200:
                out_file = open(filename, "wb")
                shutil.copyfileobj(resp, out_file)
                out_file.close()
            else:
                print("Error downloading {} with http status {}".format(url, resp.status))
            resp.release_conn()
        except:
            print("Error downloading {} with error {}".format(url, sys.exc_info()[0]))


if __name__ == "__main__":
    if not os.path.exists("images.csv"):
        print("Error: can't find images.csv!")
        sys.exit(0)

    # get args and create output directory
    imagesDF = pd.read_csv("images.csv")

    for set_type, klass in list(itertools.product(set_types, classes)):
        path = "./{}/{}".format(set_type, klass)
        if not os.path.exists(path):
            print("Creating directory {}".format(path))
            os.makedirs(path)

    total = len(imagesDF)
    print("Downloading {} images".format(total))

    for index, row in imagesDF.iterrows():
        print("Loading image {} from {}".format(index + 1, row['url']))
        download_image(row['url'], row['klass'], row['data_type'])
        print("Loaded {} image from {}".format(index + 1, total))

    print("Done {} images".format(total))
    sys.exit(0)
