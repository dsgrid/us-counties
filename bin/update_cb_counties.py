import io
import logging
from pathlib import Path
import tempfile
import zipfile

import requests

from uscounties import basepath
from uscounties.countylist import CountyList


logger = logging.getLogger(__name__)
here = Path(__file__).absolute().parent

MAX_YEAR = 2023
USE_KML = False


def download_files_shp(dirname):
    old_files = [
        (1990, "https://www2.census.gov/geo/tiger/PREVGENZ/co/co90shp/co99_d90_shp.zip", "co99_d90", "cb_1990_us_county_20m"),
        (2000, "https://www2.census.gov/geo/tiger/PREVGENZ/co/co00shp/co99_d00_shp.zip", "co99_d00", "cb_2000_us_county_20m"),
        (2010, "https://www2.census.gov/geo/tiger/GENZ2010/gz_2010_us_050_00_20m.zip", "gz_2010_us_050_00_20m", "cb_2010_us_county_20m"),
        (2013, "https://www2.census.gov/geo/tiger/GENZ2013/cb_2013_us_county_20m.zip", "cb_2013_us_county_20m", "cb_2013_us_county_20m")
    ]
    for yr, url, orig_name, new_name in old_files:
        try:
            r = requests.get(url, stream=True)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(dirname)
            if orig_name != new_name:
                for fn in Path(dirname).glob(f"{orig_name}.*"):
                    fn.rename(Path(dirname, f"{new_name}{fn.suffix}"))
        except Exception as e:
            logger.error(f"Unable to download SHP for year {yr}, because {e!r}")
            continue

    for yr in range(2014,MAX_YEAR):
        try:
            r = requests.get(f"https://www2.census.gov/geo/tiger/GENZ{yr}/shp/cb_{yr}_us_county_20m.zip", stream=True)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(dirname)
        except Exception as e:
            logger.error(f"Unable to download SHP for year {yr}, because {e!r}")
            continue


def download_files_kml(dirname):
    for yr in range(2013,2019):
        try:
            r = requests.get(f"https://www2.census.gov/geo/tiger/GENZ{yr}/cb_{yr}_us_county_20m.kmz", stream=True)
            zippath = Path(dirname) / f"cb_{yr}_us_county_20m.zip"
            with open(zippath, "wb") as f:
                f.write(r.content)
            with zipfile.ZipFile(zippath, 'r') as z:
                z.extractall(dirname)
        except Exception as e:
            logger.error(f"Unable to download KML for year {yr}, because {e!r}")
            continue
    for yr in range(2019,MAX_YEAR):
        try:
            r = requests.get(f"https://www2.census.gov/geo/tiger/GENZ{yr}/kml/cb_{yr}_us_county_20m.zip", stream=True)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(dirname)
        except Exception as e:
            logger.error(f"Unable to download KML for year {yr}, because {e!r}")
            continue


def create_save_census_lists(dirname):
    for filename in Path(dirname).glob("*.shp"):
        county_list = CountyList.load_from_census_shp(filename)
        list_filename = "_".join(filename.stem.split("_")[:4])
        county_list.save(basepath / f"{list_filename}.parquet")
    
    for filename in Path(dirname).glob("*.kml"):
        county_list = CountyList.load_from_census_kml(filename)
        list_filename = "_".join(filename.stem.split("_")[:4])
        county_list.save(basepath / f"{list_filename}.parquet")


def main():
    with tempfile.TemporaryDirectory(dir=here, delete=True) as tmpdirname:
        if USE_KML: 
            download_files_kml(tmpdirname)
        else:
            download_files_shp(tmpdirname)
        create_save_census_lists(tmpdirname)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
