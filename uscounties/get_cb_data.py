"""
Functions for getting county lists from U.S. Census Burearu (cb) files.
"""

from enum import Enum
import io
import logging
from pathlib import Path
import tempfile
from typing import Optional
import zipfile

import requests

from uscounties import basepath
from uscounties.countylist import CountyList

logger = logging.getLogger(__name__)


class FileFormats(Enum):
    SHP = "shp"
    KML = "kml"


OLD_SHP_FILES = {
    1990: ("https://www2.census.gov/geo/tiger/PREVGENZ/co/co90shp/co99_d90_shp.zip", "co99_d90", "cb_1990_us_county_20m"),
    2000: ("https://www2.census.gov/geo/tiger/PREVGENZ/co/co00shp/co99_d00_shp.zip", "co99_d00", "cb_2000_us_county_20m"),
    2010: ("https://www2.census.gov/geo/tiger/GENZ2010/gz_2010_us_050_00_20m.zip", "gz_2010_us_050_00_20m", "cb_2010_us_county_20m"),
    2013: ("https://www2.census.gov/geo/tiger/GENZ2013/cb_2013_us_county_20m.zip", "cb_2013_us_county_20m", "cb_2013_us_county_20m")
}


def get_modern_url(yr: int, fmt: FileFormats = FileFormats.SHP):
    if fmt == FileFormats.SHP:
        return f"https://www2.census.gov/geo/tiger/GENZ{yr}/shp/cb_{yr}_us_county_20m.zip"
    assert fmt == FileFormats.KML, fmt
    return f"https://www2.census.gov/geo/tiger/GENZ{yr}/kml/cb_{yr}_us_county_20m.zip"


YEARS = []

class YearsMethod(Enum):
    SIMPLE = "simple"
    SHP = "shp"
    KML = "kml"
    BOTH = "both"
    
def get_years(method: YearsMethod = YearsMethod.SIMPLE):
    global YEARS; YEARS = [yr for yr in OLD_SHP_FILES]

    def year_exists_simple(yr):
        r = requests.head(get_modern_url(yr))
        return r.ok

    def year_exists_full(yr, fmt=FileFormats.SHP):
        try:
            r = requests.get(get_modern_url(yr, fmt=fmt), stream=True); ok = r.ok
            z = zipfile.ZipFile(io.BytesIO(r.content)); z.close()
        except Exception as e:
            ok = False
        finally:
            r.close()
        return ok
    
    if method == YearsMethod.SIMPLE:
        while year_exists_simple(YEARS[-1] + 1):
            YEARS.append(YEARS[-1] + 1)
    elif method == YearsMethod.BOTH:
        while (year_exists_full(YEARS[-1] + 1,fmt=FileFormats.SHP) or 
               year_exists_full(YEARS[-1] + 1,fmt=FileFormats.KML)):
            YEARS.append(YEARS[-1] + 1)
    else:
        while year_exists_full(YEARS[-1] + 1, fmt=FileFormats(method.name)):
            YEARS.append(YEARS[-1] + 1)

get_years()


def get_start_year(fmt: FileFormats = FileFormats.SHP):
    if (fmt == FileFormats.SHP):
        return YEARS[0]
    else:
        assert (fmt == FileFormats.KML)
        return 2013
    

def update_census_list(yr: int, save_dir: Path=basepath, download_dirname: Optional[Path]=None, 
                       fmt: FileFormats=FileFormats.SHP, delete_tempdir: bool=True):
    """Creates or updates a list of U.S. Census Bureau counties for yr in save_dir.

    Parameters
    ----------
    yr : int
        Vintage year of the county list to be created or updated.
    save_dir : Path
        Directory in which to save the county list. Defaults to uscounties.basepath.
    download_dirname : Optional[Path]
        If provided, the original U.S. Census files will be download here. If not 
        provided, will create a temporary directory in the working directory that 
        should largely be invisible to the user.
    fmt : FileFormats
        Which type of file to download from the U.S. Census bureau. Defaults to 
        FileFormats.SHP.
    delete_tempdir : bool
        If True and a temporary directory is created, that directory will be 
        automatically deleted once data processing is complete. If False and a 
        temporary directory is created, that directory will persist until the 
        user manually deletes it.

    Returns
    -------
    CountyList
    """
    assert yr in YEARS, f"Data for {yr} is not known to be available. Available years: {YEARS}"
    call_func = _update_census_list_shp if fmt == FileFormats.SHP else _update_census_list_kml
    if not download_dirname:
        with tempfile.TemporaryDirectory(dir='.', delete=delete_tempdir) as tmpdirname:
            return call_func(yr, save_dir, tmpdirname)
    
    return call_func(yr, save_dir, download_dirname)


def _update_census_list_shp(yr: int, save_dir: Path, download_dirname: Path):
    _download_shp_file(yr, download_dirname)
    return _save_census_list_from_shp(yr, save_dir, download_dirname)


def _download_shp_file(yr: int, dirname: Path):
    url = None; orig_name = None; new_name = None
    if yr in OLD_SHP_FILES:
        url, orig_name, new_name = OLD_SHP_FILES[yr]
    if url is None:
        url = get_modern_url(yr, fmt = FileFormats.SHP)

    try:
        r = requests.get(url, stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(dirname)
        if (orig_name is not None) and (orig_name != new_name):
            for fn in Path(dirname).glob(f"{orig_name}.*"):
                fn.rename(Path(dirname, f"{new_name}{fn.suffix}"))
    except Exception as e:
        logger.error(f"Unable to download SHP for year {yr}, because {e!r}")
        raise


def _save_census_list_from_shp(yr: int, save_dir: Path, download_dirname: Path):
    for filepath in Path(download_dirname).glob(f"*.shp"):
        if int(filepath.stem.split("_")[1]) != yr:
            continue
        county_list = CountyList.load_from_census_shp(filepath)
        save_filename = "_".join(filepath.stem.split("_")[:4])
        county_list.save(save_dir / f"{save_filename}.parquet")
        return county_list
    msg = f"Did not find a .shp file for {yr}"; logger.error(msg)
    raise Exception(msg)


def _update_census_list_kml(yr: int, save_dir: Path, download_dirname: Path):
    _download_kml_file(yr, download_dirname)
    return _save_census_list_from_kml(yr, save_dir, download_dirname)


def _download_kml_file(yr: int, dirname: Path):
    try:
        if yr < 2019:
            r = requests.get(f"https://www2.census.gov/geo/tiger/GENZ{yr}/cb_{yr}_us_county_20m.kmz", stream=True)
            zippath = Path(dirname) / f"cb_{yr}_us_county_20m.zip"
            with open(zippath, "wb") as f:
                f.write(r.content)
            with zipfile.ZipFile(zippath, 'r') as z:
                z.extractall(dirname)
        else:
            r = requests.get(get_modern_url(yr, fmt=FileFormats.KML))
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(dirname)
    except Exception as e:
        logger.error(f"Unable to download KML for year {yr}, because {e!r}")
        raise


def _save_census_list_from_kml(download_dirname: Path, yr: int, save_dir: Path=basepath):
    for filepath in Path(download_dirname).glob(f"*.kml"):
        if int(filepath.stem.split("_")[1]) != yr:
            continue
        county_list = CountyList.load_from_census_kml(filepath)
        save_filename = "_".join(filepath.stem.split("_")[:4])
        county_list.save(save_dir / f"{save_filename}.parquet")
        return county_list
    msg = f"Did not find a .kml file for {yr}"; logger.error(msg)
    raise Exception(msg)


def update_census_lists(save_dir: Path=basepath, download_dirname: Optional[Path]=None, 
                        fmt: FileFormats=FileFormats.SHP, start_year: Optional[int]=None, 
                        hard_fail: bool=False, delete_tempdir: bool=True):
    """Creates or updates lists of U.S. Census Bureau counties from start_year to 
    the present.

    Parameters
    ----------
    save_dir : Path
        Directory in which to save the county lists. Defaults to uscounties.basepath.
    download_dirname : Optional[Path]
        If provided, the original U.S. Census files will be download here. If not 
        provided, will create a temporary directory in the working directory that 
        should largely be invisible to the user (assuming delete_tempdir is True).
    fmt : FileFormats
        Which type of file to download from the U.S. Census bureau. Defaults to 
        FileFormats.SHP.
    start_year : Optional[int]
        Year from which to start downloading lists of counties. Defaults to the 
        return value of get_start_year(fmt=fmt).
    hard_fail : bool
        If True, exceptions are raised, thereby stopping processing. If False, 
        exceptions are logged and processing continues with subsequent years.
    delete_tempdir : bool
        If True and a temporary directory is created, that directory will be 
        automatically deleted once data processing is complete. If False and a 
        temporary directory is created, that directory will persist until the 
        user manually deletes it.
    """
    start_year = start_year if start_year else get_start_year(fmt=fmt)

    assert start_year in YEARS, (f"Data for {start_year} is not known to be available. "
                                 f"Available years: {YEARS}")
    if not download_dirname:
        with tempfile.TemporaryDirectory(dir='.', delete=delete_tempdir) as tmpdirname:
            return _update_census_lists(save_dir, tmpdirname, fmt, start_year, hard_fail=hard_fail)
    
    return _update_census_lists(save_dir, download_dirname, fmt, start_year, hard_fail=hard_fail)


def _update_census_lists(save_dir: Path, download_dirname: Path, fmt: FileFormats, 
                         start_year: int, hard_fail: bool):
    download_func = None; load_func = None
    if fmt == FileFormats.SHP:
        download_func = _download_shp_file
        load_func = CountyList.load_from_census_shp
    else:
        assert fmt == FileFormats.KML, fmt
        download_func = _download_kml_file
        load_func = CountyList.load_from_census_kml
    
    downloaded_yrs = []
    for yr in YEARS:
        if yr < start_year:
            continue
        try:
            download_func(yr, download_dirname)
            downloaded_yrs.append(yr)
        except Exception as e:
            logger.error(f"Unable to download data for year {yr}, because {e}")
            if hard_fail:
                raise

    for filepath in Path(download_dirname).glob(f"*.{fmt.value}"):
        yr = int(filepath.stem.split("_")[1])
        if not (yr in downloaded_yrs):
            continue
        try:
            county_list = load_func(filepath)
        except Exception as e:
            logger.error(f"Unable to parse data for year {yr}, because {e}")
            if hard_fail:
                raise
            continue
        save_filename = "_".join(filepath.stem.split("_")[:4])
        county_list.save(save_dir / f"{save_filename}.parquet")

    return

