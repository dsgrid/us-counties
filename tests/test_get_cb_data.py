import pandas as pd
import shlex
import subprocess
import sys
import tempfile

from uscounties import basepath
from uscounties.countylist import CountyList
import uscounties.get_cb_data as gcb

from tests import tests_basepath


def assert_census_lists_equal(left, right):
    assert left.columns == right.columns
    assert left.description == right.description
    assert left.county_set == right.county_set
    pd.testing.assert_frame_equal(left, right)


def test_years():
    assert gcb.YEARS[-1] >= 2022
    assert len(gcb.YEARS) == len(set(gcb.YEARS))


def test_downloads_up_to_date():
    assert CountyList.CB_VINTAGES == gcb.YEARS, set(CountyList.CB_VINTAGES).symmetric_difference(set(gcb.YEARS))


def test_update_old_shp():
    yr = 1990
    with tempfile.TemporaryDirectory(dir=tests_basepath, delete=True) as tempdirname:
        census_list = gcb.update_census_list(yr, save_dir=tempdirname, download_dirname=tempdirname, delete_tempdir=True)
    expected = CountyList.load_cb_vintage(yr)
    assert_census_lists_equal(census_list, expected)


def test_update_latest():

    def run_command(cmd, cwd=None):
        command = shlex.split(cmd, posix="win" not in sys.platform)
        return subprocess.call(command, cwd=cwd)

    script_path = basepath / "bin" / "update_cb_counties.py"
    yr = gcb.YEARS[-1]
    
    with tempfile.TemporaryDirectory(dir=tests_basepath, delete=True) as kml_dir:
        cmd = f"python {script_path} -sd {kml_dir} -ff kml -sy {yr}"
        assert not run_command(cmd, cwd=kml_dir)
        kml_census_list = CountyList.load(kml_dir / f"cb_{yr}_us_county.parquet")

    with tempfile.TemporaryDirectory(dir=tests_basepath, delete=True) as shp_dir:
        cmd = f"python {script_path} -sd {shp_dir} -ff shp -sy {yr}"
        assert not run_command(cmd, cwd=shp_dir)
        shp_census_list = CountyList.load(shp_dir / f"cb_{yr}_us_county.parquet")

    expected = CountyList.load_cb_vintage(yr)
    assert_census_lists_equal(kml_census_list, expected)
    assert_census_lists_equal(shp_census_list, expected)
        