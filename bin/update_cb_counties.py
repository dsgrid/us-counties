import argparse
import logging
from pathlib import Path
import tempfile
from typing import Optional

from uscounties import basepath
from uscounties.get_cb_data import YEARS, FileFormats, update_census_lists

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates or updates lists of "
                 "U.S. Census Bureau counties from start_year to the present.")
    parser.add_argument("-sd", "--save-dir", type=Path, default=basepath, 
                        help="Directory in which to save the county lists. "
                        f"Defaults to {basepath}.")
    parser.add_argument("-dd", "--download-dirname", type=Path, default=None, 
                        help="If provided, the original U.S. Census files will "
                        "be download here. If not provided, will create a temporary "
                        "directory in the working directory that should largely "
                        "be invisible to the user (assuming delete_tempdir is True).")
    parser.add_argument("-ff", "--file-format", type=FileFormats, default=FileFormats.SHP,
                        help="Whether to parse census SHP or KML files to retrieve "
                        "county information. Valid values are " + 
                        ", ".join([f"{fmt.value!r}" for fmt in FileFormats]) + 
                        f". Default value: {FileFormats.SHP.value!r}")
    parser.add_argument("-sy", "--start-year", type=int, default=None, 
                        help="Year from which to start downloading lists of counties. "
                        "Defaults to the return value of get_start_year(fmt=fmt), i.e., "
                        f"{YEARS[0]} for FileFormats.SHP and 2013 for FileFormats.KML.")
    parser.add_argument("-hf", "--hard-fail", action="store_true", dest="hard_fail",
                        default=False, help="If True, exceptions are raised, thereby "
                        "stopping processing. If False (the default), exceptions "
                        "are logged and processing continues with subsequent years.")
    parser.add_argument("-kt", "--keep-tempdir", action="store_false", 
                        dest="delete_tempdir", default=True, 
                        help="Set this flag if you would like the temporary "
                        "directory created for data downloads to persist.")
    
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    update_census_lists(save_dir=args.save_dir, download_dirname=args.download_dirname, 
                        fmt=args.file_format, start_year=args.start_year, hard_fail=args.hard_fail,
                        delete_tempdir=args.delete_tempdir)
