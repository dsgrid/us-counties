from __future__ import annotations

import copy
from dataclasses import dataclass, fields, astuple
import logging
from pathlib import Path
from typing import Optional

import fastparquet
import geopandas as gpd
import numpy as np
import pandas as pd
from pykml import parser

from uscounties import basepath


logger = logging.getLogger(__name__)


@dataclass
class CountyListColumns:
    """Names of columns containing specific county data in a DataFrame"""
    
    fips: Optional[str]        # full 5-digit FIPS code for the county
    state_fips: Optional[str]  # 2-digit FIPS code for the state
    county_fips: Optional[str] # 3-digit FIPS code for the county

    county_name: Optional[str] # county name, in title case
    state_abbr: Optional[str]  # two-letter state abbreviation
    state_name: Optional[str]  # state name, in title case
        
    @property
    def list_colnames(self):
        result = []
        if self.fips:
            result.append(self.fips)
        if self.state_fips:
            result.append(self.state_fips)
        if self.county_fips:
            result.append(self.county_fips)
        if self.county_name:
            result.append(self.county_name)
        if self.state_abbr:
            result.append(self.state_abbr)
        if self.state_name:
            result.append(self.state_name)
        
        return result
    

@dataclass(order=True,frozen=True)
class County:
    fips: str
    state_fips: str
    county_fips: str
        
    county_name: str
    state_abbr: str
    state_name: str
        

class CountyList:
    STANDARDIZED_COLUMNS = CountyListColumns(
        "fips",
        "state_fips",
        "county_fips",
        "name",
        "state",
        "state_name")
    
    STATE_LOOKUP = pd.read_parquet(basepath / "state_lookup.parquet")

    CB_VINTAGES = [int(filename.stem.split('_')[1]) for filename in basepath.glob("cb_*_us_county.parquet")]
    
    def __init__(self, df, columns: CountyListColumns, description=""):
        self.check_column_names_standard(columns)
        # TODO: Check FIPS values
        self._df = df
        self._columns = columns
        self.fill_in_columns()
        self._description = description
        self._set = {County(*x[self._columns.list_colnames]) for _ind, x in self._df.iterrows()}
        if self.df_count != self.set_count:
            logger.warning(f"County list {self.description!r} contains duplicates. "
                           f"Dataframe count: {self.df_count}; Set count: {self.set_count}")
        
    @property
    def df(self):
        return self._df.copy()
    
    @property
    def columns(self):
        return copy.copy(self._columns)
    
    @property
    def description(self):
        return self._description
    
    @property
    def county_set(self):
        return copy.copy(self._set)
    
    @property
    def df_count(self):
        return len(self._df.index)
    
    @property
    def set_count(self):
        return len(self._set)
    
    @property
    def cb_vintage(self):
        if self.description.startswith("U.S. Census Bureau"):
            return int(self.description[-4:])
        return None
    
    def fill_in_columns(self):
        # Fill in sub-FIPS if have full FIPS
        if self._columns.fips and (not self._columns.state_fips):
            self._df.loc[:, self.STANDARDIZED_COLUMNS.state_fips] = self._df[self._columns.fips].apply(lambda x: x[:2])
            self._columns.state_fips = self.STANDARDIZED_COLUMNS.state_fips
        if self._columns.fips and (not self._columns.county_fips):
            self._df.loc[:, self.STANDARDIZED_COLUMNS.county_fips] = self._df[self._columns.fips].apply(lambda x: x[-3:])
            self._columns.county_fips = self.STANDARDIZED_COLUMNS.county_fips

        # Fill in State information
        if not self._columns.state_fips:
            if self._columns.state_abbr:
                self._df = self._df.merge(self.STATE_LOOKUP[[self.STANDARDIZED_COLUMNS.state_fips, self.STANDARDIZED_COLUMNS.state_abbr]],
                                          on=self.STANDARDIZED_COLUMNS.state_abbr)
                self._columns.state_fips = self.STANDARDIZED_COLUMNS.state_fips
            else:
                assert self._columns.state_name
                self._df = self._df.merge(self.STATE_LOOKUP[[self.STANDARDIZED_COLUMNS.state_fips, self.STANDARDIZED_COLUMNS.state_name]],
                                          on=self.STANDARDIZED_COLUMNS.state_name)
                self._columns.state_fips = self.STANDARDIZED_COLUMNS.state_fips
        
        assert self._columns.state_fips
        assert self._columns.county_fips

        # Fill in full FIPS
        if not self._columns.fips:
            self._df.loc[:, self.STANDARDIZED_COLUMNS.fips] = self._df[self._columns.state_fips] + self._df[self._columns.county_fips]
            self._columns.fips = self.STANDARDIZED_COLUMNS.fips
        
        # Fill in State names and abbreviations
        if not self._columns.state_abbr:
            self._df = self._df.merge(self.STATE_LOOKUP[[self.STANDARDIZED_COLUMNS.state_fips, self.STANDARDIZED_COLUMNS.state_abbr]], 
                                      on=self.STANDARDIZED_COLUMNS.state_fips)
            self._columns.state_abbr = self.STANDARDIZED_COLUMNS.state_abbr
        if not self._columns.state_name:
            self._df = self._df.merge(self.STATE_LOOKUP[[self.STANDARDIZED_COLUMNS.state_fips, self.STANDARDIZED_COLUMNS.state_name]], 
                                      on=self.STANDARDIZED_COLUMNS.state_fips)
            self._columns.state_name = self.STANDARDIZED_COLUMNS.state_name
        
        colnames = self.columns.list_colnames
        other_cols = [col for col in self._df.columns if not col in colnames]
        self._df = self._df[colnames + other_cols]
        
    def save(self, filename):
        fastparquet.write(filename, self._df, custom_metadata={"description": self._description})
    
    @classmethod
    def load(cls, filename):
        pf = fastparquet.ParquetFile(filename)
        df = pf.to_pandas()
        description = pf.key_value_metadata["description"]
        return CountyList(df, cls.STANDARDIZED_COLUMNS, description=description)
       
    def left_minus_right(self, other: CountyList, ignore_states=[]):
        result = self.county_set - other.county_set
        return [x for x in result if not x.state_abbr in ignore_states]
    
    def right_minus_left(self, other: CountyList, ignore_states=[]):
        result = other.county_set - self.county_set
        return [x for x in result if not x.state_abbr in ignore_states]

    def num_diffs(self, other: CountyList, ignore_states=[]):
        return (len(self.left_minus_right(other, ignore_states=ignore_states)) + 
                len(self.right_minus_left(other, ignore_states=ignore_states)))

    def closest_vintage(self, ignore_states=[], min_vintage=2010):
        """
        Returns the "cb_*_us_county.parquet" CountyList that is the closest (per 
        num_diffs(ignore_states=ignore_states)) to this CountyList.
        """
        current = current_count = None
        for vintage in CountyList.CB_VINTAGES:
            if vintage < min_vintage:
                continue
            other = CountyList.load_cb_vintage(vintage)
            count = self.num_diffs(other, ignore_states=ignore_states)
            logger.info(f"There are {count} diffs with {other.cb_vintage}. {current_count=}")
            if (current is None) or (count < current_count):
                logger.debug(f"Switching to {other.cb_vintage}")
                current = other
                current_count = count
        return current
    
    @classmethod
    def load_cb_vintage(cls, year):
        assert year in cls.CB_VINTAGES, f"No U.S. Census Bureau list of counties available for {year}"
        return cls.load(basepath / f"cb_{year}_us_county.parquet")
    
    @classmethod
    def load_csv(cls, filename, description="", 
                 columns: CountyListColumns = CountyListColumns(*[None]*6), 
                 keep_extraneous_columns = False,
                 **read_csv_kwargs):
        cls.check_columns(columns)
        
        df = pd.read_csv(filename,**read_csv_kwargs)
        cls.fixup_fips_codes(df, columns)
        
        df, new_columns = cls.standardize_column_names(df, columns, keep_extraneous_columns=keep_extraneous_columns)
        
        return CountyList(df, new_columns, description=description)
    
    @classmethod
    def load_xlsx(cls, filename, description="", 
                 columns: CountyListColumns = CountyListColumns(*[None]*6), 
                  keep_extraneous_columns = False,
                 **read_excel_kwargs):
        cls.check_columns(columns)
        
        df = pd.read_excel(filename,**read_excel_kwargs)
        cls.fixup_fips_codes(df, columns)
        
        df, new_columns = cls.standardize_column_names(df, columns, keep_extraneous_columns=keep_extraneous_columns)
        
        return CountyList(df, new_columns, description=description)

    @classmethod
    def load_from_census_shp(cls, filepath):
        filepath = Path(filepath)
        assert filepath.exists(), f"{filepath} does not exist"
        assert filepath.suffix == ".shp", f"{filepath} is not a .shp file"

        df = gpd.read_file(filepath)

        census_year = int(filepath.stem.split("_")[1])

        try:
            if census_year == 1990:
                columns = CountyListColumns(None, "ST", "CO", "NAME", None, None)
            elif census_year in [2000, 2010]:
                columns = CountyListColumns(None, "STATE", "COUNTY", "NAME", None, None)
            elif census_year <= 2019:
                columns = CountyListColumns("GEOID", "STATEFP", "COUNTYFP", "NAME", None, None)
            else:
                columns = CountyListColumns("GEOID", "STATEFP", "COUNTYFP", "NAME", "STUSPS", "STATE_NAME")
            df, new_columns = cls.standardize_column_names(df, columns)
        except: 
            logger.error(f"Failed to open {filepath} as a CensusList. Data head:\n{df.head(1).T}\n"
                         f"Expected columns: {columns}")
            raise
        
        return CountyList(df, new_columns, description=f"U.S. Census Bureau SHP for {census_year}")
    
    @classmethod
    def load_from_census_kml(cls, filepath):
        filepath = Path(filepath)
        assert filepath.exists(), f"{filepath} does not exist"
        assert filepath.suffix == ".kml", f"{filepath} is not a .kml file"
        with open(filepath) as f:
            root = parser.parse(f).getroot()

        counties = []
        for pm in root.Document.Folder.Placemark:
            item = dict()
            for attr in str(pm.description).split("<th>"):
                if attr.startswith("<"):
                    continue
                name, attr = attr.split("</th>\n<td>", 1)
                value, _remainder = attr.split("</td>", 1)
                item[name] = value
            counties.append(item)
        df = pd.DataFrame(counties)
        
        census_year = int(filepath.stem.split("_")[1])

        try:
            if census_year <= 2019:
                columns = CountyListColumns("GEOID","STATEFP","COUNTYFP","NAME",None,None)
            else:
                columns = CountyListColumns("GEOID","STATEFP","COUNTYFP","NAME","STUSPS","STATE_NAME")
            df, new_columns = cls.standardize_column_names(df, columns)
        except: 
            logger.error(f"Failed to open {filepath} as a CensusList. Data head:\n{df.head()}\n"
                         f"Expected columns: {columns}")
            raise
        
        return CountyList(df, new_columns, description=f"U.S. Census Bureau KML for {census_year}")
        
    @classmethod
    def check_columns(cls, columns: CountyListColumns):
        # columns must supply fips or county_fips plus one of state_fips, state_abbr, or state_name
        if columns.fips is None:
            if (columns.county_fips is None) or (
                (columns.state_fips is None) and (columns.state_abbr is None) and 
                (columns.state_name is None)):
                raise Exception("Must provide either the fips column or the county_fips"
                                "column and at least one state identifier.")
        
        # columns must supply county_name
        if columns.county_name is None:
            raise Exception("Must provide a county name.")
            
    @classmethod
    def fixup_fips_codes(cls, df, columns: CountyListColumns):
        if columns.fips:
            if pd.api.types.is_integer_dtype(df[columns.fips].dtype):
                df[columns.fips] = df[columns.fips].apply(lambda x: f"{x:05d}").astype(str)
            elif pd.api.types.is_float_dtype(df[columns.fips].dtype):
                df[columns.fips] = df[columns.fips].apply(
                    lambda x: f"{int(x):05d}" if not np.isnan(x) else ""
                ).astype(str)
        if columns.state_fips:
            if pd.api.types.is_integer_dtype(df[columns.state_fips].dtype):
                df[columns.state_fips] = df[columns.state_fips].apply(lambda x: f"{x:02d}").astype(str)
            elif pd.api.types.is_float_dtype(df[columns.state_fips].dtype):
                df[columns.state_fips] = df[columns.state_fips].apply(
                    lambda x: f"{int(x):02d}" if not np.isnan(x) else ""
                ).astype(str)
        if columns.county_fips:
            if pd.api.types.is_integer_dtype(df[columns.county_fips].dtype):
                df[columns.county_fips] = df[columns.county_fips].apply(lambda x: f"{x:03d}").astype(str)
            elif pd.api.types.is_float_dtype(df[columns.county_fips].dtype):
                df[columns.county_fips] = df[columns.county_fips].apply(
                    lambda x: f"{int(x):03d}" if not np.isnan(x) else ""
                ).astype(str)
            
    @classmethod
    def standardize_column_names(cls, df, columns: CountyListColumns, keep_extraneous_columns=False):
        result = []; result_columns = CountyListColumns(*[None]*6)
        
        if columns.fips:
            result.append(df[columns.fips])
            result[-1].name = cls.STANDARDIZED_COLUMNS.fips
            result_columns.fips = result[-1].name
        
        if columns.state_fips:
            result.append(df[columns.state_fips])
            result[-1].name = cls.STANDARDIZED_COLUMNS.state_fips
            result_columns.state_fips = result[-1].name
        
        if columns.county_fips:
            result.append(df[columns.county_fips])
            result[-1].name = cls.STANDARDIZED_COLUMNS.county_fips
            result_columns.county_fips = result[-1].name
            
        if columns.county_name:
            result.append(df[columns.county_name])
            result[-1].name = cls.STANDARDIZED_COLUMNS.county_name
            result_columns.county_name = result[-1].name
            
        if columns.state_abbr:
            result.append(df[columns.state_abbr])
            result[-1].name = cls.STANDARDIZED_COLUMNS.state_abbr
            result_columns.state_abbr = result[-1].name
            
        if columns.state_name:
            result.append(df[columns.state_name])
            result[-1].name = cls.STANDARDIZED_COLUMNS.state_name
            result_columns.state_name = result[-1].name
            
        result = pd.concat(result, axis=1)
        
        if keep_extraneous_columns:
            colnames = columns.list_colnames
            to_keep = [col for col in df.columns if not col in colnames]
            if to_keep:
                result = pd.concat([result, df[to_keep]], axis=1)
                
        return result, result_columns
    
    @classmethod
    def check_column_names_standard(cls, columns: CountyListColumns):
        if columns.fips:
            assert columns.fips == cls.STANDARDIZED_COLUMNS.fips, (columns.fips, cls.STANDARDIZED_COLUMNS.fips)
        if columns.state_fips:
            assert columns.state_fips == cls.STANDARDIZED_COLUMNS.state_fips, (columns.state_fips, cls.STANDARDIZED_COLUMNS.state_fips)
        if columns.county_fips:
            assert columns.county_fips == cls.STANDARDIZED_COLUMNS.county_fips, (columns.county_fips, cls.STANDARDIZED_COLUMNS.county_fips)
        if columns.county_name:
            assert columns.county_name == cls.STANDARDIZED_COLUMNS.county_name, (columns.county_name, cls.STANDARDIZED_COLUMNS.county_name)
        if columns.state_abbr:
            assert columns.state_abbr == cls.STANDARDIZED_COLUMNS.state_abbr, (columns.state_abbr, cls.STANDARDIZED_COLUMNS.state_abbr)
        if columns.state_name:
            assert columns.state_name == cls.STANDARDIZED_COLUMNS.state_name, (columns.state_name, cls.STANDARDIZED_COLUMNS.state_name)

    @classmethod
    def lower(cls, df, columns: CountyListColumns):
        result = df.copy()
        for col in [columns.county_name, columns.state_abbr, columns.state_name]:
            if col:
                result.loc[:,col] = result[col].apply(lambda x: x.lower())
        return result
            
            
def replace_state_lookup(county_list: CountyList):
    return (
        county_list.df[["state_fips", "state", "state_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .to_parquet("state_lookup.parquet", index=False)
    )


def fill_in_fips(
        df, columns: CountyListColumns, 
        target_vintage=CountyList.CB_VINTAGES[-1],
        include_old_vintages=False
    ):
    result = df.copy()
    logger.info(f"Filling in FIPS codes for\n{result}")
    tmp, tmp_columns = CountyList.standardize_column_names(
        df, columns, keep_extraneous_columns=False
    )
    tmp = CountyList.lower(tmp, tmp_columns).reset_index()
    tmp.loc[:, "fips"] = None
    merge_on = [
        x 
        for x in [getattr(tmp_columns, field.name) for field in fields(CountyListColumns)] 
        if x is not None
    ]
    logger.info(f"Starting from standardized version\n{tmp}")

    vintages = [target_vintage] 
    if include_old_vintages:
        for v in [1990, 2000]:
            if v != target_vintage:
                vintages.append(v)
    vintages += [v for v in CountyList.CB_VINTAGES if (v != target_vintage) and (v > 2000)]

    for yr in vintages:
        cb_counties = CountyList.load_cb_vintage(yr)
        cb_counties = CountyList.lower(cb_counties.df, columns=CountyList.STANDARDIZED_COLUMNS)
        cb_counties = cb_counties[merge_on + ["fips"]]
        cb_counties.rename(columns={"fips": "fips_cb"}, inplace=True)
        tmp = tmp.merge(cb_counties, on=merge_on, how="left")
        inds = tmp.fips.isna()
        tmp.loc[inds, "fips"] = tmp.loc[inds, "fips_cb"]
        logger.info(f"After joining vintage {yr} on {merge_on} have \n{tmp}")
        del tmp["fips_cb"]

    result = result.merge(tmp[["index", "fips"]], how="left", left_index=True, right_on="index")
    result = result.drop_duplicates()
    del result["index"]
    logger.info(f"After filling in FIPS codes and dropping duplicates:\n{result}")
    return result
        

def to_dataframe(iter_of_dataclass):
    list_of_dataclass = list(iter_of_dataclass)
    columns = [field.name for field in fields(list_of_dataclass[0])]
    return pd.DataFrame(
        [astuple(dc_obj) for dc_obj in list_of_dataclass], columns=columns
    )