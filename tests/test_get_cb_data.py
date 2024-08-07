from uscounties.countylist import CountyList
import uscounties.get_cb_data as gcb

def test_years():
    assert gcb.YEARS[-1] >= 2022
    assert len(gcb.YEARS) == len(set(gcb.YEARS))

def test_downloads_up_to_date():
    assert CountyList.CB_VINTAGES == gcb.YEARS, set(CountyList.CB_VINTAGES).symmetric_difference(set(gcb.YEARS))
