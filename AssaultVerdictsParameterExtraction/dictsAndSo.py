import json
DISTRICTS = "districts"
TEL_AVIV = 'תל אביב'
TA = "א\"ת"
AT = "ת\"א"
TEL_AVIV_YAFFO = "תל אביב-יפו"

BEER_SHEVA = 'באר שבע'
BS = "ב\"ש"
SB = "ש\"ב"

HAIFA = "חיפה"

JERUSALEM = "ירושלים"
YAM = "י-ם"
MAY = "ם-י"

NORTH = "צפון"
KRAYOT = "קריות"
TZFAT = "צפת"
data = {}
data[DISTRICTS] = []
data[DISTRICTS].append({
    TEL_AVIV: [TA, AT, TEL_AVIV,TEL_AVIV_YAFFO]})

data[DISTRICTS].append({
    BEER_SHEVA: [BS,SB,BEER_SHEVA]})

data[DISTRICTS].append({
    HAIFA: [HAIFA]})

data[DISTRICTS].append({JERUSALEM:[JERUSALEM,MAY,YAM]})

with open('data.txt', 'w') as outfile:
    json.dump(data, outfile)