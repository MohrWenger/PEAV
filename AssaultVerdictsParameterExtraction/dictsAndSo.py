import json
DISTRICTS = "districts"
TEL_AVIV = 'תל אביב'
TA = "א\"ת"
AT = "ת\"א"
TEL_AVIV_YAFFO = "תל אביב-יפו"
OR_YEHUDA = "אור יהודה"
AZUR = "אזור"
EFAL = "אפעל"
BNEI_BRAK = "בני ברק"
GIVATAIM_NAME = "גבעתיים"
GIVATAIM = "גבעתי*ים"
HERTELYA_NAME  = "הרצליה"
HRTZELIA = "הרצלי*יה"
HERTZ = "הרצ\'"
KFAR_SHMARIAU = "כפר שמריהו"
KYRIAT_ONO = "קרי*ית אונו"
RAMAT_GAN_NAME = "רמת גן"
RAMAT_GAN = "רמת (- )*גן"
RAMAT_SHARON_NAME = "רמת השרון"
RAMAT_SHARON = "רמת (- )*השרון"
BAT_YAM_NAME = "בת ים"
BAT_YAM = "בת (- )*ים"


MERCAZ = "מרכז"
NATANYA = "נתניה"
NAT = "נת\'"
KFAR_SABA = "כפר סבא"
KFS = "כפ\"ס"
KS = "כ\"ס"

PETAH_TIKVA = "פתח תקווה"
PT = "פ\"ת"
RISHON_LETZION = "ראשון לציון"
RASHLATZ = "ראשל\"צ"
RSHLATZ = "רשל\"צ"
REHOVOT = "רחובות"
RH = "רח\'"
RAMLA = "רמלה"


DAROM = "דרום"
EILAT = "אילת"
EI = "אי\'"
ASHKELON = "אשקלון"
ASHDOD = "אשדוד"
ASHD = "אשד\'"
DIMONA = "דימונה"
KIRYAT_GAT = "קריית גת"
BEER_SHEVA = 'באר שבע'
BS = "ב\"ש"
SB = "ש\"ב"


TZAFON = "צפון"
BEIT_SHEAN = "בית שאן"
TVERIA = "טבריה"
MSADA = "מסעדה"
NATZRAT = "נצרת"
NTS = "נצ\'"
AFULA = "עפולה"
TZFAT = "צפת"
KATZRIN = "קצרין"
KIRYAT_SMONA = "קריית שמונה"

HAIFA = "חיפה"
HI= "חי\'"
HADERA = "חדרה"
ACO = "עכו"
KRAYOT = "קריות"

JERUSALEM = "ירושלים"
YAM = "י-ם"
MAY = "ם-י"
BEIT_SHEMESH = "בית (-)*שמש"

# data = {}
# data[DISTRICTS] = []
# data[DISTRICTS].append({
#     TEL_AVIV_YAFFO: [TA, AT, TEL_AVIV,TEL_AVIV]})
#
# data[DISTRICTS].append({
#     BEER_SHEVA: [BS,SB,BEER_SHEVA]})
#
# data[DISTRICTS].append({
#     HAIFA: [HAIFA]})
#
# data[DISTRICTS].append({JERUSALEM:[JERUSALEM,MAY,YAM]})
#

data = {}

data[TEL_AVIV_YAFFO] = TA+"|"+AT+"|"+TEL_AVIV+"|"+TEL_AVIV
data[BEER_SHEVA] = BS+"|"+SB+"|"+BEER_SHEVA
data[JERUSALEM] = JERUSALEM+"|"+MAY+"|"+YAM
data[PETAH_TIKVA] = PETAH_TIKVA+"|"+PT
data[KFAR_SABA] = KFAR_SABA+"|"+KFS+"|"+KS
data[NATZRAT] = NATZRAT+"|"+NTS
data[HAIFA] = HAIFA+"|"+HI
data[REHOVOT] = REHOVOT+"|"+RH
data[HERTELYA_NAME] = HERTELYA_NAME+"|"+HRTZELIA+"|"+HERTZ
data[EILAT] = EILAT+"|"+EI
data[NATANYA] = NATANYA+"|"+NAT
data[ASHDOD] = ASHDOD +"|"+ASHD
data[RISHON_LETZION] = RISHON_LETZION+"|"+RASHLATZ+"|"+RSHLATZ

with open('data.txt', 'w') as outfile:
    json.dump(data, outfile)

county_list = {}

county_list [TEL_AVIV_YAFFO] = TEL_AVIV_YAFFO+"|"+HRTZELIA+"|"+BAT_YAM
county_list [MERCAZ] = MERCAZ+"|"+PETAH_TIKVA+"|"+RISHON_LETZION+"|"+REHOVOT+"|"+RAMLA+"|"+NATANYA+"|"+KFAR_SABA
county_list [JERUSALEM] = JERUSALEM+"|"+BEIT_SHEMESH
county_list [TZAFON] = TZAFON+"|"+BEIT_SHEAN+"|"+TVERIA+"|"+NATZRAT+"|"+MSADA+"|"+AFULA+"|"+TZFAT+"|"+KATZRIN+"|"+KIRYAT_SMONA
county_list[DAROM] = EILAT+"|"+ASHKELON+"|"+ASHDOD+"|"+BEER_SHEVA+"|"+DIMONA+"|"+KIRYAT_GAT
county_list[HAIFA] = HAIFA+"|"+HADERA+"|"+ACO+"|"+KRAYOT

with open('county_list.txt', 'w') as outfile:
    json.dump(county_list, outfile)

num_unit = {}
ONE = "אח[(ת)(ד)] |שנה "
TWO = "שנתיים |חודשיים "
THREE = "שלוש |שלושה "
FOUR = "ארבע |ארבעה "
FIVE = "חמש |חמ(י)*שה "
SIX = "שש |ש(י)*שה "
SEVEN = "ש(י)*בע(ה)* "
EIGHT = "שמונה "
NINE = "תשע(ה)* "
TEN = "עשר(ה)* "

# num_unit["all_nums"] = ONE+"|"+TWO+"|"+THREE+"|"+FOUR+"|"+FIVE+"|"+SIX+"|"+SEVEN+"|"+EIGHT+"|"+NINE+"|"+TEN
num_unit['1 שנים'] = ONE
num_unit['2 שנים'] = TWO
num_unit['3'] = THREE
num_unit['4'] = FOUR
num_unit['5'] = FIVE
num_unit['6'] = SIX
num_unit['7'] = SEVEN
num_unit['8'] = EIGHT
num_unit['9'] = NINE
num_unit['10'] = TEN
with open('nums_reg.txt', 'w') as outfile:
    json.dump(num_unit, outfile)
