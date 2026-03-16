import pandas as pd

df = pd.read_csv('DataSets/SSMI.csv', parse_dates=['Date'], index_col='Date')
null_dates = df[df.isnull().any(axis=1)].index

known_holidays_dates = set()

def easter(year):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19*a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2*e + 2*i - h - k) % 7
    m = (a + 11*h + 22*l) // 451
    month = (h + l - 7*m + 114) // 31
    day = ((h + l - 7*m + 114) % 31) + 1
    return pd.Timestamp(year, month, day)

for year in range(1990, 2022):
    known_holidays_dates.add(pd.Timestamp(year, 1, 1))   # Neujahr
    known_holidays_dates.add(pd.Timestamp(year, 1, 2))   # Berchtoldstag
    known_holidays_dates.add(pd.Timestamp(year, 5, 1))   # Tag der Arbeit
    known_holidays_dates.add(pd.Timestamp(year, 8, 1))   # Bundesfeiertag
    known_holidays_dates.add(pd.Timestamp(year, 12, 24)) # Heiligabend
    known_holidays_dates.add(pd.Timestamp(year, 12, 25)) # Weihnachten
    known_holidays_dates.add(pd.Timestamp(year, 12, 26)) # Stephanstag
    known_holidays_dates.add(pd.Timestamp(year, 12, 31)) # Silvester

    e = easter(year)
    known_holidays_dates.add(e - pd.Timedelta(days=2))   # Karfreitag
    known_holidays_dates.add(e + pd.Timedelta(days=1))   # Ostermontag
    known_holidays_dates.add(e + pd.Timedelta(days=39))  # Auffahrt
    known_holidays_dates.add(e + pd.Timedelta(days=50))  # Pfingstmontag

unexplained = []
for d in null_dates:
    if d not in known_holidays_dates:
        unexplained.append(d)

print(f"Total missing: {len(null_dates)}")
print(f"Explained by holidays: {len(null_dates) - len(unexplained)}")
print(f"Unexplained: {len(unexplained)}")
print()
if unexplained:
    print("Unexplained missing dates:")
    for d in unexplained:
        # Check surrounding data
        loc = df.index.get_loc(d)
        prev_val = df.iloc[loc-1]['Close'] if loc > 0 else None
        next_val = df.iloc[loc+1]['Close'] if loc < len(df)-1 else None
        print(f"  {d.strftime('%Y-%m-%d')} ({d.strftime('%A')})  prev_close={prev_val}  next_close={next_val}")
