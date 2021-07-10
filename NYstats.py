import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import descartes
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import itertools
from math import ceil
import seaborn as sn

pd.set_option('display.max_columns', None, 'display.max_rows', None)

df = pd.read_csv("AB_NYC_2019.csv")


df.drop(index = df[df['availability_365'] < 10].index, inplace = True)  # remove listings that could be rented for less than 10 days in a year
df.drop(index = df[df['price'] == 0].index, inplace = True)     # remove listings with price equal to 0
df.drop(columns = ['id', 'name', 'host_name', 'calculated_host_listings_count'], inplace=True)      # drop columns that do not carry important info for price prediction

df['pricelog'] = np.log10(df["price"]) + 1      # use log10 to check if it smooths the distribution

groupby = df.groupby(['neighbourhood_group'])

streetmap = gpd.read_file("geo_export_b11fa371-dd25-4311-8aa2-0efcd25bb4dd.shp")

crs = {'init': 'epsg:4326'}
geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]

gdf = gpd.GeoDataFrame(df,
                       crs=crs,
                       geometry=geometry)


def minmaxlat(df):   # Function that takes df with geo data and shows the range of long and lat of the data
    minlong = df["longitude"].min()
    maxlong = df["longitude"].max()
    minlat = df["latitude"].min()
    maxlat = df["latitude"].max()
    print("min and max lat:", minlat, maxlat, "min and max long:", minlong, maxlong)

def mappoints(streetmap, gdf):   # Function that takes shp map and dataframe and plots the map and specific points

    fig, ax = plt.subplots(figsize=(20, 20))
    streetmap.plot(ax=ax, alpha=0.4, color="grey")
    filtlow100 = gdf['price'] < 100
    filt100_200 = np.logical_and(gdf['price'] >= 100, gdf['price'] < 200)
    filt200_350 = np.logical_and(gdf['price'] >= 200, gdf['price'] < 350)
    filtabove = gdf['price'] > 350

    filtlist = [filtlow100, filt100_200, filt200_350, filtabove]
    color = itertools.cycle(('green', 'blue', 'yellow', 'red'))
    label = itertools.cycle(('< 100', '100 - 199', '200 - 349', '> 350'))
    for filt in filtlist:
        gdf.loc[filt].plot(ax = ax, markersize = 10, color = next(color), label = next(label), alpha=0.7)
    plt.legend(prop={'size': 15})
    plt.show()

def stats(dfcol):
    dfcol = dfcol.replace(0, np.NaN)
    print(dfcol.describe()) #show statistics for price without 0s

def hist(df, filter = None, column = "price", interval = 50):   #Plots histograms for nominal price -  without top 3% of values (outliers), for log price - for all of the data. Can apply filters to group the data,
    plt.style.use("seaborn")
    toppercfilt = df["price"] >= df["price"].quantile(0.975)
    dfhist = df.drop(index = df[toppercfilt].index, inplace = False)
    bins = np.arange(int(ceil((dfhist[column].min()))), int(ceil((dfhist[column].max() / interval) * interval)), interval)
    dfhist[column].hist(by = filter, edgecolor="black", bins = bins, grid = False, figsize = (30, 30))
    plt.title('Histogram')
    plt.margins(x = 0, y = 0.05)
    plt.grid(axis = 'x')
    plt.xticks(bins)

    plt.show()  # create histograms with custom ticks on x axis.

def mapdistrict(streetmap, gdf):  #plots NY map and shows airbnb points divided into districts, markers grow with the prices

    fig, ax = plt.subplots(figsize = (20,20))
    streetmap.plot(ax = ax, alpha = 0.5, color = "grey")
    color = itertools.cycle(('red', 'green', 'brown', 'blue', 'orange'))
    markersize = gdf["price"] / 20
    for key, grp in gdf.groupby(['neighbourhood_group']):
        grp.plot(ax = ax, label = key, marker = '>', color = next(color), markersize = markersize, alpha=0.7)
    plt.legend()
    plt.show()

def statsbygroup(df, groupby):
    statdf = groupby["price"].describe()
    statdf = statdf.merge(groupby['price'].agg(['median', 'skew']), left_index=True, right_index= True)
    print(statdf)

def piechart(vc, name, style = 'fivethirtyeight'):
    labels = vc.index.tolist()
    counts = vc.tolist()

    plt.style.use(style)
    fig, ax = plt.subplots(figsize = (10, 10))

    def displayval(pct, allvals):
        absolute = int(round(pct/100.*np.sum(allvals)))
        return "{:.1f}%  ({:d})".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(counts, autopct = lambda  pct: displayval(pct, counts),
                                      textprops = dict(color = "w"), shadow = True, wedgeprops = {'edgecolor': 'black'})
    ax.legend(wedges, labels, loc = "upper right", fontsize = 'small')
    plt.setp(autotexts, size = 7, weight = "bold")
    plt.title(name)
    plt.show()

def avgperdist(groupby):
    means = groupby["price"].mean().sort_values(ascending=False)
    labels = [dist for dist in means.index]
    means = [int(i) for i in groupby["price"].mean().sort_values(ascending = False)]
    medians = [int (i) for i in groupby["price"].median().sort_values(ascending = False)]
    legends = ["mean", "median"]

    plt.style.use("seaborn")
    fig, ax = plt.subplots(figsize = (10, 10))

    bars = ax.bar(labels, means, width = 0.4, alpha = 0.5)
    bars2 = ax.bar(labels, medians, width = 0.4, alpha = 0.5)

    for bar, bar2 in zip(bars, bars2):
        yval = bar.get_height()
        yval2 = bar2.get_height()
        ax.text(bar.get_x(), yval - 8, (str(yval) + "$"), size = 20)
        ax.text(bar2.get_x(), yval2 - 8, (str(yval2) + "$"), size=20)

    ax.legend(legends)
    plt.show()

def heatmap(df):
    dfheat = df.drop(columns=['pricelog'], inplace=False)
    dfcorr = np.triu(dfheat.corr())
    # dfcorr.to_excel(excel_writer="correlation matrix.xlsx")

    plt.figure(figsize=(8, 8))
    corrmap = sn.heatmap(dfheat.corr().round(decimals=2), annot=True, mask = dfcorr, cmap= 'coolwarm')
    plt.tight_layout()
    plt.show()

def scatter(df):
    toppercfilt = df["price"] >= df["price"].quantile(0.975)
    dfscat = df.drop(index=df[toppercfilt].index, inplace=False)
    sn.scatterplot(x='number_of_reviews', y='price', data=dfscat, hue='room_type')
    plt.title("Number of reviews vs price by room type")
    plt.show()

def plotscatter(df, scatterattrib):
    plt.style.use('ggplot')
    scatter_matrix(df[scatterattrib], figsize=(12, 8))
    plt.tight_layout()
    plt.show()

mappoints(streetmap, gdf)   # Show all listings on a map. Colours for price ranges
print(stats(df["price"]))    # Call function that checks .describe stats of a given column of DF (0s changed to NAN)

hist(df)   #Call a function that plots histogram
hist(df, df['neighbourhood_group'])   #shows histograms filtered for NBH group (for all groups)
hist(df, filter = None, column = "pricelog", interval = 0.07)

mapdistrict(streetmap, gdf)     # Call function that shows aparts divided into districts. Markers grow with prices.

statsbygroup(df, groupby)  #displays stats for data divided into neib groups (for full raw data)

vc = df['neighbourhood_group'].value_counts()
name = 'Listings by Neighbourhood Group'
piechart(vc, name)   #piechart of count of listings from given districts

vc2 = df['room_type'].value_counts()
name2 = 'Listing type distribution'
piechart(vc2, name2, 'seaborn')

avgperdist(groupby)   #plot mean and median prices per district

heatmap(df) #plots correlation heatmap for the df

scatter(df)     # scatterplot to show Number of reviews vs price by room type

# plot matrix of scatter plots for selected attributess (cross)
scatterattrib = ['latitude', 'longitude', 'price', 'number_of_reviews', 'availability_365']
plotscatter(df, scatterattrib)
