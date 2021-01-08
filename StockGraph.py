
#Dependencies: alphavantage, plotly
#Arguments: 
# - list of stocks to investigate

# NB - API has a limit of 5 calls for data per minute  -- manage this if you want to compare more? Add a delay?

# Document all functions

# Put the rest into a proper main function

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import plotly as plotly
import numpy as np
import pprint as pprint
import networkx as nx
import sys
import itertools

"""
Compare ranges and distribution of depth predictions for each network against the ground truth range.
Inputs:
File names of each 16-bit depth maps and corresponding ground truth file.
Returns:
Violin plot comparing distributions of predicted depths and individual distribution plots for each network.
"""


def plot_daily_stock_price(stock_name, stock_data):

    # Plot figure
    figure(num=None, figsize=(15, 7), dpi=80, facecolor='w', edgecolor='k')
    stock_data['4. close'].plot() #plot closing price only
    plt.tight_layout()
    plt.ylabel("Closing Price (USD)")
    plt.xlabel("Date")
    plt.title("Daily Closing Price for " + stock_name + " Stock" )
    plt.grid()
    plt.show()


def create_stock_graph(edges):
    
    #create graph of stocks 
    G = nx.Graph()

    for edge in edges:
        print(edges[edge])
        G.add_edge(edge[0], edge[1], weight=edges[edge])
        
    #G.add_edge('WELL', 'AMD', weight=0.32)

    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]
    pos = nx.spring_layout(G)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)
    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color='b', style='dashed')
    # labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

    plt.axis('off')
    plt.show()


def get_edge_weight(last_n_closing_prices, stock_1, stock_2):
    covariance = np.cov(last_n_closing_prices[stock_1],last_n_closing_prices[stock_2])
    print("\n Covariance Matrix: ", stock_1, "and", stock_2, " \n", covariance) #find covariance matrix
    precision_matrix = np.linalg.inv(covariance)
    print("\n Precision Matrix: ", stock_1, "and", stock_2, " \n", precision_matrix) #find precision matrix
    edge_weight = np.abs(precision_matrix[0][1])
    print("\n Edge Weight: ", stock_1, "and", stock_2, " \n", edge_weight) #edge weight = 0 indicates conditional independence
    return edge_weight



print("\n----------------------- StockGraph-------------------------\n")
print("Program to perform portfolio optimization through unsupervised learning and \n graphical networks")
print("------------------------------------------------------------\n")

#print 'Number of arguments:', len(sys.argv), 'arguments.'
#print 'Argument List:', str(sys.argv)

# Arguments
stock_list = ['AMD', 'LRCX', 'WELL', 'VNO'] # stock symbols to compare
n = 5 # number of days used in comparison

# Get time series data, returns a tuple (data, meta_data)
# First entry is a pandas dataframe containing data, second entry is a dict containing meta_data
print("Sourcing stock market data from AlphaVantage API...")
key = 'RY4QOCLLB7ZIVZ8M' # your API key here
ts = TimeSeries(key, output_format='pandas') # choose output format, defaults to JSON (python dict)
ti = TechIndicators(key)

# Create dictionary of form stock_symbol: (data, meta_data)
stock_dict = {}
for stock in stock_list:
    print(stock)
    stock_dict[stock] = ts.get_daily(symbol=stock) 
    

# Visualization to verify correct data retrieval
print("\nProducing graph of daily closing price for ", stock_list[0], " stock...")
stock_data = stock_dict[stock_list[0]][0] # get stock data - first entry in tuple
plot_daily_stock_price(stock_list[0], stock_data)

# Print table of closing prices
print("\nTable showing closing price for ", stock_list[0], " stock over the last 5 reported days (USD)")
print(stock_data['4. close'].head(5))


# Get last n close of day returns for each stock
print("\nGet most recent", n, "close of day returns for each stock")
last_n_closing_prices = {}
for stock in stock_dict:
    stock_data = stock_dict[stock][0]
    last_n_closing_prices[stock] = stock_data['4. close'].head(n).to_numpy() #why did I previously use tail?

# Prints the nicely formatted dictionary
pprint.pprint(last_n_closing_prices)

combos = list(itertools.combinations(stock_list, 2))

#more efficient way?    
edge_dict = {}
for combination in combos:
    print(combination[0], combination[1])
    edge_dict[combination[0], combination[1]] = get_edge_weight(last_n_closing_prices, combination[0], combination[1])

# Prints the nicely formatted dictionary
print("\n Calculated Network Weights between Stocks \n")
pprint.pprint(edge_dict)


network = create_stock_graph(edge_dict)