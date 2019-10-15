import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import urllib.request as request
import json
import os
from scipy.sparse import csr_matrix as csr_matrix
from matplotlib import cm
from tempfile import TemporaryFile
import  torch
import torch_geometric
import argparse
from utils import makepath



## Replace function to rename countries for uniformity
def replace_(country_list, replace_dict):
    nnn = country_list.copy()
         #nnn[i] = a.replace('&','and')
    for k, v in replace_dict.items():
        for i,a in enumerate(nnn):
            if a == k:
                nnn[i] = v

    return nnn

def makeplot(edge_list, countries_attributes, country_dict, output_dir):
    G = nx.Graph()  # Define Graph here
    G = G.to_undirected()
    G.add_weighted_edges_from(edge_list)
    pos = nx.spring_layout(G)
    A = nx.adjacency_matrix(G)
    A = A.todense()
    # attr_names = countries_profile.columns[2:]

    attr_dict = get_node_attributes(countries_attributes, country_dict)
    # attr_dict = set_node_attributes(scaled_data, attr_names)
    nx.set_node_attributes(G, attr_dict)

    plt.figure(figsize=(20, 12))

    nx.draw(G, pos, node_size=400, with_labels=True, edge_color='#C0C0C0')
    plt.savefig(output_dir + 'graph_raw.png')
    plt.show()

    return
# Import data between countries into tuples of countries and edges
def make_directed_edges(data, compare_dict):
    data = data.copy()
    edges = []
    for i in range(len(data)):

        c = (compare_dict[str(data.iloc[i,1])], compare_dict[str(data.iloc[i,2])],
               round(data.iloc[i,3],2))
        edges.append(c)
    #edges = sorted(iedges)
    return edges

def check_cyclic_edges(edge_list, remove_edges = False):
    self_edges = []
    new_edge_list = []
    idx = []
    for i in range(len(edge_list)):
        if (edge_list[i][0] == edge_list[i][1]):
            #print(edge_list[i])
            self_edges.append(edge_list[i])
            idx.append(i)
        else:
            new_edge_list.append(edge_list[i])
    if remove_edges:
        return new_edge_list, self_edges
    else:
        return edge_list, self_edges

# Function to make a dictionary of nodes and attributes
def get_node_attributes(attributes, dict_):
    attr_names = attributes.columns[1:]

    attr_dict = {}

    for i in range(len(attributes)):
        attr_dict[dict_[attributes.loc[i][0]]] = {attr_names[j]: k for j, k in enumerate(attributes.loc[i][1:])}
    return attr_dict


def income_level_dict(income_grp, country_dict):
    groups = income_grp.iloc[:,1]
    classes = list(set(groups))
    c_dict = {}
    for c in classes:
        l = income_grp[groups== c].iloc[:,0]
        c_dict[c] = [country_dict[a] for a in l]
    return c_dict


# Function to make a dictionary of nod# Function to make a dictionary of nodes and attributes
def get_node_attributes(attributes, dict_):
    attr_names = attributes.columns[1:]

    attr_dict = {}

    for i in range(len(attributes)):
        attr_dict[dict_[attributes.loc[i][0]]] = {attr_names[j]: k for j, k in enumerate(attributes.loc[i][1:])}
    return attr_dict

## Read data of countries import and exports with partner countries from directory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", default = '../data', type = str, required = False)
    parser.add_argument("--output_dir", default = '../data/processed', type = str, required = False)
    parser.add_argument("-makeplot", type = bool, default = True, help = "Plot graph")

    args = parser.parse_args()
    input_dir = args.raw_data_dir
    comtradeurl = os.path.join(input_dir, "comtrade_data")
    makepath(args.output_dir)
    print("Processing data...")
    replace_dict = np.load(input_dir + '/countries_rename.npy', allow_pickle=True).item()  # Get dict items from npy file

    frames = []
    for name in os.listdir(comtradeurl):
        a = pd.read_csv(os.path.join(comtradeurl, name))
        a = a[['Trade Flow','Reporter','Partner','Trade Value (US$)']]
        frames.append(a)

    trade = pd.concat(frames, ignore_index=True)
    trade = trade.dropna()

    HCI_data = pd.read_csv(os.path.join(input_dir, 'HCIcountry.csv'))
    c_income_group = HCI_data[['Short Name','Income Group']]
    c_income_group = c_income_group.rename(columns = {'Short Name': 'country'})
    inc_levels = set(c_income_group['Income Group'])


    inc_levels_dict = {i:j for j,i in enumerate(inc_levels)}

    countries_attributes = pd.read_csv(os.path.join(input_dir, "country_profile_variables2017.csv"))
    countries_attributes = countries_attributes.replace(['~0','~0.0','-~0.0','...'],0)
    countries_attributes = countries_attributes.apply(lambda x: pd.to_numeric(x, errors = 'ignore'))

    # Create feature dictionary for easy selection
    feature_indices_dict = {i:j for i,j in enumerate(list(countries_attributes.columns))}

    countries_attributes.iloc[:,2:] = countries_attributes.iloc[:,2:].select_dtypes(exclude = 'object')
    countries_attributes = countries_attributes.dropna(axis = 'columns')
    countries_attributes = countries_attributes.drop(['Region'], axis = 1)
    countries_attributes.head()

    cols = countries_attributes.columns[1:]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(countries_attributes.iloc[:,1:])
    scaled_data = pd.DataFrame(scaled_data, columns = cols)
    countries_attributes.iloc[:,1:] = scaled_data
    countries_attributes.head()

    #----------------------------------------------------------------------------------
    countries_distances = pd.read_csv(os.path.join(input_dir,"countries_distances.csv"))
    countries_distances = countries_distances.rename(columns = {'pays1':'country1', 'pays2':'country2'})
    countries_distances = countries_distances[['country1','country2','dist']]
    countries_names = list(countries_distances['country1'])
    countries_distances.head()

    #-----------------------------------------------------------------------------------
    dat1 = list(countries_attributes['country'])
    dat2 = list(c_income_group['country'])
    dat3 = list(set(countries_distances['country1']))
    dat3_1 = list(countries_distances['country1'])
    dat3_2 = list(countries_distances['country2'])

    dat1 = replace_(dat1, replace_dict)
    dat2 = replace_(dat2, replace_dict)
    dat3 = replace_(dat3, replace_dict)
    dat3_1  = replace_(dat3_1, replace_dict)

    countries_attributes['country'] = dat1
    c_income_group['country'] = dat2
    countries_distances['country1'] = dat3_1
    countries_distances['country2'] = dat3_2
    countries_attributes = countries_attributes.drop_duplicates(subset = 'country', inplace = False)


    #----------------------------------------------------------------------------------------
    # [print(i) for i in c_income_group['country']]
    common_countries = []  # Countries found in all three lists of countries
    c1_nc23 = []  # countries found in c1 but not in c2 and c3
    ncm123 = []
    c2_nc13 = []  # countries found in c2 but not in c1 and c3
    c3_nc12 = []  # countries found in c3 but not in c1 and c2
    for c in dat1:
        if c in dat2 and c in dat3:
            common_countries.append(c)
        else:
            ncm123.append(c)

    for c in dat2:
        if c in dat1 and c in dat3:
            pass
        else:
            c2_nc13.append(c)

    for c in dat3:
        if c in dat1 and c in dat2:
            pass
        else:
            c3_nc12.append(c)

    print(len(common_countries))

    #-----------------------------------------------------------------------------------------

    ## Make a dictionary of countries and their given codes as keys for easy reference

    country_dict = {j:i for i, j in enumerate(sorted(set(common_countries)))}
    #country_dict

    #----------------------------------------------------------------------------------------
    # Select countries with names or data appearing in each of the datasets

    countries_attributes = countries_attributes[countries_attributes['country'].isin(common_countries)].reset_index(drop =True)
    c_income_group = c_income_group[c_income_group['country'].isin(common_countries)]
    countries_dists = countries_distances[countries_distances['country1'].isin(common_countries)]
    countries_dists = countries_dists[countries_dists['country2'].isin(common_countries)]
    #--------------------------------------------------------------------------

    cdist = countries_dists.copy()
    edge_list = []
    for i in range(len(cdist)):
        c = (country_dict[str(cdist.iloc[i, 0])], country_dict[str(cdist.iloc[i, 1])],
             round(cdist.iloc[i, 2], 2))

        edge_list.append(c)
    edge_list = sorted(edge_list)
    # edge_list
    #------------------------------------------------------------------------------

    edges_dists = pd.DataFrame(edge_list)
    #-----------------------------------------------------------------------------------------------
    trade_reporters = list(set(trade['Reporter']))
    trade_partners = list(set(trade['Partner']))
    flow = list(set(trade['Trade Flow']))

    imports_data = trade[trade['Trade Flow'] == 'Import'].reset_index(drop=True)
    reimports_data = trade[trade['Trade Flow'] == 'Re-Import'].reset_index(drop=True)
    exports_data = trade[trade['Trade Flow'] == 'Export'].reset_index(drop=True)
    reexports_data = trade[trade['Trade Flow'] == 'Re-Export'].reset_index(drop=True)

    imp_partners = imports_data['Partner']
    imp_reporters = imports_data['Reporter']
    imports_data['Partner'] = replace_(imp_partners, replace_dict)
    imports_data['Reporter'] = replace_(imp_reporters, replace_dict)

    exp_partners = exports_data['Partner']
    exp_reporters = exports_data['Reporter']
    exports_data['Partner'] = replace_(exp_partners, replace_dict)
    exports_data['Reporter'] = replace_(exp_reporters, replace_dict)

    #-----------------------------------------------------------------------------------------------

    i = 0
    reps = replace_(trade['Reporter'], replace_dict)
    pars = replace_(trade['Partner'], replace_dict)
    als = list(reps) + list(pars)
    cin = []
    cnot = []
    for c in als:
        if c in list(common_countries):
            cin.append(c)
            i += 1
        else:
            cnot.append(c)


    cin1 = []
    cnot1 = []
    for c in list(common_countries):
        if c in als:
            cin1.append(c)
        else:
            cnot1.append(c)



    i = 0
    cin = []
    cnot =[]
    partns = replace_(trade['Partner'], replace_dict)
    for c in list(common_countries):
        if c in list(partns):
            cin.append(c)
            i +=1
        else:
            cnot.append(c)

    imports_data = imports_data[imports_data['Reporter'].isin(common_countries)].reset_index(drop = True)
    imports_data = imports_data[imports_data['Partner'].isin(common_countries)].reset_index(drop = True)

    exports_data = exports_data[exports_data['Reporter'].isin(common_countries)].reset_index(drop = True)
    exports_data = exports_data[exports_data['Partner'].isin(common_countries)].reset_index(drop = True)

    reimports_data = reimports_data[reimports_data['Reporter'].isin(common_countries)].reset_index(drop = True)
    reimports_data = reimports_data[reimports_data['Partner'].isin(common_countries)].reset_index(drop = True)

    reexports_data = reexports_data[reexports_data['Reporter'].isin(common_countries)].reset_index(drop = True)
    reexports_data = reexports_data[reexports_data['Partner'].isin(common_countries)].reset_index(drop = True)


    # Make edges and remove recurring ones
    iedges = make_directed_edges(imports_data, country_dict)
    iedges, iself_edges = check_cyclic_edges(iedges, remove_edges=True)

    eedges = make_directed_edges(exports_data, country_dict)
    eedges, eself_edges = check_cyclic_edges(eedges, remove_edges=True)


    # cdict = income_level_dict(c_income_group, country_dict)
    il_dict = income_level_dict(c_income_group, country_dict)

    inc = c_income_group.sort_values(by = ['country'])
    labels = list(map(inc_levels_dict.get,inc['Income Group']))
    data = countries_attributes.sort_values(by = ['country'])

    attr_names = data.iloc[:,1:].columns
    attr_data = data.iloc[:,1:].values
    attr_shape = attr_data.shape
    class_names = list(inc_levels)


    isrc, itar, iwei = zip(*iedges)  # Unzip import edges
    esrc, etar, ewei = zip(*eedges)  # Unzip export edges
    dsrc, dtar, dwei = zip(*edge_list) #Unzip distance edges

    imat = csr_matrix((iwei,(isrc, itar))).todense()
    emat = csr_matrix((ewei,(esrc, etar))).todense()
    sparse_adj_dists = csr_matrix((dwei,(dsrc, dtar))) # Make sparse adjacency matrix for distances

    tmat = imat - emat # Trade balance incidence matrix
    sparse_adj_trade = csr_matrix(tmat) # Make sparse adjacency matrix for trade balance

    trade_savez_files = TemporaryFile()
    output_file = os.path.join(args.output_dir, "trade_savez_files")
    saver = np.savez(output_file, attr_data = attr_data, attr_shape = attr_shape,
             sparse_adj_trade = sparse_adj_trade, sparse_adj_dists = sparse_adj_dists, labels = labels, class_names = class_names)
    
    print("Done!... Preprocessed data saved in ", args.output_dir)
    if args.makeplot:
        graph_outpath = "../images"
        makepath(graph_outpath)
        makeplot(edge_list, countries_attributes, country_dict, graph_outpath)

if __name__ == "__main__":
    main()
    
    