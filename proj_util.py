from feature_analysis import *
#use 'execfile(file.py)' to evaluate file as a series of python statements

def plot_values(key):
    point_indicies = range(len(data_dict))
    plt.scatter(point_indicies,[d[1][key] for d in data_dict])
    plt.title(key)
    plt.xticks(point_indicies,[x[0] for x in data_dict],rotation='vertical')
    plt.tight_layout()
    plt.show()

def print_values(key):
    for k in poi_id.data_dict[key]:
        print k," : ",poi_id.data_dict[key][k]


del poi_id.data_dict['TOTAL']
del poi_id.data_dict['THE TRAVEL AGENCY IN THE PARK']
data_dict = poi_id.data_dict.items()
data_dict.sort()
data_keys = data_dict[0][1].keys()

if __name__ == "__main__":
    for k in data_keys:
        if k != 'email_address':
            plot_values(k)
