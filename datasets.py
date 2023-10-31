"""Dataset registrations."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

import common

# 读取自己的数据集
def LoadDmv(filename='dmv-clean.csv', 
            cols=['VIN','City','Zip','County','Reg Valid Date','Reg Expiration Date', 'zvalue'], zvalue=False, dist=None):
    csv_file = './datasets/{}'.format(filename)
    cols = ['Record Type', 'VIN', 'Registration Class', 'City', 'State', 'Zip', 'County', 
            'Body Type', 'Fuel Type', 'Reg Valid Date', 'Reg Expiration Date', 'Color',
            'Scofflaw Indicator', 'Suspension Indicator', 'Revocation Indicator']  
    cols = [cols[1], cols[3], cols[5], cols[6], cols[9], cols[10]]
    if 'dmv-clean' in filename:
       cols = ['Record Type','Registration Class','State','County','Body Type','Fuel Type','Reg Valid Date','Color','Scofflaw Indicator','Suspension Indicator','Revocation Indicator']
    
    if zvalue:
        cols.append('zvalue')
        csv_file = csv_file.split('.csv')[0] + "-zorder.csv"
    # cols = ['VIN','City','Zip','County','Reg Valid Date','Reg Expiration Date', 'zvalue']
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg Valid Date': np.datetime64, 'Reg Expiration Date': np.datetime64}
    
    filename = filename.split('.')[0]
    if dist:
        filename = f'{filename}_{dist}'
    return common.CsvTable(filename, csv_file, cols, type_casts, nrows=1000000)


def LoadLineitem(filename='lineitem.csv', cols=[]):
   '''
   l_orderkey         1500000
   l_partkey           200000
   l_suppkey            10000
   l_linenumber             7
   l_quantity              50
   l_extendedprice     933900
   l_discount              11
   l_tax                    9
   l_returnflag             3
   l_linestatus             2
   l_shipdate            2526
   l_commitdate          2466
   l_receiptdate         2554
   l_shipinstruct           4
   l_shipmode               7
   l_comment          4580667
   '''
   
   csv_file = './datasets/{}'.format(filename)

   # cols = ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity',
   #     'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag',
   #     'l_linestatus', 'l_shipdate', 'l_commitdate', 'l_receiptdate',
   #     'l_shipinstruct', 'l_shipmode', 'l_comment']
    
   # Note: other columns are converted to objects/strings automatically.  We
   # don't need to specify a type-cast for those because the desired order
   # there is the same as the default str-ordering (lexicographical).
   
   # return common.CsvTable(filename, csv_file, cols, sep='|')
   return common.CsvTable("Lineitem", csv_file, cols, sep=',')


def LoadRandomWalk(colsNum, rowNum, dist, zvalue=False):
    filename = f'RandomWalk-{int(rowNum/1000)}K-{colsNum}Col_{dist}.csv'
    cols = list(map(str, range(colsNum)))
   
    if zvalue:
        cols.append('zvalue')
        filename = filename.split('.')[0] + "-zorder.csv"
    csv_file = './datasets/{}'.format(filename)

    if os.path.exists(csv_file):      
        print(f"Fine the dataset: {filename }")      
        return common.CsvTable(filename.split('.')[0], csv_file, cols, sep=',')
    else:
        print(f"Do not find the dataset {filename}, generating...")
        n_series = colsNum
        n_steps = rowNum
        steps = np.random.normal(0, 1, size=(n_series, n_steps))
        position = np.cumsum(steps, axis=1).T
        df = pd.DataFrame(position, columns=cols)
        # Save dataframe to csv file
        df.to_csv(f'./datasets/{filename}', header=True, index=False)
        # np.savetxt(, position, delimiter=',', header=','.join(cols))
        return common.CsvTable(filename.split('.')[0], csv_file, cols, sep=',')

def visualize_distribution(dataset):
    colNum = dataset.shape[1]
    fig, axs = plt.subplots(colNum, figsize=(10, colNum*5))

    for i in range(colNum):
        if isinstance(dataset, pd.DataFrame):
            axs[i].hist(dataset.iloc[:, i], bins=30, density=True)
        elif isinstance(dataset, np.ndarray):
            axs[i].hist(dataset[:, i], bins=30, density=True)
        else:
            raise ValueError(f"Dataset type {type(dataset)} is not supported")
        axs[i].set_title(f'Distribution of column {i+1}')

    plt.tight_layout()
    # plt.show()
    plt.savefig("figure.png")


def GenGAUDataset(colNum, rowNum):
    dataset = np.zeros((rowNum, colNum))
    
    for i in range(colNum):
        # Generate different mean and std for each column
        mean = np.random.uniform(-10, 10)
        std = np.random.uniform(1, 5)
        
        # For some columns, generate a skewed distribution
        if i % 2 == 0:
            skewness = np.random.uniform(-10, 10)
            dataset[:, i] = skewnorm.rvs(a=skewness, loc=mean, scale=std, size=rowNum)
        else:
            dataset[:, i] = np.random.normal(loc=mean, scale=std, size=rowNum)
        
        # make sure all colums are postive
        dataset[:, i] = dataset[:, i] - dataset[:, i].min() + 1
        
    return dataset
    
def LoadGAUDataset(colsNum, rowNum, dist="UNI", zvalue=False):
    filename = f'GAUData-{int(rowNum/1000)}K-{colsNum}Col_{dist}.csv'
    tablename = filename.split('.')[0]
    
    cols = list(map(str, range(colsNum)))
    
    if zvalue:
        cols.append('zvalue')
        filename = filename.split('.')[0] + "-zorder.csv"
    csv_file = './datasets/{}'.format(filename)
    if os.path.exists(csv_file):
        print(f"Fine the dataset: {filename}")      
        return common.CsvTable(tablename, csv_file, cols, sep=',')
    else:
        print(f"Do not find the dataset {filename}, generating...")
        dataset = GenGAUDataset(colsNum, rowNum)
        df = pd.DataFrame(dataset, columns=cols)
        # Save dataframe to csv file
        df.to_csv(f'./datasets/{filename}', header=True, index=False)
        # np.savetxt(, position, delimiter=',', header=','.join(cols))
        return common.CsvTable(tablename, df, cols, sep=',')


def process_ecg_tiny(dist, file_path='ptbdb_normal.csv'):
    file_path = './datasets/{}'.format(file_path)
    colnames = list(map(str, range(188)))
    data = pd.read_csv(file_path, header=None, names=colnames)
    print(data.shape)
    cols = list(map(str, range(data.shape[1])))
    return common.CsvTable(f'ECG_{dist}', data, cols, sep=',')

def LoadUniData(filename, cols=['col_0','col_1'], zvalue=False):
    csv_file = './datasets/{}'.format(filename)
    if zvalue:
        cols.append('zvalue')
        # csv_file = csv_file.split('.csv')[0] + "-zorder.csv"
    csv_file = pd.read_csv(csv_file, nrows=int(1e6))
    return common.CsvTable('UniData', csv_file, cols, sep=',')

if __name__ == '__main__':
   # table = LoadDmv('dmv-clean.csv')
   # print(table.data.info())
#    LoadRandomWalk(100, 10000)
    # LoadGAUDataset(100, int(1e6), dist="UNI")
    dataset = pd.read_csv('./datasets/GAUData-1000K-100Col_UNI.csv', usecols=['0', '1', '2'])
    visualize_distribution(dataset)
      