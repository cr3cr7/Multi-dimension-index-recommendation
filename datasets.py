"""Dataset registrations."""
import os

import numpy as np

import common

# 读取自己的数据集
def LoadDmv(filename='Vehicle__Snowmobile__and_Boat_Registrations.csv'):
    csv_file = './datasets/{}'.format(filename)
    cols = ['Record Type', 'VIN', 'Registration Class', 'City', 'State', 'Zip', 'County', 
            'Body Type', 'Fuel Type', 'Reg Valid Date', 'Reg Expiration Date', 'Color',
            'Scofflaw Indicator', 'Suspension Indicator', 'Revocation Indicator']
    """ cols = ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity',
       'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag',
       'l_linestatus', 'l_shipdate', 'l_commitdate', 'l_receiptdate',
       'l_shipinstruct', 'l_shipmode', 'l_comment'] """
    
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg Valid Date': np.datetime64, 'Reg Expiration Date': np.datetime64}
    #return common.CsvTable('lineitem', csv_file, cols)
    return common.CsvTable('DMV', csv_file, cols, type_casts)

def LoadLineitem(filename='lineitem.csv'):
    csv_file = './datasets/{}'.format(filename)

    cols = ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity',
       'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag',
       'l_linestatus', 'l_shipdate', 'l_commitdate', 'l_receiptdate',
       'l_shipinstruct', 'l_shipmode', 'l_comment']
    
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    return common.CsvTable(filename, csv_file, cols, sep='|')


