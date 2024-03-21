import re
import numpy as np
import pandas as pd
import os

# 示例SQL查询
sql_query = """
select
    sum(l_extendedprice * l_discount) as revenue
from
    lineitem
where
    l_shipdate >= date '1993-01-01'
    and l_shipdate < date '1993-01-01' + interval '1' year
    and l_discount between 0.07 - 0.01 and 0.07 + 0.01
    and l_quantity < 25
LIMIT 1;
"""

def LoadLineitem(filename='lineitem.csv', cols=[]):
    lineitem_columns = [
    "l_orderkey",
    "l_partkey",
    "l_suppkey",
    "l_linenumber",
    "l_quantity",
    "l_extendedprice",
    "l_discount",
    "l_tax",
    "l_returnflag",
    "l_linestatus",
    "l_shipdate",
    "l_commitdate",
    "l_receiptdate",
    "l_shipinstruct",
    "l_shipmode",
    "l_comment"
    ]
    type_casts = {'l_receiptdate': np.datetime64, 'l_commitdate': np.datetime64, 'l_shipdate': np.datetime64}

    csv_file = './datasets/{}'.format(filename)
    table = pd.read_table(csv_file, 
                          sep='|', 
                          header=None, 
                          names=lineitem_columns, 
                          nrows=1000000)[cols]
    for col, typ in type_casts.items():
        if col not in table.columns:
            continue
        if typ != np.datetime64:
            table[col] = table[col].astype(typ, copy=False)
        else:
            table[col] = pd.to_datetime(table[col], 
                                        infer_datetime_format=True, 
                                        cache=True)
   # Note: other columns are converted to objects/strings automatically.  We
   # don't need to specify a type-cast for those because the desired order
   # there is the same as the default str-ordering (lexicographical).
   
   # return common.CsvTable(filename, csv_file, cols, sep='|')
    return table

def parse_date_interval(date_str):
    # 解析日期和间隔
    match = re.match(r"date '(\d{4}-\d{2}-\d{2})' \+ interval '(\d+)' (year|month|day)", date_str)
    if not match:
        raise ValueError("Invalid date interval format: ", date_str)
    
    start_date, interval, unit = match.groups()
    interval = int(interval)

    # 根据单位调整numpy的日期单位
    if unit == "year":
        np_unit = 'Y'
    elif unit == "month":
        np_unit = 'M'
    elif unit == "day":
        np_unit = 'D'
    else:
        raise ValueError(f"Unsupported interval unit: {unit}")

    # 创建numpy日期对象
    # start_date_np = np.datetime64(start_date)
    # end_date_np = start_date_np + np.timedelta64(interval, np_unit)
    start_date = pd.to_datetime(start_date)
    if np_unit == 'Y':
        end_date = start_date + pd.DateOffset(years=interval)
    elif np_unit == 'M':
        end_date = start_date + pd.DateOffset(months=interval)
    elif np_unit == 'D':
        end_date = start_date + pd.DateOffset(days=interval)
    else:
        raise ValueError("Unsupported unit for np.timedelta64")

    # 将结果转换为np.datetime64
    end_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
    return end_date

table = LoadLineitem('lineitem.tbl', cols=['l_shipdate', 'l_discount', 'l_quantity'])
min_max_stats = table.agg(['min', 'max'])
print(min_max_stats)
scan_cond = []
all_files = os.listdir('./datasets/Q6/')
for cur_file in all_files:
    with open(
        os.path.join('./datasets/Q6', cur_file), 'r') as outfile:
        parsed_conditions = []
        after_where = False
        for line in outfile:
            # after where 
            if 'where' in line:
                after_where = True
                continue
            if after_where:
                if 'l_shipdate' in line or\
                    'l_discount' in line or\
                    'l_quantity' in line:
                    # processs into [[cols], [[min, max], ...]]
                    
                    condition = line.strip().strip('and ')
        
                    # 解析 BETWEEN AND 条件
                    if 'between' in condition:
                        col, rest = condition.split(' between ')
                        min_val, max_val = re.split(r'\s+and\s+', rest)
                        print((col, 'BETWEEN', [min_val, max_val]))
                        parsed_conditions.append((col, 'BETWEEN', [min_val, max_val]))
                    # 解析其他条件
                    else:
                        match = re.match(r"(\S+) (>=|<=|<|>|=) (.+)", condition)
                        if match:
                            col, op, val = match.groups()
                            parsed_conditions.append((col, op, [val]))
                            print((col, op, [val]))
                        else:
                            raise NotImplementedError('Error: {}'.format(condition))

        col_ranges = {}
        for col, op, values in parsed_conditions:
            if col not in col_ranges:
                col_ranges[col] = [None, None]  # [min, max]
            
            if op == '>=':
                if 'date' in values[0]:
                    res = pd.Timestamp(values[0].strip('date ').strip("'")).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    res = eval(values[0])
                col_ranges[col][0] = res
            elif op == '<':
                if 'date' in values[0]:
                    res = parse_date_interval(values[0])
                else:
                    res = eval(values[0])
                col_ranges[col][1] = res
            elif op == 'BETWEEN':
                values = [eval(val) for val in values]
                col_ranges[col] = values  # BETWEEN包含了[min, max]
            else:
                raise NotImplementedError(f"Unsupported operator: {op}")
        for col, (min_val, max_val) in col_ranges.items():
            if min_val is None:
                min_val = min_max_stats.loc['min', col]
            if max_val is None:
                max_val = min_max_stats.loc['max', col]
            col_ranges[col] = [min_val, max_val]
        print(col_ranges)
        cols = list(col_ranges.keys())
        range_values = list(col_ranges.values())

        scan_cond.append([cols, range_values])
print(scan_cond)
import pickle
with open('./datasets/scan_condation_lineitem.pkl', 'wb') as f:
    pickle.dump(scan_cond, f)