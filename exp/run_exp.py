import holoclean
import pandas as pd
from os.path import join
from argparse import ArgumentParser
from detect import NullDetector, ViolationDetector
from repair.featurize import InitFeaturizer
from repair.featurize import InitAttFeaturizer
from repair.featurize import InitSimFeaturizer
from repair.featurize import FreqFeaturizer
from repair.featurize import OccurFeaturizer
from repair.featurize import ConstraintFeat
from repair.featurize import LangModelFeat

# parser = ArgumentParser("run a detection comparison")
# parser.add_argument("data_dir", type=str, help="the data directory which contains the\
#     relevant annotations, constrains, and original input", default="data")
# parser.add_argument("data_name", type=str, help="the name of the dataset to be run on")
# parser.add_argument("input_file", type=str, help="the relative path to the input file")
# parser.add_argument("dc_file", type=str, help="the relative path to the constraints file")
# parser.add_argument("clean_file", type=str, help="the relative path to the annotated file")

# args = parser.parse_args()
# 1. Setup a HoloClean session.
hc = holoclean.HoloClean(db_host='localhost', pruning_topk=0.1, epochs=12, weight_decay=0.01, threads=20, batch_size=1, verbose=True,
                         timeout=3 * 60000).session

# 2. Load training data and denial constraints.

hc.load_data('hospital', 'data',  'food5k/food5k.csv')
hc.load_dcs('data', 'food5k/food5k_constraints.txt')
hc.ds.set_constraints(hc.get_dcs())

# hc.load_data(args.data_name, args.data_dir, args.input_file)
# hc.load_dcs(args.data_dir, args.dc_file)
# hc.ds.set_constraints(hc.get_dcs())

# 3. Detect erroneous cells using these two detectors.
detectors = [NullDetector(), ViolationDetector()]
hc.detect_errors(detectors)

# 4. Repair errors utilizing the defined features.
hc.setup_domain()
featurizers = [InitAttFeaturizer(), InitSimFeaturizer(), FreqFeaturizer(), OccurFeaturizer(), LangModelFeat(),
               ConstraintFeat()]
hc.repair_errors(featurizers)


# 5. Evaluate the correctness of the results.
def get_tid(row):
    # if row['tid'] == 0:
    #     return row['tid']
    return row['tid']


def get_attr(row):
    return row['attribute'].lower()


def get_value(row):
    return row['correct_val'].lower()


def flattening_df(input_df, target_col_nam="_tid_,_attribute_,_value_", ind=0):
    columns_name = list(input_df.columns.values)
    # print(columns_name)
    temp = []
    for row in input_df.iterrows():
        index, data = row
        temp.append(data.tolist())

    data = []
    cols = target_col_nam.split(',')
    for counter, val in enumerate(temp):
        for (idx, el) in enumerate(val):
            if columns_name[idx] != '_tid_':
                data.append([counter + ind, columns_name[idx], el])
    result = pd.DataFrame(data, columns=cols)
    result[cols[0]] = pd.to_numeric(result[cols[0]])
    return result


# inferred df
inferred_df = hc.ds.repaired_data.df

# print("Inferred dataframe")
# print(inferred_df.head(4))

HC = flattening_df(inferred_df).sort_values(['_tid_', '_attribute_'], ascending=[True, True]).reset_index(drop=True)

print("HC dataframe")
print(HC.head(4))

# error dataframe
# error_df = hc.detect_engine.errors_df
#
# # flattening_df(error_df)
# print("Error dataframe")
# print(error_df.head(4))
# original dat:qa


init = flattening_df(hc.ds.raw_data.df).sort_values(['_tid_', '_attribute_'], ascending=[True, True]).reset_index(drop=True)

print("init dataframe")
print(init.head(4))

# GT = pd.read_csv('data/hospital_clean.csv').sort_values(['_tid_', '_attribute_'], ascending=[True, True]).reset_index(drop=True)
# # GT = pd.read_csv(join(args.data_dir, args.clean_file))





# hc.evaluate(args.data_dir, args.clean_file, get_tid, get_attr, get_value)
hc.evaluate('data', 'food5k/food5k_clean.csv', get_tid, get_attr, get_value)
GT = hc.eval_engine.clean_data.df.sort_values(['_tid_', '_attribute_'], ascending=[True, True]).reset_index(drop=True)

print("GT dataframe")
print(GT.head(4))

# HC_pos = HC.select()
HC_pos = HC.loc[(HC['_tid_'] == init['_tid_']) & (HC['_attribute_'] == init['_attribute_']) & (HC['_value_'] != init['_value_'])]
HC_neg = HC.loc[(HC['_tid_'] == init['_tid_']) & (HC['_attribute_'] == init['_attribute_']) & (HC['_value_'] == init['_value_'])]
GT_pos = GT.loc[(GT['_tid_'] == init['_tid_']) & (GT['_attribute_'] == init['_attribute_']) & (GT['_value_'] != init['_value_'])]
GT_neg = GT.loc[(GT['_tid_'] == init['_tid_']) & (GT['_attribute_'] == init['_attribute_']) & (GT['_value_'] == init['_value_'])]


TP = pd.merge(HC_pos, GT_pos, how='inner', on=['_tid_', '_attribute_'])
FN = pd.merge(GT_pos, HC_neg, how='inner', on=['_tid_', '_attribute_'])
TN = pd.merge(HC_neg, GT_neg, how='inner', on=['_tid_', '_attribute_'])
FP = pd.merge(HC_pos, GT_neg, how='inner', on=['_tid_', '_attribute_'])

print(len(TP), len(FN), len(TN), len(FP))


# Flattening

