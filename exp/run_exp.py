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

parser = ArgumentParser("run a detection comparison")
parser.add_argument("data_dir", type=str, help="the data directory which contains the\
    relevant annotations, constrains, and original input", default="data")
parser.add_argument("data_name", type=str, help="the name of the dataset to be run on")
parser.add_argument("input_file", type=str, help="the relative path to the input file")
parser.add_argument("dc_file", type=str, help="the relative path to the constraints file")
parser.add_argument("clean_file", type=str, help="the relative path to the annotated file")

args = parser.parse_args()
# 1. Setup a HoloClean session.
hc = holoclean.HoloClean(pruning_topk=0.1, epochs=12, weight_decay=0.01, threads=20, batch_size=1, verbose=True, timeout=3*60000).session

# 2. Load training data and denial constraints.
hc.load_data(args.data_name, args.data_dir, args.input_file)
hc.load_dcs(args.data_dir, args.dc_file)
hc.ds.set_constraints(hc.get_dcs())

# 3. Detect erroneous cells using these two detectors.
detectors = [NullDetector(), ViolationDetector()]
hc.detect_errors(detectors)

# 4. Repair errors utilizing the defined features.
hc.setup_domain()
featurizers = [InitAttFeaturizer(), InitSimFeaturizer(), FreqFeaturizer(), OccurFeaturizer(), LangModelFeat(), ConstraintFeat()]
hc.repair_errors(featurizers)

# 5. Evaluate the correctness of the results.
def get_tid(row):
    return row['tid'] - 1

def get_attr(row):
    return row['attribute'].lower()

def get_value(row):
    return row['correct_val'].lower()

# inferred df
inferred_df = hc.ds.repaired_data.df

print(inferred_df.head(4))

# error dataframe
error_df = hc.detect_engine.errors_df
print(error_df.head(4))
#original data
print(hc.ds.raw_data.df.head(4))

clean_df = pd.read_csv(join(args.data_dir, args.clean_file))
print(clean_df.head(4))

# hc.evaluate(args.data_dir, args.clean_file, get_tid, get_attr, get_value)