from automix import Automix, Threshold, SelfConsistency, POMDP
import pandas as pd
import sys

##### Parse Arguments #######
test_file, val_file = sys.argv[1], sys.argv[2]
val_size = int(sys.argv[3])

##### Load Data Files #######
test_df = pd.read_json(test_file, orient="records", lines=True)
val_df = pd.read_json(val_file, orient="records", lines=True)
val_df = val_df.sample(val_size)

##### Init Automix #######
# POMDP:
method = POMDP(num_bins = 8)
# Threshold:
# method = Threshold(num_bins = 8)
# Self-Consistency:
# method = SelfConsistency(num_bins = 8)


mixer = Automix(method)

##### Run Automix #######
mixer.train(val_df)
results = mixer.evaluate(test_df, return_dict = True)

##### Print Results #######
print(results)