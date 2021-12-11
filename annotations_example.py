from annotation_functions import *

# This script demonstrates how functions in annotations_functions can be used.
# most of these functions are standalone and do not require that you've run them in order. 
# all functions expect working directory to be endometriosis, with labels stored in:
#    - endometriosis/labeling/prodigy
# and annotation output stored in:
#    - endometriosis/labeling/prodigy/output

# determine the type of label
label_type = "relations"

# get file paths --> useful for double checking, but most functions already embed this function
labels_file_path, annotations_path, output_csv_path = get_file_paths_for_label(label_type)

# only need to run this after a new prodigy output --> reformats from json to csv, with labels properly formatted
format_raw_annotations(label_type)

# simple printout of label stats
get_label_stats(get_label_stats)

# load annotations (what is created by format_raw_annotations())
annotations_df = load_annotations(label_type)

# get the classification accuracy as more data is added --> will plot over increased nums of samples
classification_accuracy_by_data_amount(label_type)

# get the overall cross validated accuracy for each label
accuracy_scores = get_accuracy_metrics(label_type)
