# Random Seeds Problem within NER
## Requirements
NOTE: This has only been tested on Ubuntu 16.04

1. Python 3.6.1 or above
2. install the pip requirements `pip install -r requirements.txt`

Need to download the CoNLL 2003 NER dataset and store it within `./original_dataset` where the train, dev, and test splits are at the following respective paths `./original_dataset/train.txt`, `./original_dataset/dev.txt`, `./original_dataset/test.txt`. NOTE: Ensure that all of the splits have been pre-processed so that they are in BIO format and not IOB format.

## Results of the NER models on different train, validation, and test splits
First we must create different train, validation, and test splits. To do this we create a different directory for each new random train, validation, and test split. In the paper we have 150 different random splits which is created using the following command:

`python creating_data_sets.py ./original_dataset ./copy 150`

Where `./copy` is the new directory that will store 150 folders named 0 to 149 where in each numbered folder are three files `train.txt`, `dev.txt`, and `test.txt`.

After creating 150 new random datasets we run the 2 models 5 times using a different seed each time on each of the 150 new random datasets. Where the results are stored in respective dataset folder under the file `results.json`. To do this run the following command:

`./datasets_and_random_seeds.sh 0 150 ./copy PATH_TO_GLOVE_FILE PATH_TO_TEMP_DIR PATH_TO_PYTHON_RUNNABLE`

`PATH_TO_GLOVE_FILE` -- This is the path to the 100 dimension Glove file which can be downloaded from [here](http://nlp.stanford.edu/data/glove.6B.zip)
`PATH_TO_TEMP_DIR` -- Can be any folder but most likely a folder from `/tmp` directory
`PATH_TO_PYTHON_RUNNABLE` -- an example would be `/home/andrew/Envs/NER/bin/python`

The results from our experiments can be found in `./results/ner_dataset_and_random_seeds.json` of which this single results file can be genererated from all of the different dataset results files be using the following script:

`python join_dataset_results.py ./copy ./results/ner_dataset_and_random_seeds.json`

## To create the visulisation from the paper

The visulisation is saved in the `./image` directory, of which the violin plot showing the distribution of changing the random seed and the data split for the CNN and LSTM model can be produced using the following command:

`python violin_plot_of_ner_scores.py ./results/ner_dataset_and_random_seeds.json ./image/ner_violin_plot.png test 0 --remove_bottom_5`

To get the exact plot in the paper which shows 3 additional violin plots for fixed data split but different random seed run the following command:

`python violin_plot_of_ner_scores.py ./results/ner_dataset_and_random_seeds.json ./image/ner_violin_plot_with_fixed.png test 3 --remove_bottom_5`

The violin plot shows all the results including the bottom five results for each encoder. The bottom 5 were removed as some of the methods did not converag. To create the non removed plot run the following command:

`python violin_plot_of_ner_scores.py ./results/ner_dataset_and_random_seeds.json ./image/ner_violin_plot_bottom_5.png test 0`

### Note to self
A problem with the code is the following. Running the results for only seeds 250 times is bad as it runs out of space within the removal directory therefore it is better to run the code in multiples of 5 and removing the temporary directory each time and re-creating it.

Another problem which is similar to the above is that running the code that changes the train, val, and test splits could be problematic when running it continously might be better like above to split it up into chunks to avoid some problems like max files open and max recuccursion depth.

