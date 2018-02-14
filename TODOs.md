Almost certainly necessary:
* Enable basic dropout and get it working
* Enable batch normalization and get it working

Possibly necessary:
* Enable basic L2 regularization and get it working
* Set up adaptive L2 regularization so that the validation loss stays between 1.1
  and 1.5 times the training loss
* Modify the softmax to assume 10% random answer rate
* Decrease rate of labelling
* Enable ensembling of reward prediction again

Only once everything is basically working:
* Fix TODOs in code

Probably unnecessary:
* Generate segments from all workers, rather than only the first
