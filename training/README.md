```
├─ training
     |
     ├─── data
     |    ├─── subtask.py
     |    ├─── task.py
     |    └─── multi_subtask_dataset.py
     |
     ├─── model
     |    ├─── optimization
     |    |    ├─── grad_aggregator.py
     |    |    └─── grads_wrapper.py
     |    ├─── head.py
     |    ├─── model_factory.py
     |    └─── model.py
     |    
     ├─── trainer
     |    └─── trainer.py
     |
     └─── tokenizer
          └─── mb-mtl-tokenizer
```
Training subdirectory consists of three main components:
- `data` directory contains data structures for task,subtask and dataset. Task (subtask) class define the processing, loss scaling, and other task-specific information. You can add your own subtask by inheriting from the Subtask class, only specifiying `load_data` and `get_scaling_weight` methods. 
- `model` The model architecture is declared in `model.py` and `head.py`. The `Model` class is an encoder-only transformer model. You can specify task-specific head for your tasks in `head.py`. Currently the file contains heads for Classification, Regression, TokenClassification and LanguageModelling tasks.
- `trainer` contains a main `trainer.py` class which orchestrates the whole multi-task training. The trainer is implicitly defined by the task-list and sets of configuration parameters.
