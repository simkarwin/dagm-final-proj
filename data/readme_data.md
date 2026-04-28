## Dataset Configuration

To train the model and print results, you must update the dataset paths in the configuration section of the code (specifically lines 26 to 32).

These paths point to the JSON annotation files and corresponding image directories for training, validation, and testing.

### Update the following paths in your script:

```python
TRAIN_JSON = "/home/ehk224/Downloads/floodnet/data/train_annotations.json"
VAL_JSON   = "/home/ehk224/Downloads/floodnet/data/valid_annotations.json"
TEST_JSON  = "/home/ehk224/Downloads/floodnet/data/test_annotations.json"

TRAIN_IMG_DIR = "/home/ehk224/Downloads/floodnet/Images/train_images"
VAL_IMG_DIR   = "/home/ehk224/Downloads/floodnet/Images/valid_images"
TEST_IMG_DIR  = "/home/ehk224/Downloads/floodnet/Images/test_images"