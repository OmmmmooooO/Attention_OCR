Attenion-OCR README.
***
  This markdown is for "Attention-based Extraction of Structured Information from Street View Imagery" or "Attention-OCR" project.


### If you want to use your own images stored in fsns format, just follow the following steps:

### Step 1.
 Use `image_to_fsns_format.ipynb` to transfer and/or augment your image to [H, W, Channel]=[150, 600, 3] included 4 different/same views per image.
### Step 2.
Then use `tfrecord_gen.py` to transfer the images made by step 1 to fsns dataset.

### Step 3.
Move the binary files to train/test/validation folders and run `train.py` or `evil.py`.


For further information, check this out:
[https://github.com/OmmmmooooO/models/tree/master/attention_ocr]
