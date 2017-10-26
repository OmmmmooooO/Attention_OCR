from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf
import glob
import os
import re
import sys
import cv2

# Iuput image should be in (150*600*3), 4 views/png.
# charset_file is original fsns charset_size=134.txt file.

image_input_filename='/media/ryk/ML_disk/Project/FSNS_license/attention_ocr/python/datasets/data/fsns/resized/*.png'
tfrecords_output_filename = 'validation_6'
charset_file='/media/ryk/ML_disk/Project/FSNS_license/attention_ocr/python/datasets/data/fsns/charset_size=134.txt'

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def encode_utf8_string(text, length, charset, null_char_id=133):

    lookup={'1':13, '2':52, '3':62, '4':72, '5':58, '6':56, '7':15,
            '8':14, '9':64, '0':16, 'A':37, 'B':67, 'C':30, 'D':41,
            'E':39, 'F':74, 'G':46, 'H':43, 'I':77, 'J':78, 'K':68,
            'L':73, 'M':44, 'N':55, 'O':60, 'P':47, 'Q':94, 'R':49,
            'S':32, 'T':59, 'U':61, 'V':25, 'W':70, 'X':84, 'Y':86, 
            'Z':66, '-':17}
    
    keys = lookup.keys()
    values = lookup.values()

    char_ids_padded = []
    char_ids_unpadded = []

    pad_null_for6chars = [null_char_id for x in range(31)]
    pad_null_for7chars = [null_char_id for x in range(30)]
    pad_null_for8chars = [null_char_id for x in range(29)]

    """ for debugging and comparison
    charset_values = []
    charset_keys = []
    charset_values.extend(charset.values())
    charset_keys.extend(charset.keys())
    """

    for i in range(length):
        char_ids_padded.append(values[keys.index(list(text)[i])])

    char_ids_unpadded = list(char_ids_padded)
    if length == 6:
        char_ids_padded.extend(pad_null_for6chars)
    if length == 7:
        char_ids_padded.extend(pad_null_for7chars) 
    if length == 8:
        char_ids_padded.extend(pad_null_for8chars)        

    return char_ids_padded, char_ids_unpadded

# For debugging/checking only.
def read_charset(filename, null_character=u'\u2591'):
  """Reads a charset definition from a tab separated text file.

  charset file has to have format compatible with the FSNS dataset.

  Args:
    filename: a path to the charset file.
    null_character: a unicode character used to replace '<null>' character. the
      default value is a light shade block.

  Returns:
    a dictionary with keys equal to character codes and values - unicode
    characters.
  """
  pattern = re.compile(r'(\d+)\t(.+)')
  charset = {}
  with tf.gfile.GFile(filename) as f:
    for i, line in enumerate(f):
      m = pattern.match(line)
      if m is None:
        logging.warning('incorrect charset file. line #%d: %s', i, line)
        continue
      code = int(m.group(1))
      char = m.group(2).decode('utf-8')
      if char == '<nul>':
        char = null_character
      charset[code] = char
  return charset

writer = tf.python_io.TFRecordWriter(tfrecords_output_filename)
num_of_views = 4

# Read charset_size=134.txt as charset just for debugging.
charset = read_charset(charset_file)

for file in glob.iglob(image_input_filename):
    # File name is text or ground truth. Length is either 7 or 8.
    # Image size: [H x W x 3] == [150 X 600 X 3] (4 views/image)

    head, tail = os.path.split(file)
    filename=tail.split('.', 1)
    text=filename[0]
    length=len(text)
    img = np.array(Image.open(file))
    char_ids_padded, char_ids_unpadded=encode_utf8_string(text, length, charset, null_char_id=133)
    #print char_ids_padded, char_ids_unpadded
    
    _,pngVector = cv2.imencode('.png',img)
    imgStr = pngVector.tostring()
    
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature([imgStr]),
        'image/format': _bytes_feature(['PNG']),
        'image/width': _int64_feature([img.shape[1]]),
        'image/orig_width': _int64_feature([img.shape[1]/num_of_views]),
        'image/class': _int64_feature(char_ids_padded),
        'image/unpadded_class': _int64_feature(char_ids_unpadded),
        'image/text': _bytes_feature([text])}))

    writer.write(example.SerializeToString())
   
writer.close()
    
