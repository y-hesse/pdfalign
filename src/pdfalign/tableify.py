'''

Author: Yannik Hesse
Github: https://github.com/y-hesse/pdfalign
Date: 14.08.2024 

'''

import numpy as np
from typing import Optional, Tuple
import pandas as pd
import pymupdf
from PIL import Image

TESSERACT_AVAIL = True

try:
    import pytesseract
    from pytesseract import Output
except ImportError:
    TESSERACT_AVAIL = False

class EmptyPageException(Exception):
  """
  Custom exception raised when a page is empty.

  Attributes:
    page_number -- the page number that was empty
    message -- explanation of the error
  """

  def __init__(self, page_number, message="Page is empty."):
    self.page_number = page_number
    self.message = message
    super().__init__(self.message)


def f1(x,y):
    return np.abs(x-y)

def meanShift(x, strength=25):
    x = x.copy()
    while True:
        a = x.copy()
        change = False
        for i, _ in enumerate(x):
            x[i] = np.mean(a[np.where(f1(a,x[i]) < strength)])
            if x[i] != a[i]:
                change = True
        if not change:
            break
    return x

def create_table(num, ids, strength_x = 5, strength_y = 50, add__whitespace_supports = False):
    x = meanShift(num.T[2], strength_x)
    y = meanShift(num.T[1], strength_y)

    x_l = len(np.unique(x))
    y_l = len(np.unique(y))

    table = np.zeros((x_l, y_l), np.dtypes.StrDType)

    for i, a in enumerate(np.unique(x)):
        for j, b in enumerate(np.unique(y)):
            if table[i, j] == 0:
                table[i, j] = ''
            table[i, j] += ' '.join(ids.iloc[num.T[0][np.where((x == a) & (y == b))].astype(int)]['text'].tolist())
            if add__whitespace_supports and table[i, j] == '':
                table[i, j] = '-'

    return table

def get_numpy_table(ids, strength = (5, 50), add__whitespace_supports=False):
    num = np.zeros((len(ids), 3))

    i = -1
    for j, row in ids.iterrows():
        i += 1
        num[i][0] = j
        num[i][1] = row.left
        num[i][2] = row.top

    return create_table(num, ids, strength[0], strength[1], add__whitespace_supports)

def table2string(tab):
    text_out = ""
    for i, x in enumerate(tab):
        line = " ".join(x)
        text_out += line + "\n"
    return text_out

def get_pages_text(pymupdf_page, strength = (5, 50), add__whitespace_supports=False):
    d = pymupdf_page.get_text("words")
    w, h = pymupdf_page.cropbox.width, pymupdf_page.cropbox.height
    out = pd.DataFrame(d)

    if len(out) == 0:
        # no content on page found
        raise EmptyPageException(pymupdf_page.number, "Page is empty.")

    out['left'] = (out[0] / w)*1000 #+ (out[2] - out[0])/2
    out['top'] = (out[3] / h)*1000 #+ (out[1] - out[3])/2
    out['text'] = out[4]
    table = get_numpy_table(out, strength, add__whitespace_supports)
    
    return table

def pdf2img(pymupdf_page):
    zoom=5
    mat = pymupdf.Matrix(zoom, zoom)
    w, h = pymupdf_page.cropbox.width, pymupdf_page.cropbox.height
    pix = pymupdf_page.get_pixmap(matrix = mat)
    img = np.array(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    return img

def get_pages_tesseract(img, tesseract_params=None, strength = (5, 50),add__whitespace_supports=False):
    if not TESSERACT_AVAIL:
        print("pytesseract not installed, can't use tesseract")
        return []
    
    if tesseract_params == None:
        tesseract_params = {'lang': 'eng', 'config': '--psm 11'}

    d = pytesseract.image_to_data(img, output_type=Output.DICT, **tesseract_params)
    out = pd.DataFrame(d)

    if len(out) == 0:
        # no content on page found
        raise EmptyPageException(-1, "Page is empty.")
    
    out['left'] = (out['left'] / out.iloc[0]['width'])*1000
    out['top'] = (out['top'] / out.iloc[0]['height'])*1000

    return get_numpy_table(out, strength, add__whitespace_supports)
    

def align(input_path: str, force_tesseract: bool = False, tesseract_params: Optional[dict] = None, strength: Optional[Tuple[int, int]] = (5, 50), add__whitespace_supports: bool = False):
    """
    This function is used to extract text from images or pdfs
    It uses either tesseract or pymupdf to handle pdf text extraction.

    It will align the text in a human readable way. This can be very useful for downstream tasks or inputs to llms.
    Especially if you are dealing with invoices or data that contains big tables this will align and output the table in a readable way.

    This method is extremly simple on purpose as many tools try to do clever things that do more harm then good, by for example introducing
    deliminators at the wrong place.
    """ 
    
    if isinstance(input_path, str):

        # handle file as if it is a pdf
        if ".pdf" in input_path or ".PDF" in input_path:
            doc = pymupdf.open(input_path)
            output = []
            for page in doc:
                try:
                    if (len(page.get_text()) < 20 and TESSERACT_AVAIL) or force_tesseract: # use tesseract as this page doesnt seem to contain any text
                        img = pdf2img(page)
                        table = get_pages_tesseract(img, tesseract_params, strength=strength, add__whitespace_supports=add__whitespace_supports)
                    else:
                        table = get_pages_text(page, strength=strength, add__whitespace_supports=add__whitespace_supports)
                    output.append({"text": table2string(table), "dataframe": pd.DataFrame(table)})
                except EmptyPageException as e:
                    output.append({"text": None, "dataframe": None})
            return output

        # handle file as if it is a single image
        if not TESSERACT_AVAIL:
            raise "pytesseract is not installed, can't use tesseract"
        
        input_path = Image.open(input_path) 
    # we assume if a non string is passed it is most likely an image object that we can pass to pytesseract
    table = get_pages_tesseract(input_path, tesseract_params=tesseract_params, add__whitespace_supports=add__whitespace_supports)
    return {"text": table2string(table), "dataframe": pd.DataFrame(table)}