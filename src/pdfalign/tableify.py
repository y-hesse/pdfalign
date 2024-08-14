'''

Author: Yannik Hesse
Github:
Date: 14.08.2024 

'''

import numpy as np
from typing import Optional
import pandas as pd
import pymupdf
from PIL import Image

TESSERACT_AVAIL = True

try:
    import pytesseract
    from pytesseract import Output
except ImportError:
    TESSERACT_AVAIL = False

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

def create_table(num, ids, strength_x = 5, strength_y = 50):
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

    return table

def get_numpy_table(ids):
    num = np.zeros((len(ids), 3))

    i = -1
    for j, row in ids.iterrows():
        i += 1
        num[i][0] = j
        num[i][1] = row.left
        num[i][2] = row.top

    return create_table(num, ids)

def table2string(tab):
    text_out = ""
    for x in tab:
        line = " ".join(x)
        text_out += line + "\n"
    return text_out

def get_pages_text(pymupdf_page):
    d = pymupdf_page.get_text("words")
    out = pd.DataFrame(d)
    out['left'] = out[0]
    out['top'] = out[3]
    out['text'] = out[4]
    table = get_numpy_table(out)
    
    return table

def pdf2img(pymupdf_page):
    zoom=5
    mat = pymupdf.Matrix(zoom, zoom)
    pix = pymupdf_page.get_pixmap(matrix = mat)
    img = np.array(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    return img

def get_pages_tesseract(img, tesseract_params=None):
    if not TESSERACT_AVAIL:
        print("pytesseract not installed, can't use tesseract")
        return []
    
    if tesseract_params == None:
        tesseract_params = {'lang': 'eng', 'config': '--psm 11'}

    d = pytesseract.image_to_data(img, output_type=Output.DICT, **tesseract_params)
    out = pd.DataFrame(d)

    return get_numpy_table(out)
    

def align(input_path: str, force_tesseract: bool = False, tesseract_params: Optional[dict] = None):
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
                if (len(page.get_text()) < 100 and TESSERACT_AVAIL) or force_tesseract: # use tesseract as this page doesnt seem to contain any text
                    img = pdf2img(page)
                    table = get_pages_tesseract(img, tesseract_params)
                else:
                    table = get_pages_text(page)
                output.append({"text": table2string(table), "dataframe": pd.DataFrame(table)})
            return output

        # handle file as if it is a single image
        if not TESSERACT_AVAIL:
            raise "pytesseract is not installed, can't use tesseract"
        
        input_path = Image.open(input_path) 
    # we assume if a non string is passed it is most likely an image object that we can pass to pytesseract
    table = get_pages_tesseract(input_path, tesseract_params)
    return {"text": table2string(table), "dataframe": pd.DataFrame(table)}