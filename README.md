# pdfalign
[![PyPI](https://img.shields.io/pypi/v/pdfalign.svg)](https://pypi.org/project/pdfalign/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pdfalign)](https://img.shields.io/pypi/dm/pdfalign)
![GitHub](https://img.shields.io/github/license/y-hesse/pdfalign.svg)

pdfalign is a very simple tool to extract text from a pdf in a grid aligned format. This is especially useful in table extraction pipelines.
Some use cases include invoice data extraction and pdf processing for rag systems.
The simple algorithm that pdfalign uses is the meanshift algorithm to group text blocks together and to algin grids. In some scenarios this may produce very sparse documents / tests.
In most cases this is however not a problem for llms to handle

## Features
- **Tesseract support**: Can use tesseract automatically, if no text is detected on a pdf page.
- **Image support**: Not only works with pdfs but also with for example scan copys of invoices.
- **Output as a pandas dataframe**: One of the outputs is a pandas dataframe which can be used for further processing / transformations.

## Installation

Install pdfalign using pip:

```bash
pip install pdfalign
```

## Usage
Here's a quick example on how to use PDFalign to extract grid aligned text from a PDF file:
```python
from pdfalgin import align

# Extract grid algined text from a pdf
# which returns a list for each pdf page
pages = algin("sample.pdf", force_tesseract=False, tesseract_params=None)

for page in pages:
    print(page['text'])
    print(page['dataframe'])

```
If you want to use an `Image` you have to make sure that `pytesseract` is installed. Here is the code for that:
```python
from pdfalgin import algin

# Initialize an Image object
img = Image("sample.jpg")

# Extract all tables from the image
# which returns a list of Table objects
text, dataframe = algin(img)
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.