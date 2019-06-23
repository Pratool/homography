In order to run the homography application, you will need:
- Python 3.5+
    - the latest released python3 package on most linux distributions' package
      managers should suffice
- matplotlib, numpy PyPI packages
    - both of these can be installed from pip (be sure to use pip3 if python2 is
      the default python interpreter!)
    - my personal recommendation is to create a python3 virtual environment for
      installed packages
- a PNG image
- a JSON-formatted correspondences configuration file
    - there should be exactly four correspondences
    - the first of an array will always match the input image point
    - the second of an array will always match the destination image point
    - see configs/t4_onto_t5.json for a specific example

The command line tool can be invoked with:
`python3 homography.py --help`

For example, this command takes an image as input and transforms it with the
correspondences provided and the output is layerd on top of a background image.
This is a demonstration of the "billboard" effect. Note that none of the iamges
have been provided.
```python homography.py --input-image media/t4.png \
    --correspondences configs/t4_onto_t5_2.json \
    --output-image test02.png \
    --background media/t5.png```
