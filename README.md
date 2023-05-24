# PrivaTree: An algorithm for training differentially-private decision trees
This library contains the implementation of PrivaTree and several other methods for training differentially-private decision trees for binary classification. These models protect the model from leaking information about training data in exchange for a potential drop in utility. This privacy-utility trade-off is decided by a parameter $\epsilon > 0$, where $\epsilon = \infty$ achieves no privacy but high utility, and lower numbers achieve more privacy. PrivaTree consistently outperforms other methods as it uses the privacy budget set by $\epsilon$ more efficiently than other works, and finds a balance between making accurate splits and generating accurate leaf labels.

_PrivaTree was developed and implemented by Daniël Vos with the help of Jelle Vos, Tianyu Li, Zekeriya Erkin, and Sicco Verwer. The accompanying paper can be found on [arxiv]()._

## Installing
To install the required dependencies run (preferably in a virtual environment):
```
pip install -r requirements.txt
```
PrivaTree requires a python version >= 3.7.

Experiment code can be found in the base directory of the repository.

## Performance
![comparison_other_work_adult](https://github.com/daniel-vos/differential-privacy/assets/1685648/95d48bc7-99bd-46f5-9180-1036db0bef0e)

In the figure above, PrivaTree can be seen to outperform other works on the _UCI adult_ dataset, averaged over 50 iterations. 'Decision tree' refers to a non-private decision tree. Diffprivlib refers to IBM's [differential privacy library](https://diffprivlib.readthedocs.io/en/latest/). The other works are also implemented in this repository.

## API
The API mostly reflects that of scikit learn, but we note that this library is still under development.

The main implementation can be found under `privatree/privatree.py`.
