# Multimodal Eigenwords

Python implementation for [Multimodal Eigenwords](https://aclanthology.info/papers/W17-2405/w17-2405), which extends [Eigenwords](http://jmlr.org/papers/v16/dhillon15a.html) for multimodal word embedding.

<div style="white-space: nowrap;">
  <img src="https://github.com/kafku/mmeigenwords/blob/master/misc/model.png" height="180">
  <img src="https://github.com/kafku/mmeigenwords/blob/master/misc/example.png" height="180">
</div>



## Prerequisits

- Anaconda3
  - numpy >= 1.15.4
  - scipy >= 1.1.0
  - scikit-learn >= 0.20.1
- h5py >= 2.9.0
- more-itertools >= 4.3.0
- tqdm >= 4.30.0
- dask >= 1.1.1
- gensim >= 3.5.0
- imageio >= 2.4.1
- openbals (suppose it's installed using `conda`)
- g++ >= 4.8.5


## Usage

See the demo for [Multimodal Eigenwords](https://github.com/kafku/mmeigenwords/blob/master/mmeigenwords_demo.ipynb).
Before you run scripts on the notebook, you need to conduct the following steps.

```bash
# compile cpp source
cd src/
make

cd ../data
# download input files (corpus, image features, etc.)
./download_inputs.sh

# download images
# Note that this may take some time, and that some images may have been removed form flickr
./download_images.sh
```

## Citation

```
@InProceedings{W17-2405,
  author = "Fukui, Kazuki and Oshikiri, Takamasa and Shimodaira, Hidetoshi",
  title = "Spectral Graph-Based Method of Multimodal Word Embedding",
  booktitle = "Proceedings of TextGraphs-11: the Workshop on Graph-based Methods for Natural Language Processing",
  year = "2017",
  publisher = "Association for Computational Linguistics",
  pages = "39--44",
  location = "Vancouver, Canada",
  doi = "10.18653/v1/W17-2405",
  url = "http://aclweb.org/anthology/W17-2405"
}
```
