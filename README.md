# Notice
All .pickle files were excluded because you *really* shouldn't be running .pickle files from the internet. 

#Install
Run `./install.sh` before running.

#Running
Type `source env/bin/activate` after installing, and then type:

```
python3 train.py --help
```

For all the information you need.

Try `python3 train.py --file 'networks/best-nn.pickle' --random 1000` as an example, or even `python3 train.py --file 'data/r-v-r-2500-50.pickle' --graph 1`

#Tests
Using pytest (and whilsts inside the env) try `pytest -v neuron_tests.py`
