#### Practical work on information theory

*Task*: Compute various metrics upon provided data about transmission channel

#### How to run solution:
```bash
git clone foobar
cd foobar

# vi or any other editor of your choice
vi test_data.json # edit test data

# NOTE: keep data format as it is

# init virtualenv
python3 -m virtualenv venv
source venv

# install dependencies (numpy, essentially :) )
python3 -m pip install -r requirements.txt

# run solution on test data
python3 main.py test_data.json
```