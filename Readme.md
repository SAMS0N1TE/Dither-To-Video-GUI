## Install Dependencies
You should probably use a venv
```
python -m venv venv
source venv/bin/activate
```
# On Windows:
```
venv\Scripts\activate
```
# Run:
```sh 
pip install -r requirements.txt
```

## Usage
You need to have [ffmpeg](https://www.ffmpeg.org/) installed and only `.mp4` files are supported.
You should place videos in the input folder, but you also do have the options of chosing your own directory. However it defaults to the iput and output folders. 

```
Run python src/main.py
```
