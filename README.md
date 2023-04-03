# Snip_With_Ocr_Your_Own

easyocr을 이용한 캡쳐도구 제작

## Installation & Usage

### Stand-alone (Recommended)
1. Download at the [google drive](https://drive.google.com/file/d/1RnwuJ8uxA8KGJMFjFPa9xEQYDBNY3MCH/view?usp=share_link) (Recommended)
2. 압축해제 후 snip_onnx.exe 파일 클릭 or 해당 파일을 작업표시줄에 등록

### Run the python code (Needs python installation)
1. Clone the repo
```bash
git clone https://github.com/fryegg/snip_with_ocr_your_own.git
```
2. Install the required library
```bash
pip install requirements.txt
```
3. Download the detection and recognition model at this [google drive](https://drive.google.com/drive/folders/1n_LOrJHkMVcZhyCgg37PYMAcsJ7_Sxsn?usp=share_link)
Place two kinds of model at the "onnx_models" folder 

4. Run the python code "snip_onnx.py" or "snip.py"
```python
python snip.py OR python snip_onnx.py
```

## Reference

* I highly copy the code easy_ocr_only_onnx folder and onnx model from this [repo](https://github.com/Kromtar/EasyOCR-ONNX.git)
* EasyOCR is originated from this [repo](https://github.com/JaidedAI/EasyOCR)

## License

[MIT](https://choosealicense.com/licenses/mit/)
