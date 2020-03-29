# audioCodec
 A class to encode and decode wav file using filter banks and psychoacoustic models

# model
![Model](drawing.svg "Model")

# requirements
* [**numpy**](https://www.numpy.org)
* [**scipy**](https://www.scipy.org) (dependencies on scipy will be removed)

# usage
```python
from audioCodec import codec

wavFile = 't16.wav' # input audio file
outBin = 'out.bin' # binary file name to store quantized signal in
outWav = 'reconstructed.wav' # wav file name for reconstricted signal
alpha=1. # tonality index
nSubband=256 # number of filter bank subbands "taps"

myObj = codec(wavFile,outBin,outWav,alpha,nSubband)

myObj.applyAnalyzer() # apply analyzer
# myObj.analyzedFrame will get you can get the analyzed signal 

myObj.applySynthesiser() # apply synthesiser
myObj.reconsctucedSignal # to get the output after reconstruction

```

# todo
* add psychoacoustic
* add quantization
* remove scipy

