# from sound import *
import numpy as np
import wave
import struct
import pickle

class codec:
    '''
    A class to encode and decode wav file using filter banks and psycoacoustic models
    '''
    def __init__(self,wavFile,nSubband=256):
        '''
        :param wavfile:  input wav file
        :param outBin:   output binary file
        :param outWav:   output wav file
        :param alpha:    tonality index
        :param nSubband: number of filter bank subbands
        '''
        self.wavFile = wave.open(wavFile, 'r') 
        self.samplingRate = self.wavFile.getframerate() # get frame rate
        self.nChannels = self.wavFile.getnchannels() # get number of channels (1,2)
        self.nFrames = self.wavFile.getnframes() # get number of frames
        self.width = self.wavFile.getsampwidth()
        data=self.wavFile.readframes(self.nFrames) # temp storage for frames
        self.frames =  np.array(struct.unpack('h' *(self.nFrames*self.width) ,data)).reshape(-1,self.nChannels) # get the audio frames 
        self.nSubband = nSubband
        self._analyzed = False
        self._synthesised = False
        self._psychoApplied = False
        self._quantized = False

    def applyAnalyzer(self):
        '''
        Filter bank analyzer part in transmitter
        '''
        
        self.filterLength=2*self.nSubband #filter bank length "length of each filter tap"
        n= np.arange(self.filterLength)
        k=np.arange(self.nSubband)
        self.windoFun= np.sin(np.pi/self.filterLength *(n+0.5))  #Window function


        temp1=np.cos((np.pi/self.nSubband*(k+0.5)).reshape(self.nSubband,1).dot((n+0.5- self.nSubband/2).reshape(1,self.filterLength)))
        temp2 = self.windoFun * np.sqrt(2/self.nSubband) # temporary variable for creating filter coefficients
        self.filterBank = np.einsum('i,ji->ji',temp2 , temp1) # filter coefficients

        # applying filter bank to audio input signal
        if self.nChannels == 1:
            self.analyzedFrames = np.array([np.convolve(i,self.frames)[0:self.nFrames][::self.nSubband] for i in self.filterBank  ]) # appltyin MDCT and downsampling the output
        elif self.nChannels == 2:
            Y0 = np.array([np.convolve(i,self.frames[:,0])[0:self.nFrames][::self.nSubband] for i in self.filterBank  ]) # appltyin MDCT and downsampling the output for the 1st channel
            Y1 = np.array([np.convolve(i,self.frames[:,1])[0:self.nFrames][::self.nSubband] for i in self.filterBank  ]) # appltyin MDCT and downsampling the output for the 2nd channel
            self.analyzedFrames = np.array([Y0,Y1])
        self._analyzed = True
        
    def applyPsychoacoustic(self,alpha=1.,w = 0.5):
        if not self._analyzed:
            raise RuntimeError("Signal must be analyezed before applying psychoacoustic model")

        # creating mapping matrix W which consist of ones for each Bark subband and zeros otherwise
        self.alpha = alpha
        self.barkSubband = int(24/w) # get the number of bark scale subbands based on selected width
        samplingRateBark  = 6. * np.arcsinh(self.samplingRate/600)  # transform sampling rate from frequency domain to bark domain
        barkSpacing = samplingRateBark/(self.barkSubband-1)
        frequencyBins = np.linspace(0,self.samplingRate/2,self.nSubband)
        barkBins = 6. * np.arcsinh(frequencyBins/600)
        self.mappingToBarkMat = np.zeros((self.barkSubband, self.nSubband)) # 
        self.mappingToBarkMat[np.around(barkBins/barkSpacing).astype(int)  ,np.arange(self.nSubband) ] = 1
        self.mappingFromBarkMat = np.nan_to_num((np.diag(1/np.sum(self.mappingToBarkMat,1))**0.5).dot(self.mappingToBarkMat)).T
        spreadingFuncMat = self._spreadingFunctionMat()
        LTQ=np.clip((3.64*(frequencyBins/1000.)**-0.8 -6.5*np.exp(-0.6*(frequencyBins/1000.-3.3)**2.)+1e-3*((frequencyBins/1000.)**4.)),-20,80)
        if self.nChannels == 1:
            self.barkedSignal = (np.dot( np.abs(self.analyzedFrames.T)**2.0, self.mappingToBarkMat.T)).T**(0.5)
            self.maskThresholdBark = (np.dot(self.barkedSignal.T**alpha, spreadingFuncMat**alpha)**(1.0/alpha)).T
            self.maskThresholdDFT = np.dot(self.maskThresholdBark.T, self.mappingFromBarkMat.T).T
            self.maskThresholdDFTQ = np.maximum(10.0**((LTQ.reshape(-1,1)-60)/20),self.maskThresholdDFT)
            self.quantizationStep = (12*self.maskThresholdDFTQ**2)**0.5
            self.quantizationStepBark = (np.dot( np.abs(self.quantizationStep.T)**2.0, self.mappingToBarkMat.T)).T**(0.5)
            self.scaleFactors = np.log2(self.quantizationStepBark)*4
        elif self.nChannels ==2:
            barkedSignal1 = (np.dot( np.abs(self.analyzedFrames[0].T)**2.0, self.mappingToBarkMat.T)).T**(0.5)
            barkedSignal2 = (np.dot( np.abs(self.analyzedFrames[1].T)**2.0, self.mappingToBarkMat.T)).T**(0.5)
            self.barkedSignal = np.array([barkedSignal1,barkedSignal2])

            maskThresholdBark1 = (np.dot(barkedSignal1.T**alpha, spreadingFuncMat**alpha)**(1.0/alpha)).T
            maskThresholdBark2 = (np.dot(barkedSignal2.T**alpha, spreadingFuncMat**alpha)**(1.0/alpha)).T
            self.maskThresholdBark = np.array([maskThresholdBark1,maskThresholdBark2])

            maskThresholdDFT1 = np.dot(maskThresholdBark1.T, self.mappingFromBarkMat.T).T
            maskThresholdDFT2 = np.dot(maskThresholdBark2.T, self.mappingFromBarkMat.T).T            
            self.maskThresholdDFT = np.array([maskThresholdDFT1,maskThresholdDFT2])

            maskThresholdDFTQ1 = np.maximum(10.0**((LTQ.reshape(-1,1)-60)/20),maskThresholdDFT1)
            maskThresholdDFTQ2 = np.maximum(10.0**((LTQ.reshape(-1,1)-60)/20),maskThresholdDFT2)
            self.maskThresholdDFTQ = np.array([maskThresholdDFTQ1,maskThresholdDFTQ2])

            quantizationStep1 = (12*maskThresholdDFTQ1**2)**0.5
            quantizationStep2 = (12*maskThresholdDFTQ2**2)**0.5
            self.quantizationStep = np.array([quantizationStep1,quantizationStep2])

            quantizationStepBark1 = (np.dot( np.abs(quantizationStep1.T)**2.0, self.mappingToBarkMat.T)).T**(0.5)
            quantizationStepBark2 = (np.dot( np.abs(quantizationStep2.T)**2.0, self.mappingToBarkMat.T)).T**(0.5)
            self.quantizationStepBark = np.array([quantizationStepBark1,quantizationStepBark2])

            scaleFactors1 = np.log2(quantizationStepBark1)*4
            scaleFactors2 = np.log2(quantizationStepBark2)*4
            self.scaleFactors = np.array([scaleFactors1,scaleFactors2])

        self._psychoApplied = True


    def _spreadingFunctionMat(self):
        # from Prof schuller implementation
        maxbark= 6. * np.arcsinh(self.samplingRate/2/600) #upper end of our Bark scale:22 Bark at 16 kHz

        spreadingfunctionBarkdB=np.zeros(2*self.barkSubband)
        spreadingfunctionBarkdB[0:self.barkSubband]=np.linspace(-maxbark*27,-8,self.barkSubband)-23.5
        spreadingfunctionBarkdB[self.barkSubband:2*self.barkSubband]=np.linspace(0,-maxbark*12.0,self.barkSubband)-23.5
        spreadingfunctionBarkVoltage=10.0**(spreadingfunctionBarkdB/20.0*self.alpha)
        spreadingfuncmatrix=np.zeros((self.barkSubband,self.barkSubband))
        for k in range(self.barkSubband):
            spreadingfuncmatrix[k,:]=spreadingfunctionBarkVoltage[(self.barkSubband-k):(2*self.barkSubband-k)]
        return spreadingfuncmatrix

    def quantize(self):
        if not self._psychoApplied:
            raise RuntimeError("Psychoacoustic processing must be applied before quantization")

        self.quantizedSignal = np.round(self.analyzedFrames/self.quantizationStep)
        self._quantized = True

    def dequantize(self):
        ''' dequantization using scaleFactors'''

        if not self._quantized:
            raise RuntimeError("quantization must be applied before dequantization")

        SF = 2**(self.scaleFactors/4)
        if self.nChannels ==1:
            self.recoveredStep = np.dot(SF.T, self.mappingFromBarkMat.T).T
        elif self.nChannels ==2:
            recoveredStep1 = np.dot(SF[0].T, self.mappingFromBarkMat.T).T
            recoveredStep2 = np.dot(SF[1].T, self.mappingFromBarkMat.T).T
            self.recoveredStep = np.array([recoveredStep1,recoveredStep2])
        self.dequantizedSignal = self.quantizedSignal * self.recoveredStep
        self.analyzedFrames = self.dequantizedSignal


    def applySynthesiser(self):
        '''
        Filter bank synthesiser part in receiver
        '''
        if not self._analyzed:
            raise RuntimeError("Signal must be analyezed before synthesised")
        self._synthesised = True
        G = np.flip(self.filterBank,1) # the impulse response of the receiver filter bank the just the flipping the trasmitter filter bank

        if self.nChannels == 1:
            # upsampling
            Y_upsampled = np.zeros((self.analyzedFrames.shape[0],self.analyzedFrames.shape[1]*self.nSubband))
            Y_upsampled[:,::self.nSubband] = self.analyzedFrames 
            x_tilde = np.array([np.convolve(i,j)[0:self.nFrames] for i,j in zip(G,Y_upsampled)  ]) # applyin inverse-MDCT
            self.reconsctucedSignal = np.sum(x_tilde, axis=0)  # Perfectly reconstructed signal

        elif self.nChannels == 2:
            # upsampling 2 channels
            Y_upsampled = np.zeros((2,self.analyzedFrames.shape[1],self.analyzedFrames.shape[2]*self.nSubband))
            Y_upsampled[0][:,::self.nSubband] = self.analyzedFrames[0]
            Y_upsampled[1][:,::self.nSubband] = self.analyzedFrames[1]
            x_tilde0 = np.array([np.convolve(i,j)[0:self.nFrames] for i,j in zip(G,Y_upsampled[0])  ]) # applyin inverse-MDCT
            x_tilde0 = np.sum(x_tilde0,axis=0) # Perfectly reconstructed ch1
            x_tilde1 = np.array([np.convolve(i,j)[0:self.nFrames] for i,j in zip(G,Y_upsampled[1])  ]) # applyin inverse-MDCT
            x_tilde1 = np.sum(x_tilde1,axis=0) # Perfectly reconstructed ch2
            self.reconsctucedSignal = np.array([x_tilde0,x_tilde1]).T  # Perfectly reconstructed signal
        self._synthesised = True

    def binaryWrite(self,outBin):
        if not self._analyzed:
            raise RuntimeError('Signal must be analyzed first')
        binFile = open(outBin, 'wb')
        pickle.dump(self.analyzedFrames,binFile ) 
        binFile.close()

    def wavWrite(self,outWav):
        if not self._synthesised:
            raise RuntimeError('Signal must be synthesised to be saved in wav file')
        snd = self.reconsctucedSignal.flatten().astype(int)
        length=len(snd)
        wf = wave.open(outWav, 'wb')
        wf.setnchannels(self.nChannels)
        wf.setsampwidth(self.width)
        wf.setframerate(self.samplingRate)
        data=struct.pack( 'h' * length, *snd )
        wf.writeframes(data)
        wf.close()