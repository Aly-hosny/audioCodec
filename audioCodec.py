# from sound import *
import numpy as np
import wave
import struct

class codec:
    '''
    A class to encode and decode wav file using filter banks and psycoacoustic models
    '''
    def __init__(self,wavFile,outBin,outWav,alpha=1.,nSubband=256):
        '''
        :param wavfile:  input wav file
        :param outBin:   output binary file
        :param outWav:   output wav file
        :param alpha:    tonality index
        :param nSubband: number of filter bank subbands
        '''
        self.wavFile = wave.open(wavFile, 'r')
        self.samplingRate = self.wavFile.getframerate()
        self.nChannels = self.wavFile.getnchannels()
        self.nFrames = self.wavFile.getnframes()
        self.width = self.wavFile.getsampwidth()
        data=self.wavFile.readframes(self.nFrames) # temp storage for frames
        self.frames =  np.array(struct.unpack('h' *(self.nFrames*self.width) ,data)).reshape(-1,self.nChannels) # get the audio frames 
        self.nSubband = nSubband
        self.outBin = outBin
        self.outWav = outWav
        self.alpha = alpha


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
        


    def applySynthesiser(self):
        '''
        Filter bank synthesiser part in receiver
        '''
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