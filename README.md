gantest
===
Chainerによる学習処理の叩き台。  
現状CycleGANとpix2pixが入ってます。

## Description
CNNを試そうとすると大体同じような処理になるので、  
色々なパターンに対応できる叩き台が欲しいと思い作りました。

pix2pixは [https://github.com/pfnet-research/chainer-pix2pix](https://github.com/pfnet-research/chainer-pix2pix) を参考に、  
CycleGANは [https://github.com/Aixile/chainer-cyclegan](https://github.com/Aixile/chainer-cyclegan) を参考にさせて頂きました。  
有用なコードを有難うございます。

## Demo
![](./doc/fig1.jpg)

## Requirement
- [Anaconda3](https://www.continuum.io/downloads) 4.2 [Windows 64bit](https://repo.continuum.io/archive/Anaconda3-4.2.0-Windows-x86_64.exe) / 
 [32bit](https://repo.continuum.io/archive/Anaconda3-4.2.0-Windows-x86.exe)、[Linux 64bit](https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh) / [32bit](https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86.sh)、[MacOS X](https://repo.continuum.io/archive/Anaconda3-4.2.0-MacOSX-x86_64.sh)  
- [CUDA8](https://developer.nvidia.com/cuda-downloads) ※ChainerでGPU使うなら必要
- [cuDNN5](https://developer.nvidia.com/cudnn) ※ChainerでGPU使うなら必要
- VisualStudio2015 ※CPythonやらCUDAコンパイラで必要になる
- [Chainer](https://github.com/pfnet/chainer) 1.23
- OpenCV3

## Usage

	python ./train.py CycleGAN

## Contribution

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## Author

[nQuantums](https://github.com/nQuantums)
