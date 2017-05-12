"""ディープラーニング関係ヘルパ、テンソルのCPU↔GPU変換処理など.
"""
import os

# テンソル内スカラーの型、そのうち Chainer が FP16 対応したらそっち使いたい
dtype = None
# テンソル計算モジュール、numpy または cupy
xp = None
# Chainer2.0以降で廃止予定らしいのでここに宣言しておく
test = False
# 学習済みデータ保存先ディレクトリ名
traindDataDir = "traineddata"
# 評価データ保存先ディレクトリ名
evalDataDir = "evaldata"

_cupy = None
_cuda = None
_chainer = None


def Init(gpu):
	"""環境を初期化する.
	# Args:
		gpu: 使用するGPUインデックス、負数ならGPU未使用となる.
	"""
	# 複数回の初期化は想定しない
	if not (xp is None):
		return

	global dtype
	global xp
	global np
	global _cupy
	global _cuda
	global _chainer

	import chainer
	_chainer = chainer

	# 必要に応じてGPU使用の初期化を行う
	if 0 <= gpu:
		print("Using cuda device {}.".format(gpu))
		from chainer import cuda
		import cupy
		_cuda = cuda
		_cupy = cupy

		cuda.get_device(gpu).use()
		# cuda.set_max_workspace_size(64 * 1024 * 1024)  # 64MB
		xp = cupy
	else:
		print("Using numpy.")
		import numpy as np
		xp = np

	# float16 使いたいがまだ未対応らしいため float32 を使う
	dtype = xp.float32

def ToGpu(x):
	"""GPU利用可能状態ならGPUメモリオブジェクトに変換する.
	# Args:
		x: 変換対象オブジェクト.
	# Returns:
		変換後のオブジェクト.
	"""
	if xp is _cupy:
		if isinstance(x, _chainer.Chain):
			return x.to_gpu()
		elif isinstance(x, _chainer.Optimizer):
			return x
		else:
			return _cuda.to_gpu(x)
	else:
		return x

def ToCpu(x):
	"""GPU利用可能状態ならCPUメモリオブジェクトに変換する.
	# Args:
		x: 変換対象オブジェクト.
	# Returns:
		変換後のオブジェクト.
	"""
	if xp is _cupy:
		if isinstance(x, _chainer.Chain):
			return x.to_cpu()
		elif isinstance(x, _chainer.Optimizer):
			return x
		else:
			return _cuda.to_cpu(x)
	else:
		return x

def SaveTrainedData(name, objsDic, ext):
	"""学習済みのオブジェクトを所定のディレクトリへ保存する.
	# Args:
		objsDic: オブジェクト名をキーとするオブジェクトデータ配列、キーはファイル名の一部となる.
		ext: ファイル名の拡張子.
	"""
	if not os.path.isdir(traindDataDir):
		os.mkdir(traindDataDir)
	serializers = _chainer.serializers
	for key, chain in objsDic.items():
		chainFile = os.path.join(traindDataDir, name + ("." + key if len(key) != 0 else "") + ext)
		serializers.save_npz(chainFile, ToCpu(chain))

def LoadTrainedData(name, objsDic, ext):
	"""学習済みのオブジェクトを所定のディレクトリから読み込む.
	# Args:
		objsDic: オブジェクト名をキーとするオブジェクトデータ配列、キーはファイル名の一部.
		ext: ファイル名の拡張子.
	"""
	serializers = _chainer.serializers
	for key, chain in objsDic.items():
		chainFile = os.path.join(traindDataDir, name + ("." + key if len(key) != 0 else "") + ext)
		if os.path.isfile(chainFile):
			serializers.load_npz(chainFile, chain)

def SaveChains(name, chainsDic):
	"""学習済みの chainer.Chain 派生オブジェクトを所定のディレクトリへ保存する.
	# Args:
		objsDic: オブジェクト名をキーとするオブジェクトデータ配列、キーはファイル名の一部となる.
	"""
	SaveTrainedData(name, chainsDic, ".chain")

def LoadChains(name, chainsDic):
	"""学習済みの chainer.Chain 派生オブジェクトを所定のディレクトリから読み込む.
	# Args:
		objsDic: オブジェクト名をキーとするオブジェクトデータ配列、キーはファイル名の一部となる.
	"""
	LoadTrainedData(name, chainsDic, ".chain")

def SaveOptimizers(name, optimizersDic):
	"""学習済みの chainer.Optimizer 派生オブジェクトを所定のディレクトリへ保存する.
	# Args:
		objsDic: オブジェクト名をキーとするオブジェクトデータ配列、キーはファイル名の一部となる.
	"""
	SaveTrainedData(name, optimizersDic, ".optimizer")

def LoadOptimizers(name, optimizersDic):
	"""学習済みの chainer.Optimizer 派生オブジェクトを所定のディレクトリから読み込む.
	# Args:
		objsDic: オブジェクト名をキーとするオブジェクトデータ配列、キーはファイル名の一部となる.
	"""
	LoadTrainedData(name, optimizersDic, ".optimizer")

def GetEvalDataDir():
	"""評価データ保存先ディレクトリ名の取得、もしディレクトリが存在しなかったら自動で作成される.
	# Returns:
		評価データ保存先ディレクトリ名.
	"""
	if not os.path.isdir(evalDataDir):
		os.mkdir(evalDataDir)
	return evalDataDir


if __name__ == "__main__":
	Init(0)
