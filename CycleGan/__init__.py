_dsbuilder = None
_dsloader = None
_imgcnv = None
_model = None

def GetDsBuilder():
	"""データセットビルダーの取得.
	# Returns:
		データセットビルダ.
	"""
	global _dsbuilder
	if _dsbuilder is None:
		from . import dsbuilder
		_dsbuilder = dsbuilder
	return _dsbuilder

def GetDsLoader():
	"""データセットローダーの取得.
	# Returns:
		データセットローダー.
	"""
	global _dsloader
	if _dsloader is None:
		from . import dsloader
		_dsloader = dsloader
	return _dsloader

def GetImgCnv():
	"""イメージコンバーターの取得.
	# Returns:
		イメージコンバーター.
	"""
	global _imgcnv
	if _imgcnv is None:
		from . import imgcnv
		_imgcnv = imgcnv
	return _imgcnv

def CreateModel(name, in_ch, out_ch, batch_size):
	"""モデルの取得.
	# Args:
		name: モデル名、学習済みデータ保存時にファイル名に付与される.
		in_ch: 入力CH数.
		out_ch: 出力CH数.
		batch_size: バッチサイズ.
	# Returns:
		モデル.
	"""
	global _model
	if _model is None:
		from . import model
		_model = model
	return _model.CycleGan(name, in_ch, out_ch, batch_size)
