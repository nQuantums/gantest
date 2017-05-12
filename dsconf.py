"""データセットの場所、画像サイズ、CH数などの設定.
"""
import os

_dsdir = "datasets"

MaxImageSize = (256, 256)
InChs = 3
OutChs = 3

def GetDir():
	"""データセットディレクトリ名の取得、もしディレクトリが存在しなかったら自動で作成される.
	# Returns:
		データセットディレクトリ名.
	"""
	if not os.path.isdir(_dsdir):
		os.mkdir(_dsdir)
	return _dsdir
