"""学習経過をサーバへアップロードする処理.
"""
import os

_uploaderCore = None

class Dummy():
	def __init__(self):
		"""
		"""

def Upload(file):
	"""指定ファイルを独自のプロトコルでサーバーにアップロードする.
	# Args:
		file: アップロードするファイルパス名.
	"""
	global _uploaderCore
	if _uploaderCore is None:
		if os.path.exists("uploaderCore.py"):
			import uploaderCore
			_uploaderCore = uploaderCore
		else:
			_uploaderCore = Dummy()
	elif isinstance(_uploaderCore, Dummy):
		return
	# アップロード方法は秘密
	if not _uploaderCore.Upload(file):
		_uploaderCore = Dummy()
